import torch
import torch.nn as nn
from .WaveModel import WaveModel
from .ConvLSTM import ConvLSTM
from .autoencoder import Encoder, Decoder

class WaveAEConvLSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: ConvLSTM with Autoencoder"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)
        
    def create_architecture(self):
        self.params = self.conf.convlstm
        self.params_ae = self.conf.autoencoder
        self.encoding_done = False
        
        # ConvLSTM params
        self.num_layers = self.params.num_layers
        self.in_channels = self.params.in_channels
        self.out_channels = self.params.out_channels
        self.kernel_size = self.params.kernel_size
        self.padding = self.params.padding
        self.spatial = self.params.spatial
        # AE params
        self.encoder_channels = self.params_ae.encoder_channels
        self.decoder_channels = self.params_ae.decoder_channels
        self.latent_dim = self.params_ae.latent_dim
        self.pretrained = self.params_ae.pretrained
        self.freeze_weights = self.params_ae.freeze_weights
        self.use_decoder = self.params_ae.use_decoder
        self.method = self.params_ae.method
        
        # get the correct sizes for AE
        in_channels, spatial = self.configure_ae()
        
        # Create single ConvLSTM layer
        self.arch = ConvLSTM(
            in_channels=in_channels,
            out_channels=self.out_channels,
            seq_len=self.seq_len,
            kernel_size=self.kernel_size,
            padding=self.padding,
            frame_size=(spatial, spatial)
        )
        
        # conv reduction + activation to arrive back at real/imag
        self.linear = nn.Sequential(
            nn.Conv2d(self.out_channels, 2, kernel_size=1),
            nn.Tanh())
        
    def forward(self, x, meta=None):
        self.encoding_done = False
        try:
            x = self.process_ae(x) # encoding step
        except Warning:
            pass
            
        # invoke for specified mode (i.e. many_to_many)
        lstm_out, meta = self.arch(x, meta, mode=self.io_mode, 
                                    autoregressive=self.autoreg)
        
        if self.use_decoder == True:
            preds = self.process_ae(x, lstm_out) # decoding step
        else: # no decoding
            preds = lstm_out

        return preds, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        
        # Forward pass
        preds, _ = self.forward(samples)
        
        # If not using decoder, encode the labels to match prediction space
        if self.use_decoder == False:
            self.encoding_done = False  # reset bc we need to encode again
            labels = labels.view(labels.size(0), labels.size(1), -1)
            labels = self.process_ae(labels)  # encode ground truths
            
            # Get the actual dimensions from the predictions
            pred_size = preds.size()
            #print(f"Prediction size: {pred_size}")  # Debug print
            
            # Reshape labels to match prediction dimensions
            labels = labels.view(batch_size, self.seq_len, -1)
            
            # Don't reshape predictions since they're already in encoded space
            preds = preds.view(batch_size, self.seq_len, -1)
        else:
            # Only reshape predictions if using decoder
            preds = preds.view(batch_size, self.seq_len, r_i, xdim, ydim)
        
        # Compute loss
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']
        
        return loss, preds
        
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        
        # Determine the mode based on dataloader_idx
        mode = 'valid' if dataloader_idx == 0 else 'train'
        
        # Handle predictions and labels based on decoder usage
        if self.use_decoder:
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()
        else:
            # When not using decoder, keep both in flattened encoded space
            batch_size, seq_len, encoded_dim = preds.size()
            preds_np = preds.detach().cpu().numpy()
            
            # Reshape labels to match prediction space
            labels = labels.view(labels.size(0), labels.size(1), -1)  # Flatten spatial dims
            self.encoding_done = False  # Reset encoding flag
            encoded_labels = self.process_ae(labels)  # Encode labels
            labels_np = encoded_labels.detach().cpu().numpy()
            
            # Verify shapes match
            assert preds_np.shape == labels_np.shape, f"Shape mismatch: preds {preds_np.shape} vs labels {labels_np.shape}"
        
        # Store results
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)
            
    def configure_ae(self):
        # Calculate size after each conv layer
        temp_spatial = self.spatial
        for _ in range(len(self.encoder_channels) - 1):
            # mimicking the downsampling to determine the reduced spatial size
            # (spatial + 2*padding - kernel_size) // stride + 1
            temp_spatial = ((temp_spatial + 2*1 - 3) // 2) + 1
        reduced_spatial = temp_spatial
        
        # Encoder: downsampling
        self.encoder = Encoder(
            channels=self.encoder_channels,
            conf=self.params_ae
        )
        
        # Decoder: upsampling
        if self.use_decoder == True:
            self.decoder = Decoder(
                channels=self.decoder_channels,
                conf=self.params_ae
            )
    
        if self.pretrained == True:
            # load pretrained autoencoder
            dirpath = '/develop/results/meep_meep/autoencoder/model_ae-v1/'
            checkpoint = torch.load(dirpath + "model.ckpt")
            encoder_state_dict = {}
            
            # Print architecture details
            '''print("Pretrained model architecture:")
            for key, value in checkpoint['state_dict'].items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                    
            # Update channels to match pretrained model
            self.encoder_channels = [2]  # Input channels
            current_channels = 2
            for key, value in checkpoint['state_dict'].items():
                if 'encoder' in key and 'weight' in key and len(value.shape) == 4:
                    out_channels = value.shape[0]
                    self.encoder_channels.append(out_channels)
                    current_channels = out_channels'''

            # extract layers for the encoder
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('encoder'):
                    encoder_state_dict[key.replace('encoder.', '')] = value
            
            # load state dict
            self.encoder.load_state_dict(encoder_state_dict)
            
            # freeze ae weights
            if self.freeze_weights == True:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                    
            # only if we're doing decoding now       
            if self.use_decoder == True:
                decoder_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    if key.startswith('decoder'):
                        decoder_state_dict[key.replace('decoder.', '')] = value    
                self.decoder.load_state_dict(decoder_state_dict)   
                # freeze ae weights
                if self.freeze_weights == True:
                    for param in self.encoder.parameters():
                        param.requires_grad = False       
                
        # different since representation space
        in_channels = self.encoder_channels[-1]
        spatial = reduced_spatial
        
        return in_channels, spatial
    
    def process_ae(self, x, lstm_out=None):
        if self.encoding_done == False: # encode the input
            # Handle both 5D and 3D input tensors
            if len(x.shape) == 5:
                batch, seq_len, r_i, xdim, ydim = x.size()
                if xdim != 166:
                    raise Warning(f"Input spatial size must be 166, got {xdim}.")
                x = x.view(batch, seq_len, 2, self.spatial, self.spatial)
            elif len(x.shape) == 3:
                batch, seq_len, flattened = x.size()
                # Reshape flattened dimension back to 2D spatial
                x = x.view(batch, seq_len, 2, self.spatial, self.spatial)
            else:
                raise ValueError(f"Expected 5D or 3D tensor, got shape {x.shape}")
                
            if lstm_out is None: # encoder
                # process sequence through encoder
                encoded_sequence = []
                if self.freeze_weights == True:
                    with torch.no_grad(): # no grad for pretrained ae
                        for t in range(seq_len):
                            # encode each t
                            encoded = self.encoder(x[:, t]) # [batch, latent_dim]
                            encoded_sequence.append(encoded)
                else: # trainable ae
                    for t in range(seq_len):
                        # encode each t
                        encoded = self.encoder(x[:, t]) # [batch, latent_dim]
                        encoded_sequence.append(encoded)
                # stack to get [batch, seq_len, latent_dim]
                encoded_sequence = torch.stack(encoded_sequence, dim=1)
                self.encoding_done = True
                return encoded_sequence
            
        elif lstm_out is not None: # decoder
            # decode outputted sequence
            decoded_sequence = []
            if self.freeze_weights == True:
                with torch.no_grad(): # no grad for pretrained ae
                    for t in range(lstm_out.size(1)):
                        # decode each t
                        decoded = self.decoder(lstm_out[:, t])
                        decoded_sequence.append(decoded)
            else: # trainable decoder
                for t in range(lstm_out.size(1)):
                    # decode each t
                    decoded = self.decoder(lstm_out[:, t])
                    decoded_sequence.append(decoded)
            preds = torch.stack(decoded_sequence, dim=1)
            
            return preds
        
        else: # already encoded
            return x