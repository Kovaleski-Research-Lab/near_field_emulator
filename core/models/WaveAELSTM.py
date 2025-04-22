import torch
import numpy as np
from .WavePropModel import WavePropModel
from .autoencoder import Encoder, Decoder

class WaveAELSTM(WavePropModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM with Autoencoder"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)
        
    def create_architecture(self):
        self.params = self.conf.lstm
        self.params_ae = self.conf.autoencoder
        self.encoding_done = False
        
        # ConvLSTM params
        self.num_layers = self.params.num_layers
        self.h_dims = self.params.h_dims
        # AE params
        self.i_dims = self.params_ae.latent_dim 
        self.encoder_channels = self.params_ae.encoder_channels
        self.decoder_channels = self.params_ae.decoder_channels
        self.latent_dim = self.params_ae.latent_dim
        self.pretrained = self.params_ae.pretrained
        self.freeze_weights = self.params_ae.freeze_weights
        self.use_decoder = self.params_ae.use_decoder
        self.method = self.params_ae.method
        
        # configure autoencoder to get reduced image
        _, _ = self.configure_ae()
        # flatten representation
                
        self.arch = torch.nn.LSTM(input_size=self.i_dims,
                                    hidden_size=self.h_dims,
                                    num_layers=self.num_layers,
                                    batch_first=True)
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.h_dims, self.i_dims),
            torch.nn.Tanh())
        
    def forward(self, x, meta=None):
        self.encoding_done = False
        batch, seq_len, input = x.size()
        x = self.process_ae(x) # encoding step
        if meta is None: # Init hidden and cell states if not provided
            h, c = self.init_hidden(batch)
            meta = (h, c)          
                
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        x = x.view(batch, seq_len, -1)
        
        predictions = [] # to store each successive pred as we pass through
        # prepare output tensor
        output = torch.zeros(batch, self.seq_len,
                            x.shape[2], device=x.device)
        
        if self.io_mode == 'one_to_many':
            # first timestep: use our single slice input
            lstm_out, meta = self.arch(x, meta)
            pred = self.linear(lstm_out)
            output[:, 0] = pred.squeeze(dim=1)
            predictions.append(pred) # t+1 (or + delta split)
            
            # remaining t: no input, we pass 0's as dummy vals
            for t in range(1, self.seq_len):
                dummy_input = torch.zeros_like(x)
                lstm_out, meta = self.arch(dummy_input, meta)
                pred = self.linear(lstm_out)
                output[:, t] = pred.squeeze(dim=1)
                predictions.append(pred)
            
        elif self.io_mode == 'many_to_many':
            if self.autoreg: # testing (probably)
                # Use first timestep
                current_input = x[:, 0]  # Keep seq_len dim with size 1
                current_input = current_input.unsqueeze(1)
                lstm_out, meta = self.arch(current_input, meta)
                pred = self.linear(lstm_out)
                output[:, 0] = pred.squeeze(dim=1)
                predictions.append(pred)
                
                # Generate remaining predictions using previous outputs
                for t in range(1, self.seq_len):
                    # Use previous prediction as input
                    lstm_out, meta = self.arch(pred, meta)
                    pred = self.linear(lstm_out)
                    output[:, t] = pred.squeeze(dim=1)
                    predictions.append(pred)
                    
            else: # teacher forcing
                for t in range(self.seq_len):
                    current_input = x[:, t] # ground truth at t
                    lstm_out, meta = self.arch(current_input, meta)
                    pred = self.linear(lstm_out)
                    output[:, t] = pred.squeeze(dim=1)
                    predictions.append(pred)
                
        else:
            # other io modes not currently implemented
            return NotImplementedError(f'Recurrent input-output mode "{self.io_mode}" is not implemented.')
        
        if self.use_decoder == True:
            predictions = self.process_ae(x, output) # decoding step
        else: # no decoding
            predictions = output
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        predictions = predictions.view(batch, self.seq_len, -1)
        
        return predictions, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        samples = samples.view(batch_size, input_seq_len, -1)
        labels = labels.view(batch_size, self.seq_len, -1)
        
        # Forward pass
        preds, _ = self.forward(samples)
        
        # format labels
        if self.use_decoder == False:
            self.encoding_done = False # reset bc we need to encode again
            labels = self.process_ae(labels) # encode ground truths
            # flatten latent space - (i.e. 512 -> 2, 16, 16)
            if self.method == 'linear':
                latent_spatial = np.sqrt(self.latent_dim/2).astype(int)
                xdim, ydim = latent_spatial, latent_spatial
            else:
                raise ValueError(f"Unsupported AE mode for WaveAELSTM: {self.method}")
            
        # Compute loss      
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']
        
        # reshape preds for metrics
        if self.io_mode == "one_to_many":
            preds = preds.view(batch_size, self.seq_len, r_i, xdim, ydim)
        elif self.io_mode == "many_to_many":
            preds = preds.view(batch_size, self.seq_len, r_i, xdim, ydim)
        else:
            # other modes not implemented
            raise NotImplementedError

        return loss, preds
        
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Determine the mode based on dataloader_idx
        if dataloader_idx == 0:
            mode = 'valid'
        elif dataloader_idx == 1:
            mode = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        
        # need to determine if we decoded to match preds and labels
        if self.use_decoder == False:
            self.encoding_done = False # reset bc we need to encode again
            labels = labels.view(labels.size(0), labels.size(1), -1)
            labels = self.process_ae(labels) # encode ground truths
            # flatten latent space - (i.e. 512 -> 2, 16, 16)
            # convert latent_dim to int
            if self.method == 'linear':
                latent_spatial = np.sqrt(self.latent_dim/2).astype(int)
                xdim, ydim = latent_spatial, latent_spatial
            else:
                raise ValueError(f"Unsupported AE mode for WaveAELSTM: {self.method}")
            labels = labels.view(labels.size(0), labels.size(1), 2, xdim, ydim)
            self.test_results[mode]['nf_truth'].append(labels.detach().cpu().numpy())
        else:
            self.test_results[mode]['nf_truth'].append(labels_np)
            
    def configure_ae(self):       
        # Encoder: downsampling
        self.encoder = Encoder(
            channels=self.arch_params.encoder_channels,
            conf=self.conf.autoencoder
        )
        
        # Decoder: upsampling
        if self.use_decoder == True:
            self.decoder = Decoder(
                channels=self.decoder_channels,
                conf=self.conf.autoencoder
            )
    
        if self.pretrained == True:
            # load pretrained autoencoder
            dirpath = '/develop/meep_meep/autoencoder/model_ae-linear-v1/'
            checkpoint = torch.load(dirpath + "model.ckpt")
            encoder_state_dict = {}

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
        spatial = 166
        
        return in_channels, spatial
    
    def process_ae(self, x, lstm_out=None):
        if self.encoding_done == False: # encode the input
            batch, seq_len, input = x.size()
            if input != 55112:
                raise Warning(f"Input flattened size must be 55112, got {input}.")
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