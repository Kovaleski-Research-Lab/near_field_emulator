#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import sys
import torch
import numpy as np
import torch.nn as nn
import math

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
from .WaveResponseModel import WaveResponseModel
from .autoencoder import Encoder, Decoder

sys.path.append("../")

class WaveForwardMLP(WaveResponseModel):
    """Near Field Response Prediction Model  
    Architecture: MLPs (real and imaginary)  
    Modes: Full, patch-wise"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)
        
    def create_architecture(self):
        # Determine strategy
        self.strat = None
        strategy_map = {
            0: 'standard', 1: 'patch', 2: 'distributed',
            3: 'field2field', 4: 'conv_decoder', 5: 'ae', 6: 'pca'
        }
        if self.conf.forward_strategy in strategy_map:
            self.strat = strategy_map[self.conf.forward_strategy]
        else:
            raise ValueError(f"Forward strategy {self.conf.forward_strategy} not recognized.")

        # --- Convolutional Decoder Strategy ---
        if self.strat == 'conv_decoder':
            # --- Hardcoded Decoder Parameters ---
            decoder_initial_channels = 16
            decoder_initial_size = 4
            latent_dim_required = decoder_initial_channels * (decoder_initial_size**2)
            # Define hardcoded lists for the loop
            decoder_channels_list = [128, 64, 32, 16]
            decoder_kernels_list = [4, 4, 4, 4]
            decoder_strides_list = [2, 2, 2, 2]
            decoder_paddings_list = [1, 1, 1, 1]
            decoder_use_batchnorm_hardcoded = True
            decoder_activation_hardcoded = 'tanh'
            # --- End Hardcoded Parameters ---

            # --- Configuration Validation Checks ---
            if self.name == 'cvnn':
                 raise ValueError("Standard ConvDecoder strategy is not compatible with 'cvnn' mode.")
            # Check if the MLP config matches the required latent dim
            # Using mlp_real config as per your original code
            mlp_output_dim = self.conf.mlp_real['layers'][-1]
            if mlp_output_dim != latent_dim_required:
                 raise ValueError(f"Configuration error: The last layer size in 'mlp_real' config ({mlp_output_dim}) "
                                  f"must match the hardcoded required latent dim for the decoder ({latent_dim_required})")
            # --- End Check ---

            self.output_size = self.near_field_dim**2 * 2 # Compatibility placeholder
            
            # 1. ---- Manually Build Initial MLP (Design Params -> Latent Vector) ----
            print(f"Manually building initial MLP to output {latent_dim_required} features.")
            initial_mlp_layers = []
            in_features = self.num_design_conf # Start with number of design parameters (e.g., 4)
            # Get MLP config details
            mlp_hidden_layers = self.conf.mlp_real['layers'] # e.g., [32, 64, 256]
            mlp_activation = self.conf.mlp_real['activation'] # e.g., 'relu'

            for i, layer_size in enumerate(mlp_hidden_layers):
                initial_mlp_layers.append(nn.Linear(in_features, layer_size))
                initial_mlp_layers.append(nn.Dropout(self.conf.dropout)) # Use dropout from config
                initial_mlp_layers.append(self.get_activation_function(mlp_activation))
                in_features = layer_size # Update for next loop

            # --- NO FINAL Linear(in_features, self.output_size) layer added here ---
            self.initial_mlp = nn.Sequential(*initial_mlp_layers)

            '''# 1. Initial MLP (Design Params -> Latent Vector)
            # Using mlp_real config
            mlp_latent_conf = {
                'layers': self.conf.mlp_real['layers'],
                'activation': self.conf.mlp_real['activation']
            }
            self.initial_mlp = self.build_mlp(self.num_design_conf, mlp_latent_conf)'''

            # 2. Convolutional Decoder (Latent Vector -> Field Image)
            decoder_layers = []
            # Reshape layer (using hardcoded values)
            decoder_layers.append(nn.Unflatten(1, (decoder_initial_channels, decoder_initial_size, decoder_initial_size)))

            # Build Upsampling Blocks using hardcoded lists
            in_channels = decoder_initial_channels
            num_upsample_layers = len(decoder_channels_list)

            # Check list lengths (optional but good practice)
            if not (len(decoder_channels_list) == len(decoder_kernels_list) == len(decoder_strides_list) == len(decoder_paddings_list)):
                 raise ValueError("Internal error: Hardcoded decoder lists have different lengths.")
            current_size = decoder_initial_size
            for i in range(num_upsample_layers):
                # --- Get parameters for *this* layer using index i ---
                out_ch = decoder_channels_list[i]
                kernel = decoder_kernels_list[i]
                stride = decoder_strides_list[i]
                padding = decoder_paddings_list[i]

                decoder_layers.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_ch,    # Use integer from list
                        kernel_size=kernel, # Use integer/tuple from list
                        stride=stride,      # Use integer/tuple from list
                        padding=padding,    # Use integer/tuple from list
                        bias=not decoder_use_batchnorm_hardcoded
                    )
                )
                current_size = (current_size - 1) * stride - 2 * padding + kernel
                print(f" -> Layer {i+1}: {current_size}x{current_size} ({out_ch} channels)")

                if decoder_use_batchnorm_hardcoded:
                    decoder_layers.append(nn.BatchNorm2d(out_ch))
                decoder_layers.append(self.get_activation_function(decoder_activation_hardcoded))
                in_channels = out_ch

            if current_size != 64:
                 print(f"Warning: Expected decoder output size 64x64 before final conv, but got {current_size}x{current_size}")

            # --- MODIFICATION: Final Conv2d now only maps channels, preserves 64x64 size ---
            final_conv_kernel = 3 # Use kernel=3, padding=1 to preserve size
            final_conv_padding = 1
            print(f" -> Final Conv2d: kernel={final_conv_kernel}, padding={final_conv_padding} to map channels -> 2 @ {current_size}x{current_size}")
            decoder_layers.append(
                nn.Conv2d(
                    in_channels, # Should be 16
                    2,           # Output 2 channels (real, imag)
                    kernel_size=final_conv_kernel,
                    stride=1,
                    padding=final_conv_padding
                )
            )
            # current_size remains 64
            print(f" -> Output Size before crop: {current_size}x{current_size} (2 channels)")

            # Final Convolutional Layer to get 2 channels (Real, Imag)
            # Assumes the loop resulted in the correct spatial size (56x56)
            decoder_layers.append(
                nn.Conv2d(
                    in_channels, 2, kernel_size=3, stride=1, padding=1 # Kernel 3, Pad 1 preserves size
                )
            )

            # Final Activation (applied AFTER the final conv layer)
            decoder_layers.append(self.get_activation_function('tanh'))

            self.decoder = nn.Sequential(*decoder_layers)
        elif self.strat == 'ae':
            # Load pretrained autoencoder
            dirpath = '/develop/results/meep_meep/refractive_idx/autoencoder/model_spatial-test-mcl05-k5/'
            checkpoint = torch.load(dirpath + "model.ckpt")
            
            # Initialize encoder and decoder
            self.encoder = Encoder(self.conf.autoencoder.encoder_channels, conf=self.conf.autoencoder)
            self.decoder = Decoder(self.conf.autoencoder.decoder_channels, conf=self.conf.autoencoder)
            
            # Load pretrained weights
            encoder_state_dict = {}
            decoder_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                if key.startswith('encoder'):
                    encoder_state_dict[key.replace('encoder.', '')] = value
                elif key.startswith('decoder'):
                    decoder_state_dict[key.replace('decoder.', '')] = value
            
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
            
            # Freeze encoder and decoder weights
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            
            # Build MLP to predict latent representation
            self.output_size = self.conf.autoencoder.latent_dim ** 2
            # Use a specialized MLP for latent space prediction
            self.mlp_ae = nn.Sequential(
                nn.Linear(self.num_design_conf, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, self.conf.autoencoder.latent_dim ** 2)
            )
            self.mlp_real = self.build_mlp(self.num_design_conf, self.conf.mlp_real)
            self.mlp_imag = self.build_mlp(self.num_design_conf, self.conf.mlp_imag)
        else:
            # Build full MLPs
            self.output_size = self.near_field_dim**2
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.num_design_conf, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(self.num_design_conf, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.num_design_conf, self.conf.mlp_imag)
        
    def forward(self, designs, near_fields):
        # --- Convolutional Decoder Strategy ---
        if self.strat == 'conv_decoder':
            latent_vector = self.initial_mlp(designs)
            output_field = self.decoder(latent_vector) # Shape: [batch_size, 2, 56, 56]
            return output_field # Return single tensor with real/imag channels
        
        elif self.strat == 'ae':
            # Encode the near fields to latent space
            with torch.no_grad():
                encoded_fields = self.encoder(near_fields)
            
            # forward pass through the latent MLP
            #latent_preds = self.mlp_ae(designs)
            latent_preds_real = self.mlp_real(designs)
            latent_preds_imag = self.mlp_imag(designs)
            latent_preds_real = latent_preds_real.view(-1, self.conf.autoencoder.latent_dim, self.conf.autoencoder.latent_dim)
            latent_preds_imag = latent_preds_imag.view(-1, self.conf.autoencoder.latent_dim, self.conf.autoencoder.latent_dim)
            latent_preds = [latent_preds_real, latent_preds_imag]
            
            # Return both predictions and encoded labels in latent space
            return latent_preds, encoded_fields
        
        elif self.name == 'cvnn':
            # Convert designs to complex numbers
            designs_complex = torch.complex(designs, torch.zeros_like(designs))
            if self.strat == 'patch':
                # Patch approach with complex MLPs
                batch_size = designs.size(0)
                patches = []
                for i in range(self.num_patches):
                    patch = self.cvnn[i](designs_complex)
                    patch = patch.view(batch_size, self.patch_size, self.patch_size)
                    patches.append(patch)
                # Assemble patches
                output = self.assemble_patches(patches, batch_size)
                # Crop to original size if necessary
                output = output[:, :, :self.near_field_dim, :self.near_field_dim]
            elif self.strat == 'distributed':
                # Distributed subset approach
                output = self.cvnn(designs_complex)
                output = output.view(-1, self.patch_size, self.patch_size)
            else:
                # Full approach
                #print(f"designs_complex: {designs_complex.shape}")
                output = self.cvnn(designs_complex)
                output = output.view(-1, self.near_field_dim, self.near_field_dim)
                return output
        
        else:
            # Original real-valued MLPs
            if self.strat == 'patch':
                # Patch approach
                batch_size = designs.size(0)
                real_patches = []
                imag_patches = []

                for i in range(self.num_patches):
                    # Real part
                    real_patch = self.mlp_real[i](designs)
                    real_patch = real_patch.view(batch_size, self.patch_size, self.patch_size)
                    real_patches.append(real_patch)

                    # Imaginary part
                    imag_patch = self.mlp_imag[i](designs)
                    imag_patch = imag_patch.view(batch_size, self.patch_size, self.patch_size)
                    imag_patches.append(imag_patch)

                # Assemble patches
                real_output = self.assemble_patches(real_patches, batch_size)
                imag_output = self.assemble_patches(imag_patches, batch_size)

                # Crop to original size if necessary
                real_output = real_output[:, :self.near_field_dim, :self.near_field_dim]
                imag_output = imag_output[:, :self.near_field_dim, :self.near_field_dim]
            elif self.strat == 'distributed':
                # Distributed subset approach
                real_output = self.mlp_real(designs)
                imag_output = self.mlp_imag(designs)
                # Reshape to patch_size x patch_size
                real_output = real_output.view(-1, self.patch_size, self.patch_size)
                imag_output = imag_output.view(-1, self.patch_size, self.patch_size)
            elif self.strat == 'field2field':
                # in this case, radii is actually a field, specifically our input field
                batch_size = designs.size(0)
                real_output = self.mlp_real(designs[:, 0, :, :].reshape(batch_size, -1))
                imag_output = self.mlp_imag(designs[:, 1, :, :].reshape(batch_size, -1))
                # Reshape to image size
                real_output = real_output.view(-1, self.near_field_dim, self.near_field_dim)
                imag_output = imag_output.view(-1, self.near_field_dim, self.near_field_dim)
            else:
                # Full approach
                real_output = self.mlp_real(designs)
                imag_output = self.mlp_imag(designs)
                # Reshape to image size
                real_output = real_output.view(-1, self.near_field_dim, self.near_field_dim)
                imag_output = imag_output.view(-1, self.near_field_dim, self.near_field_dim)

            return real_output, imag_output
        
    def assemble_patches(self, patches, batch_size):
        # reshape patches into grid
        patches_per_row = self.num_patches_width
        patches_per_col = self.num_patches_height
        patch_size = self.patch_size

        patches_tensor = torch.stack(patches, dim=1)  # Shape: [batch_size, num_patches, patch_size, patch_size]
        patches_tensor = patches_tensor.view(
            batch_size,
            patches_per_col,
            patches_per_row,
            patch_size,
            patch_size
        )

        # permute and reshape to assemble the image
        output = patches_tensor.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(
            batch_size,
            patches_per_col * patch_size,
            patches_per_row * patch_size
        )

        return output 

    def objective(self, batch, predictions):
        near_fields, radii = batch
        
        if self.strat == 'ae':
            # Get predictions and encoded labels in latent space
            latent_preds, encoded_labels = predictions
            preds_real = latent_preds[0]
            preds_imag = latent_preds[1]
            #print(preds_real.shape)
            labels_real = encoded_labels[:, 0, :, :]
            labels_imag = encoded_labels[:, 1, :, :]
            #print(f"preds_real shape: {preds_real.shape}")
            #print(f"labels_real shape: {labels_real.shape}")
            # Compute loss directly in latent space
            #latent_loss = self.compute_loss(latent_preds, encoded_labels, choice=self.loss_func)
            # DECODER STUFF
            preds_combined = torch.stack([preds_real, preds_imag], dim=1)
            preds_combined = self.decoder(preds_combined)
            preds_real = preds_combined[:, 0, :, :]
            preds_imag = preds_combined[:, 1, :, :]
            labels_real = near_fields[:, 0, :, :]
            labels_imag = near_fields[:, 1, :, :]
            
            # Near-field loss: compute separately for real and imaginary components
            near_field_loss_real = self.compute_loss(preds_real, labels_real, choice=self.loss_func)
            near_field_loss_imag = self.compute_loss(preds_imag, labels_imag, choice=self.loss_func)
            latent_loss = near_field_loss_real + near_field_loss_imag
            
            # compute other metrics for logging
            choices = {
                'mse': None,
                'ssim': None,
                'psnr': None
            }
            
            for key in choices:
                if key != self.loss_func:
                    loss_real = self.compute_loss(preds_real, labels_real, choice=key)
                    loss_imag = self.compute_loss(preds_imag, labels_imag, choice=key)
                    loss = loss_real + loss_imag
                    #loss = self.compute_loss(latent_preds, encoded_labels, choice=key)
                    choices[key] = loss
            
            return {"loss": latent_loss, **choices}
        else:
            # Handle predictions based on output format
            if self.name == 'cvnn' and self.strat != 'ae':
                labels = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
                labels_real = labels.real
                labels_imag = labels.imag
                preds_real = predictions.real
                preds_imag = predictions.imag
            elif self.strat == 'conv_decoder':
                # predictions is a single tensor (B, 2, H, W)
                preds_real = predictions[:, 0, :, :]
                preds_imag = predictions[:, 1, :, :]
                labels_real = near_fields[:, 0, :, :]
                labels_imag = near_fields[:, 1, :, :]
            else:
                preds_real, preds_imag = predictions
                labels_real = near_fields[:, 0, :, :]
                labels_imag = near_fields[:, 1, :, :]
            
            # Near-field loss: compute separately for real and imaginary components
            near_field_loss_real = self.compute_loss(preds_real, labels_real, choice=self.loss_func)
            near_field_loss_imag = self.compute_loss(preds_imag, labels_imag, choice=self.loss_func)
            near_field_loss = near_field_loss_real + near_field_loss_imag
        
            # compute other metrics for logging besides specified loss function
            choices = {
                'mse': None,
                #'emd': None,
                'ssim': None,
                'psnr': None
            }
            
            for key in choices:
                if key != self.loss_func:
                    loss_real = self.compute_loss(preds_real, labels_real, choice=key)
                    loss_imag = self.compute_loss(preds_imag, labels_imag, choice=key)
                    loss = loss_real + loss_imag
                    choices[key] = loss
            
            return {"loss": near_field_loss, **choices}
        
    def shared_step(self, batch, batch_idx):
        near_fields, designs = batch
        preds = self.forward(designs, near_fields)
        return preds
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        near_fields, designs = batch
        
        if self.strat == 'conv_decoder':
            # predictions is a single tensor (B, 2, H, W)
            preds_combined = predictions.cpu().numpy()
        elif self.name == 'cvnn':
            labels = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
            labels_real = labels.real
            labels_imag = labels.imag
            preds_real = predictions.real
            preds_imag = predictions.imag
            
            # collect preds and ground truths
            preds_combined = torch.stack([preds_real, preds_imag], dim=1).cpu().numpy()
        elif self.strat == 'ae':
            preds, encoded_fields = predictions
            preds_real = preds[0]
            preds_imag = preds[1]
            # Keep as tensor for decoder
            preds_combined = torch.stack([preds_real, preds_imag], dim=1)
            # Pass through decoder while still a tensor
            preds_combined = self.decoder(preds_combined)
            # Convert to numpy after decoder processing
            preds_combined = preds_combined.cpu().numpy()
            labels_real = near_fields[:, 0, :, :]
            labels_imag = near_fields[:, 1, :, :]
            

        else:
            preds_real, preds_imag = predictions
            labels_real = near_fields[:, 0, :, :]
            labels_imag = near_fields[:, 1, :, :]
        
            # collect preds and ground truths
            preds_combined = torch.stack([preds_real, preds_imag], dim=1).cpu().numpy()
        truths_combined = torch.stack([labels_real, labels_imag], dim=1).cpu().numpy()
            
        # Store predictions and ground truths for analysis after testing
        if dataloader_idx == 0:  # val dataloader
            self.test_results['valid']['nf_pred'].append(preds_combined)
            self.test_results['valid']['nf_truth'].append(truths_combined)
        elif dataloader_idx == 1:  # train dataloader
            self.test_results['train']['nf_pred'].append(preds_combined)
            self.test_results['train']['nf_truth'].append(truths_combined)
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
    def on_test_end(self):
        # Concatenate results from all batches
        for mode in ['train', 'valid']:
            # Ensure tensors are on CPU before concatenation
            self.test_results[mode]['nf_pred'] = np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else x for x in self.test_results[mode]['nf_pred']], axis=0)
            self.test_results[mode]['nf_truth'] = np.concatenate([x.cpu().numpy() if torch.is_tensor(x) else x for x in self.test_results[mode]['nf_truth']], axis=0)