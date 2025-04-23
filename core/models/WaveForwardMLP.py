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

sys.path.append("../")

class WaveForwardMLP(WaveResponseModel):
    """Near Field Response Prediction Model  
    Architecture: MLPs (real and imaginary)  
    Modes: Full, patch-wise"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)
        
    def create_architecture(self):
        # specific strategies for this model
        self.strat = None
        if self.conf.forward_strategy == 0:
            self.strat = 'standard'
        elif self.conf.forward_strategy == 1:
            self.strat = 'patch'
        elif self.conf.forward_strategy == 2:
            self.strat = 'distributed'
        elif self.conf.forward_strategy == 3:
            self.strat = 'field2field'
        elif self.conf.forward_strategy == 4:
            self.strat = "inverse"
        else:
            raise ValueError("Approach not recognized.")
        
        if self.strat == 'patch': # patch-wise
            self.output_size = (self.patch_size)**2
            self.num_patches_height = math.ceil(self.near_field_dim / self.patch_size)
            self.num_patches_width = math.ceil(self.near_field_dim / self.patch_size)
            self.num_patches = self.num_patches_height * self.num_patches_width
            # determine whether or not to use complex-valued NN
            # if not, we have separate MLPs, if so, we have one MLP
            if self.name == 'cvnn':
                self.cvnn = nn.ModuleList([
                    self.build_mlp(self.num_design_conf, self.conf['cvnn']) for _ in range(self.num_patches)
                ])
            else:
                # Build MLPs for each patch
                self.mlp_real = nn.ModuleList([
                self.build_mlp(self.num_design_conf, self.conf['mlp_real']) for _ in range(self.num_patches)
                ])
                self.mlp_imag = nn.ModuleList([
                    self.build_mlp(self.num_design_conf, self.conf['mlp_imag']) for _ in range(self.num_patches)
                ])
                
        elif self.strat == 'distributed': # distributed subset
            self.output_size = (self.patch_size)**2
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.num_design_conf, self.conf.cvnn)
            else:
                # build MLPs
                self.mlp_real = self.build_mlp(self.num_design_conf, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.num_design_conf, self.conf.mlp_imag)
                
        elif self.strat == 'field2field':
            self.output_size = self.near_field_dim**2
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.output_size, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(self.output_size, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.output_size, self.conf.mlp_imag)
        
        # TODO: phase out, already handled by its own separate subclass now
        elif self.strat == 'inverse':
            self.output_size = 1
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.near_field_dim**2, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(self.near_field_dim**2, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.near_field_dim**2, self.conf.mlp_imag)  
        
        else:
            # Build full MLPs
            self.output_size = self.near_field_dim**2
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.num_design_conf, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(self.num_design_conf, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.num_design_conf, self.conf.mlp_imag)
        
    def forward(self, designs, near_fields):
        if self.name == 'cvnn':
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
            elif self.strat == 'inverse': #TODO phase out
                # going from fields to design
                batch_size = near_fields.size(0)
                nf_complex = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
                nf_complex = nf_complex.view(batch_size, -1)
                output = self.cvnn(nf_complex)
                return output
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
            elif self.strat == 'inverse': #TODO phase out
                # this case is infeasible with dual MLPs (currently)
                raise NotImplementedError("Dual MLPs for inverse strategy not yet implemented.")
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
        
        if self.strat == 'inverse': #TODO phase out
            design_loss = self.compute_loss(predictions, radii, choice=self.loss_func)
            return {'loss': design_loss}
        else:
            if self.name == 'cvnn':
                labels = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
                labels_real = labels.real
                labels_imag = labels.imag
                preds_real = predictions.real
                preds_imag = predictions.imag
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
        
        # TODO phase out
        if self.strat == 'inverse':
            preds_combined = predictions
            truths_combined = designs
        
        else:
            if self.name == 'cvnn':
                labels = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
                labels_real = labels.real
                labels_imag = labels.imag
                preds_real = predictions.real
                preds_imag = predictions.imag
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