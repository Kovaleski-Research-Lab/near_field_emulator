#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import numpy as np
#from geomloss import SamplesLoss
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
#from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import math
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager
#from core.complexNN import ComplexLinear, ModReLU
from .CVNN import ComplexReLU, ModReLU, ComplexLinearFinal

sys.path.append("../")

class WaveMLP(LightningModule):
    """Near Field Response Prediction Model  
    Architecture: MLPs (real and imaginary)  
    Modes: Full, patch-wise"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.learning_rate = self.conf.learning_rate
        self.lr_scheduler = self.conf.lr_scheduler
        self.loss_func = self.conf.objective_function
        self.fold_idx = fold_idx
        self.patch_size = self.conf.patch_size # for non-default approaches
        self.name = self.conf.arch # mlp or cvnn
        self.num_design_conf = int(self.conf.num_design_conf)
        self.strat = None
        
        if self.conf.mlp_strategy == 0:
            self.strat = 'standard'
        elif self.conf.mlp_strategy == 1:
            self.strat = 'patch'
        elif self.conf.mlp_strategy == 2:
            self.strat = 'distributed'
        elif self.conf.mlp_strategy == 3:
            self.strat = 'field2field'
        elif self.conf.mlp_strategy == 4:
            self.strat == "inverse"
        else:
            raise ValueError("Approach not recognized.")
        
        if self.strat == 'patch': # patch-wise
            self.output_size = (self.patch_size)**2
            self.num_patches_height = math.ceil(166 / self.patch_size)
            self.num_patches_width = math.ceil(166 / self.patch_size)
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
            self.output_size = 166 * 166
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.output_size, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(self.output_size, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.output_size, self.conf.mlp_imag)
        
        elif self.strat == 'inverse':
            self.output_size = 1
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(56*56, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(56*56, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(56*56, self.conf.mlp_imag)  
        
        else:
            # Build full MLPs
            self.output_size = 166 * 166
            if self.name == 'cvnn':
                self.cvnn = self.build_mlp(self.num_design_conf, self.conf.cvnn)
            else:
                self.mlp_real = self.build_mlp(self.num_design_conf, self.conf.mlp_real)
                self.mlp_imag = self.build_mlp(self.num_design_conf, self.conf.mlp_imag)
        
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
         
    def build_mlp(self, input_size, mlp_conf):
        layers = []
        in_features = input_size
        for layer_size in mlp_conf['layers']:
            if self.name == 'cvnn': # complex-valued NN
                layers.append(ComplexLinear(in_features, layer_size))
            else: # real-valued NN
                layers.append(nn.Linear(in_features, layer_size))
            layers.append(self.get_activation_function(mlp_conf['activation']))
            in_features = layer_size
        if self.name == 'cvnn':
            layers.append(ComplexLinearFinal(in_features, self.output_size))
        else:
            layers.append(nn.Linear(in_features, self.output_size))
        return nn.Sequential(*layers)
        
    def get_activation_function(self, activation_name):
        if activation_name == 'relu':
            return nn.ReLU()
        elif activation_name == 'sigmoid':
            return nn.Sigmoid()
        elif activation_name == 'tanh':
            return nn.Tanh()
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation_name == 'modrelu':
            return ModReLU()
        elif activation_name == 'complexrelu':
            return ComplexReLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")
        
    def forward(self, radii, near_fields):
        if self.name == 'cvnn':
            # Convert radii to complex numbers
            radii_complex = torch.complex(radii, torch.zeros_like(radii))
            nf_complex = torch.complex(near_fields, torch.zeroes_like(near_fields))
            if self.strat == 'patch':
                # Patch approach with complex MLPs
                batch_size = radii.size(0)
                patches = []
                for i in range(self.num_patches):
                    patch = self.cvnn[i](radii_complex)
                    patch = patch.view(batch_size, self.patch_size, self.patch_size)
                    patches.append(patch)
                # Assemble patches
                output = self.assemble_patches(patches, batch_size)
                # Crop to original size if necessary
                output = output[:, :, :166, :166]
            elif self.strat == 'distributed':
                # Distributed subset approach
                output = self.cvnn(radii_complex)
                output = output.view(-1, self.patch_size, self.patch_size)
            elif self.strat == 'inverse':
                # going from fields to design
                output = self.cvnn(nf_complex)
                return output
            else:
                # Full approach
                #print(f"radii_complex: {radii_complex.shape}")
                output = self.cvnn(radii_complex)
                output = output.view(-1, 166, 166)
                return output
        
        else:
            # Original real-valued MLPs
            if self.strat == 'patch':
                # Patch approach
                batch_size = radii.size(0)
                real_patches = []
                imag_patches = []

                for i in range(self.num_patches):
                    # Real part
                    real_patch = self.mlp_real[i](radii)
                    real_patch = real_patch.view(batch_size, self.patch_size, self.patch_size)
                    real_patches.append(real_patch)

                    # Imaginary part
                    imag_patch = self.mlp_imag[i](radii)
                    imag_patch = imag_patch.view(batch_size, self.patch_size, self.patch_size)
                    imag_patches.append(imag_patch)

                # Assemble patches
                real_output = self.assemble_patches(real_patches, batch_size)
                imag_output = self.assemble_patches(imag_patches, batch_size)

                # Crop to original size if necessary
                real_output = real_output[:, :166, :166]
                imag_output = imag_output[:, :166, :166]
            elif self.strat == 'distributed':
                # Distributed subset approach
                real_output = self.mlp_real(radii)
                imag_output = self.mlp_imag(radii)
                # Reshape to patch_size x patch_size
                real_output = real_output.view(-1, self.patch_size, self.patch_size)
                imag_output = imag_output.view(-1, self.patch_size, self.patch_size)
            elif self.strat == 'field2field':
                # in this case, radii is actually a field, specifically our input field
                batch_size = radii.size(0)
                real_output = self.mlp_real(radii[:, 0, :, :].reshape(batch_size, -1))
                imag_output = self.mlp_imag(radii[:, 1, :, :].reshape(batch_size, -1))
                # Reshape to image size
                real_output = real_output.view(-1, 166, 166)
                imag_output = imag_output.view(-1, 166, 166)
            elif self.strat == 'inverse':
                # in this case, radii is actually a field, specifically our input field
                batch_size = radii.size(0)
                real_output = self.mlp_real(near_fields[:, 0, :, :].reshape(batch_size, -1))
                imag_output = self.mlp_imag(near_fields[:, 1, :, :].reshape(batch_size, -1)) 
            else:
                # Full approach
                real_output = self.mlp_real(radii)
                imag_output = self.mlp_imag(radii)
                # Reshape to image size
                real_output = real_output.view(-1, 166, 166)
                imag_output = imag_output.view(-1, 166, 166)

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
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # setup specified scheduler
        if self.lr_scheduler == 'ReduceLROnPlateau':
            choice = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.5, patience=5, 
                                                                    min_lr=1e-6, threshold=0.001, 
                                                                    cooldown=2)
        elif self.lr_scheduler == 'CosineAnnealingLR':
            choice = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=100,
                                                                eta_min=1e-6)
        elif self.lr_scheduler == 'None':
            return optimizer
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        
        scheduler = {
            'scheduler': choice,
            'interval': 'epoch',
            'monitor': 'val_loss',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def compute_loss(self, preds, labels, choice):
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 'emd':
            # ignoring emd for now
            raise NotImplementedError("Earth Mover's Distance not implemented!")
            '''# Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss'''
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            psnr_value = fn(preds, labels)
            loss = -psnr_value # minimize negative psnr
        elif choice == 'ssim':
            # Structural Similarity Index
            if preds.size(-1) < 11 or preds.size(-2) < 11:
                loss = 0 # if the size is too small, SSIM is not defined
            else:
                preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
                torch.use_deterministic_algorithms(True, warn_only=True)
                with torch.backends.cudnn.flags(enabled=False):
                    fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                    ssim_value = fn(preds, labels)
                    loss = -ssim_value  # SSIM is a similarity metric
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, batch, predictions):
        near_fields, radii = batch
        
        print(f"\n\n\nself.strat SELF STRAT IS ----\n\n**??!!--> {self.strat} <--!!??**\n\n")
        
        if self.strat == 'inverse':
            design_loss = self.compute_loss(predictions, radii)
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
                    # ignoring emd for now - geomloss library has issues
                    '''if key == 'emd':
                        # Reshape tensors to (batch_size, num_pixels, 1)
                        pred_real_reshaped = pred_real.view(pred_real.size(0), -1, 1)
                        real_near_fields_reshaped = real_near_fields.view(real_near_fields.size(0), -1, 1)
                        pred_imag_reshaped = pred_imag.view(pred_imag.size(0), -1, 1)
                        imag_near_fields_reshaped = imag_near_fields.view(imag_near_fields.size(0), -1, 1)

                        loss_real = self.compute_loss(pred_real_reshaped, real_near_fields_reshaped, choice=key)
                        loss_imag = self.compute_loss(pred_imag_reshaped, imag_near_fields_reshaped, choice=key)
                    else:'''
                    loss_real = self.compute_loss(preds_real, labels_real, choice=key)
                    loss_imag = self.compute_loss(preds_imag, labels_imag, choice=key)
                    loss = loss_real + loss_imag
                    choices[key] = loss
            
            return {"loss": near_field_loss, **choices}
    
    def shared_step(self, batch, batch_idx):
        near_fields, radii = batch
        preds = self.forward(radii, near_fields)
        return preds
    
    def training_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log metrics
        if self.loss_func == 'psnr' or self.loss_func == 'ssim': # PSNR is inverted for minimization, report true value here
            self.log('train_loss', -loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else: # mse
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #print(f"train_psnr_recorded: {-loss}")
        other_metrics = [f"{key}" for key in loss_dict.keys() if key != 'loss' and key != self.loss_func]
        for key in other_metrics:
            #print(f"train_{key}", loss_dict[key])
            self.log(f"train_{key}", loss_dict[key], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        #print(f"keys: {loss_dict.keys()}")
        
        # log metrics
        if self.loss_func == 'psnr' or self.loss_func == 'ssim': # PSNR is inverted for minimization, report true value here
            self.log('val_loss', -loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        else: # mse
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        #print(f"val_psnr_recorded: {-loss}")
        other_metrics = [f"{key}" for key in loss_dict.keys() if key != 'loss' and key != self.loss_func]
        for key in other_metrics:
            #print(f"valid_{key}", loss_dict[key])
            self.log(f"valid_{key}", loss_dict[key], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        preds = self.shared_step(batch, batch_idx)   
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        near_fields, radii = batch
        
        if self.strat == 'inverse':
            preds_combined = predictions
            truths_combined = radii
        
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
            self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
            self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)