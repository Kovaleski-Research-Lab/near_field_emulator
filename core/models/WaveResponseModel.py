#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import math
import abc
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear
from .CVNN import ComplexDropout, ComplexReLU, ModReLU, ComplexLinearFinal

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager
#from .CVNN import ComplexReLU, ModReLU, ComplexLinearFinal

sys.path.append("../")

class WaveResponseModel(LightningModule, metaclass=abc.ABCMeta):
    """
    Near Field Response Prediction Model
    Base Abstract Class
    
    Defines a common interface and attributes that all child classes must implement.
    Parent to classes associated with direct mapping between design and field (forward, inverse, etc.)
    """
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.learning_rate = self.conf.learning_rate
        self.lr_scheduler = self.conf.lr_scheduler
        self.loss_func = self.conf.objective_function
        self.fold_idx = fold_idx
        self.name = self.conf.arch
        self.num_design_conf = int(self.conf.num_design_conf)
        self.near_field_dim = int(self.conf.near_field_dim)
        
        self.save_hyperparameters()
        
        # Store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.create_architecture()
        
    @abc.abstractmethod
    def create_architecture(self):
        """
        Sets up specific configurations tailored to respective subclasses

        Parameters
        ----------
        input_size : Integer
            Value specifying the number of initial inputs to the MLP
        mlp_conf : Dict
            Configurations specific to the MLP architecture required.
        """
        
    @abc.abstractmethod
    def forward(self, designs, fields):
        """
        Model forward pass.

        Parameters
        ----------
        designs: torch.tensor
            design parameters, such as pillar radii sets, heights, etc.
        fields: torch.tensor
            actual near field response DFT maps
        """
        
    @abc.abstractmethod
    def shared_step(self, batch, batch_idx):
        """
        Method holding model-specific shared logic for training/validation/testing
        Each subclass must implement this method.
        """
        
    @abc.abstractmethod
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        """
        Performs general post-processing and loading of testing results for a model.
        Each subclass must implement.
        """

    @abc.abstractmethod
    def objective(self, preds, labels):
        """
        A wrapper method around compute_loss to provide a unified interface.
        Objective varies despite fundamentally being a wrapper so it is abstract
        """
        
    def build_mlp(self, input_size, mlp_conf):
        layers = []
        in_features = input_size
        for layer_size in mlp_conf['layers']:
            if self.name in ['cvnn', 'inverse']: # complex-valued NN
                layers.append(ComplexLinear(in_features, layer_size))
                dropout = ComplexDropout(self.conf.dropout)
                dropout = dropout.to(self._device)  # Move dropout to correct device
                layers.append(dropout)
            else: # real-valued NN
                layers.append(nn.Linear(in_features, layer_size))
                layers.append(nn.Dropout(self.conf.dropout))
            layers.append(self.get_activation_function(mlp_conf['activation']))
            in_features = layer_size
        if self.name in ['cvnn', 'inverse']:
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
        
    def compute_loss(self, preds, labels, choice):
        """
        Compute loss given predictions and labels.
        """
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 'emd':
            # ignoring emd for now
            raise NotImplementedError("Earth Mover's Distance not implemented!")
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
                    #print(f'SSIM VALUE: {ssim_value}')
                    ssim_comp = (1 - ssim_value)
                #loss = ssim_comp
                # Mean Squared Error
                preds = preds.to(torch.float32).contiguous()
                labels = labels.to(torch.float32).contiguous()
                fn2 = torch.nn.MSELoss()
                mse_comp = fn2(preds, labels)
                loss = self.conf.mcl_params['alpha'] * mse_comp + self.conf.mcl_params['beta'] * ssim_comp
                #loss = ssim_comp
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def configure_optimizers(self):
        """
        Setup optimzier and LR scheduler.
        """
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
        elif self.lr_scheduler == 'CosineAnnealingWarmRestarts':
            choice = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                          T_0=10,
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

     
    '''def training_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log loss metrics
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Compute PSNR/SSIM for real and imaginary components
        near_fields, designs = batch
        psnr_vals = []
        ssim_vals = []
        
        # Handle real and imaginary components separately
        for comp in range(2):
            pred_comp = preds[:, comp].unsqueeze(1)  # [B, 1, H, W]
            truth_comp = near_fields[:, comp].unsqueeze(1)  # [B, 1, H, W]
            
            # Compute PSNR
            psnr_val = self.train_psnr(pred_comp.float(), truth_comp.float())
            if not torch.isnan(psnr_val) and not torch.isinf(psnr_val):
                psnr_vals.append(psnr_val)
            
            # Compute SSIM only if size is sufficient
            if pred_comp.size(-1) >= 11 and pred_comp.size(-2) >= 11:
                ssim_val = self.train_ssim(pred_comp.float(), truth_comp.float())
                if not torch.isnan(ssim_val) and not torch.isinf(ssim_val):
                    ssim_vals.append(ssim_val)
        
        # Average metrics across components if we have valid values
        if psnr_vals:
            psnr = torch.stack(psnr_vals).mean()
            self.log("train_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if ssim_vals:
            ssim = torch.stack(ssim_vals).mean()
            self.log("train_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log loss metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # Compute PSNR/SSIM for real and imaginary components
        near_fields, designs = batch
        psnr_vals = []
        ssim_vals = []
        
        # Handle real and imaginary components separately
        for comp in range(2):
            pred_comp = preds[:, comp].unsqueeze(1)  # [B, 1, H, W]
            truth_comp = near_fields[:, comp].unsqueeze(1)  # [B, 1, H, W]
            
            # Compute PSNR
            psnr_val = self.val_psnr(pred_comp.float(), truth_comp.float())
            if not torch.isnan(psnr_val) and not torch.isinf(psnr_val):
                psnr_vals.append(psnr_val)
            
            # Compute SSIM only if size is sufficient
            if pred_comp.size(-1) >= 11 and pred_comp.size(-2) >= 11:
                ssim_val = self.val_ssim(pred_comp.float(), truth_comp.float())
                if not torch.isnan(ssim_val) and not torch.isinf(ssim_val):
                    ssim_vals.append(ssim_val)
        
        # Average metrics across components if we have valid values
        if psnr_vals:
            psnr = torch.stack(psnr_vals).mean()
            self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        if ssim_vals:
            ssim = torch.stack(ssim_vals).mean()
            self.log("val_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}'''
        
    def training_step(self, batch, batch_idx):
        preds = self.shared_step(batch, batch_idx)
        loss_dict = self.objective(batch, preds)
        loss = loss_dict['loss']
        
        # log metrics
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
        
        # log metrics
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