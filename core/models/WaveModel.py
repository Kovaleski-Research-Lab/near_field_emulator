#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import numpy as np
#from geomloss import SamplesLoss
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
#from torchvision.models import resnet50, resnet18, resnet34
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pytorch_lightning import LightningModule
import math
import abc

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
#from utils import parameter_manager

sys.path.append("../")

class WaveModel(LightningModule, metaclass=abc.ABCMeta):
    """
    Near Field Response Time Series Prediction Model
    Base Abstract Class
    
    Defines a common interface and attributes that all child classes 
    (WaveLSTM, WaveConvLSTM, WaveAELSTM, WaveAEConvLSTM, WaveModeLSTM) must implement.
    """
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.fold_idx = fold_idx
        
        # common attributes
        self.learning_rate = self.conf.learning_rate
        self.lr_scheduler = self.conf.lr_scheduler
        self.loss_func = self.conf.objective_function
        self.io_mode = self.conf.io_mode
        self.name = self.conf.arch
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seq_len = self.conf.seq_len
        self.io_mode = self.conf.io_mode
        self.spacing_mode = self.conf.spacing_mode
        self.autoreg = self.conf.autoreg
        
        # store necessary lists for tracking metrics per fold
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        # setup architecture
        self.create_architecture()
        
    @abc.abstractmethod
    def create_architecture(self):
        """
        Define model-specific layers and related components here.
        Each subclass must implement this method.
        """
        
    @abc.abstractmethod
    def forward(self, x, meta=None):
        """
        Forward pass of the model.
        Each subclass should implement its own forward logic.
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

    def compute_loss(self, preds, labels, choice='mse'):
        """
        Compute loss given predictions and labels.
        Subclasses can override if needed, but this base implementation is standard.
        """
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
            
        elif choice == 'emd':
            # ignoring emd for now
            '''# Earth Mover's Distance / Sinkhorn
            preds = preds.to(torch.float64).contiguous()
            labels = labels.to(torch.float64).contiguous()
            fn = SamplesLoss("sinkhorn", p=1, blur=0.05)
            loss = fn(preds, labels)
            loss = torch.mean(loss)  # Aggregating the loss'''
            raise NotImplementedError("Earth Mover's Distance not implemented!")
            
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self._device)
            loss = fn(preds, labels)
            
        elif choice == 'ssim':
            # Structural Similarity Index
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            torch.use_deterministic_algorithms(True, warn_only=True)
            with torch.backends.cudnn.flags(enabled=False):
                fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self._device)
                ssim_value = fn(preds, labels)
                loss = 1 - ssim_value  # SSIM is a similarity metric
                
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
    
    def objective(self, preds, labels):
        """
        A wrapper method around compute_loss to provide a unified interface.
        """
        return {"loss": self.compute_loss(preds, labels, choice=self.loss_func)}
    
    def configure_optimizers(self):
        """
        Setup optimzier and LR scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # LR scheduler setup - 2 options
        if self.lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(optimizer, 
                                             T_max=100,
                                             eta_min=1e-6)
            
        elif self.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, 
                                             mode='min', 
                                             factor=0.5, 
                                             patience=5, 
                                             min_lr=1e-6, 
                                             threshold=0.001, 
                                             cooldown=2)
            
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def training_step(self, batch, batch_idx):
        """
        Common training step shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # For sequential models, we need to handle multiple timesteps
        if isinstance(batch, list):
            # Get the ground truth sequence
            truth = batch[1]  # This should be your target sequence
            
            # Compute PSNR/SSIM for each timestep and average
            psnr_vals = []
            ssim_vals = []
            
            # Ensure predictions and truth have same number of timesteps
            for t in range(min(preds.shape[1], truth.shape[1])):
                pred_t = preds[:, t]  # [B, 2, H, W]
                truth_t = truth[:, t]  # [B, 2, H, W]
                
                # Handle real and imaginary components separately
                for comp in range(2):
                    pred_comp = pred_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    truth_comp = truth_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    
                    # Compute metrics for this component and timestep
                    psnr_t = self.train_psnr(pred_comp.float(), truth_comp.float())
                    ssim_t = self.train_ssim(pred_comp.float(), truth_comp.float())
                    
                    psnr_vals.append(psnr_t)
                    ssim_vals.append(ssim_t)
            
            # Average metrics across timesteps and components
            psnr = torch.stack(psnr_vals).mean()
            ssim = torch.stack(ssim_vals).mean()
            
        else:
            # Non-sequential case
            psnr_vals = []
            ssim_vals = []
            
            # Handle real and imaginary components separately
            for comp in range(2):
                pred_comp = preds[:, comp].unsqueeze(1)  # [B, 1, H, W]
                batch_comp = batch[:, comp].unsqueeze(1)  # [B, 1, H, W]
                
                psnr_vals.append(self.train_psnr(pred_comp.float(), batch_comp.float()))
                ssim_vals.append(self.train_ssim(pred_comp.float(), batch_comp.float()))
            
            psnr = torch.stack(psnr_vals).mean()
            ssim = torch.stack(ssim_vals).mean()
        
        self.log("train_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def validation_step(self, batch, batch_idx):
        """
        Common validation step shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        # For sequential models, we need to handle multiple timesteps
        if isinstance(batch, list):
            # Get the ground truth sequence
            truth = batch[1]  # This should be your target sequence
            
            # Compute PSNR/SSIM for each timestep and average
            psnr_vals = []
            ssim_vals = []
            
            # Ensure predictions and truth have same number of timesteps
            for t in range(min(preds.shape[1], truth.shape[1])):
                pred_t = preds[:, t]  # [B, 2, H, W]
                truth_t = truth[:, t]  # [B, 2, H, W]
                
                # Handle real and imaginary components separately
                for comp in range(2):
                    pred_comp = pred_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    truth_comp = truth_t[:, comp].unsqueeze(1)  # [B, 1, H, W]
                    
                    # Compute metrics for this component and timestep
                    psnr_t = self.val_psnr(pred_comp.float(), truth_comp.float())
                    ssim_t = self.val_ssim(pred_comp.float(), truth_comp.float())
                    
                    psnr_vals.append(psnr_t)
                    ssim_vals.append(ssim_t)
            
            # Average metrics across timesteps and components
            psnr = torch.stack(psnr_vals).mean()
            ssim = torch.stack(ssim_vals).mean()
            
        else:
            # Non-sequential case
            psnr_vals = []
            ssim_vals = []
            
            # Handle real and imaginary components separately
            for comp in range(2):
                pred_comp = preds[:, comp].unsqueeze(1)  # [B, 1, H, W]
                batch_comp = batch[:, comp].unsqueeze(1)  # [B, 1, H, W]
                
                psnr_vals.append(self.val_psnr(pred_comp.float(), batch_comp.float()))
                ssim_vals.append(self.val_ssim(pred_comp.float(), batch_comp.float()))
            
            psnr = torch.stack(psnr_vals).mean()
            ssim = torch.stack(ssim_vals).mean()
        
        self.log("val_psnr", psnr, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_ssim", ssim, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return {'loss': loss, 'output': preds, 'target': batch}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Common testing step likely shared among all subclasses
        """
        loss, preds = self.shared_step(batch, batch_idx)
        self.organize_testing(preds, batch, batch_idx, dataloader_idx)
    
    def on_test_end(self):
        """
        After testing, this method compiles results and logs them.
        """
        for mode in ['train', 'valid']:
            if self.test_results[mode]['nf_pred']:
                self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
                self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)
                
                # old approach
                '''             
                # Log or save results
                self.logger.experiment.log_results(
                    results=self.test_results[mode],
                    epoch=None,
                    mode=mode,
                    name="results"
                )'''
            else:
                print(f"No test results for mode: {mode}")
