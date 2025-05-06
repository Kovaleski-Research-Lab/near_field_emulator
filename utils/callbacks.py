#! CUSTOM CALLBACKS FOR USE IN TRAINING (core/train.py)

#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import signal
import gc
import logging
import shutil
import sys
import csv
from sklearn.model_selection import KFold
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping, Callback
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.plugins.environments import SLURMEnvironment
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.auto import tqdm
import pandas as pd

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

#--------------------------------
# Callbacks
#--------------------------------

class CSVLoggerCallback(Callback):
    """
    Callback that saves training metrics to a CSV file.
    
    This callback maintains the same CSV logging functionality as the old custom_logger.py,
    while allowing the use of TensorBoard for visualization.
    """
    def __init__(self, save_dir, fold_idx=None):
        """
        Initialize the CSV logger callback.
        
        Parameters
        ----------
            save_dir: str
                Directory where the CSV file will be saved
            fold_idx: int, optional
                Index of the current fold in cross-validation
        """
        super().__init__()
        self.save_dir = save_dir
        self.fold_idx = fold_idx
        self.csv_file = os.path.join(save_dir, 'loss.csv')
        
        # Load existing data if file exists
        if os.path.exists(self.csv_file):
            self.existing_data = pd.read_csv(self.csv_file)
            self.last_epoch = len(self.existing_data)
        else:
            self.existing_data = pd.DataFrame()
            self.last_epoch = 0
            
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called when the train epoch ends.
        
        Parameters
        ----------
            trainer: pytorch_lightning.Trainer
                The trainer instance
            pl_module: pytorch_lightning.LightningModule
                The LightningModule instance
        """
        # Get current metrics
        metrics = trainer.callback_metrics
        
        # Create new row with correct epoch number
        new_row = {
            'epoch': trainer.current_epoch,  # Don't add last_epoch, just use current_epoch
            'val/base_loss': metrics.get('val/base_loss', 0).item(),
            'val_loss': metrics.get('val_loss', 0).item(),
            'val_psnr': metrics.get('val_psnr', 0).item(),
            'val_ssim': metrics.get('val_ssim', 0).item(),
            'train/base_loss': metrics.get('train/base_loss', 0).item(),
            'train_loss': metrics.get('train_loss', 0).item(),
            'train_psnr': metrics.get('train_psnr', 0).item(),
            'train_ssim': metrics.get('train_ssim', 0).item()
        }
        
        # Append to existing data
        self.existing_data = pd.concat([self.existing_data, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV
        self.existing_data.to_csv(self.csv_file, index=False)
        
class CustomProgressBar(TQDMProgressBar):
    """Custom progress bar that adds fold information"""
    def __init__(self, fold_idx=None, total_folds=None):
        super().__init__()
        self.fold_idx = fold_idx
        self.total_folds = total_folds
        
    def get_metrics(self, trainer, model):
        base_metrics = super().get_metrics(trainer, model)
        if self.fold_idx is not None and self.total_folds is not None:
            base_metrics["fold"] = f"{self.fold_idx + 1}/{self.total_folds}"
        return base_metrics
    
class CustomEarlyStopping(Callback):
    """
    Custom Early Stopping callback that prints a single clean summary per epoch.
    Terminates training if the monitored metric does not improve sufficiently.
    
    Note: This callback inherits from Callback (not EarlyStopping) so that
    Lightning's built‐in early stopping logic does not interfere.
    """
    def __init__(self, monitor='val_loss', patience=5, min_delta=0.001, mode='min', verbose=True):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.wait_count = 0
        self.initial_score = None
        self._already_called = False  # flag to ensure one update per epoch
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self._already_called = False

    def on_validation_epoch_end(self, trainer, pl_module):
        # Only execute this logic once per epoch.   
        if self._already_called:
            return
        self._already_called = True
        
        # Get the current value of the monitored metric.
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        # On first call, set the initial score.
        if self.initial_score is None:
            self.initial_score = current_score
            self.wait_count = 0
            if self.verbose:
                print(f"Epoch {trainer.current_epoch}: {self.monitor} = {current_score:.5f} (initial score set)")
            return

        # Compute improvement.
        if self.mode == 'min':
            improvement = self.initial_score - current_score  # positive means improvement
            #improvement = -improvement
        else:
            improvement = current_score - self.initial_score

        # Check if the improvement is large enough.
        if improvement >= self.min_delta:
            self.initial_score = current_score
            self.wait_count = 0
            msg = (f"Epoch {trainer.current_epoch}: {self.monitor} improved by {improvement:.5f} "
                   f"to {current_score:.5f}; wait_count reset to 0.")
        else:
            self.wait_count += 1
            msg = (f"Epoch {trainer.current_epoch}: {self.monitor} did not improve sufficiently "
                   f"(improvement = {improvement:.5f}); wait_count = {self.wait_count}/{self.patience}.")
            if self.wait_count >= self.patience:
                msg += " Early stopping triggered."
                trainer.should_stop = True

        if self.verbose:
            tqdm.write(msg)