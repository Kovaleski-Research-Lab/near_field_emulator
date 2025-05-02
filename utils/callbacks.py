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
        self.metrics = []
        
        # Set up the CSV file path
        if self.fold_idx is not None:
            os.makedirs(os.path.join(self.save_dir, 'losses'), exist_ok=True)
            self.csv_path = os.path.join(self.save_dir, 'losses', f"fold{self.fold_idx+1}.csv")
        else:
            self.csv_path = os.path.join(self.save_dir, 'loss.csv')
            
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
        # Get metrics from the current epoch
        metrics = trainer.callback_metrics
        if metrics:
            # Convert metrics to dict format and add epoch
            metrics_dict = {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}
            metrics_dict['epoch'] = trainer.current_epoch
            self.metrics.append(metrics_dict)
            
            # Save to CSV
            self._save_metrics()
            
    def _save_metrics(self):
        """
        Save the collected metrics to a CSV file.
        """
        if not self.metrics:
            return
            
        # Get all metric keys
        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())
        
        # Ensure 'epoch' is the first column
        if 'epoch' in metrics_keys:
            metrics_keys.remove('epoch')
            metrics_keys.insert(0, 'epoch')
        
        # Write to CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_keys)
            writer.writeheader()
            writer.writerows(self.metrics)
            
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