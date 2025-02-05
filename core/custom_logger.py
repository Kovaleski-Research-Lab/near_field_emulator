#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import os
import csv
import shutil
import pickle
import logging
import numpy as np

from PIL import Image

from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.logger import rank_zero_experiment

#--------------------------------
# Initialize: Experiment Logger
#--------------------------------

class Writer:
    def __init__(self, path, name = "loss.csv", fold_idx=None):
        
        logging.debug("custom_logger.py - Initializing Writer")

        self.name = name
        self.path = path
        self.fold_idx = fold_idx
        self.metrics = []
        
        # Add fold-specific paths for good organization
        if self.fold_idx is not None:
            os.makedirs(os.path.join(self.path, 'losses'), exist_ok=True)
            self.path_metrics = os.path.join(self.path, 'losses', f"fold{self.fold_idx+1}.csv")
            #self.path_train = os.path.join(self.path, "train_info", f"fold{self.fold_idx+1}")
            #self.path_valid = os.path.join(self.path, "valid_info", f"fold{self.fold_idx+1}")
        else:
            self.path_metrics = os.path.join(self.path, self.name)
            #self.path_train = os.path.join(self.path, "train_info")
            #self.path_valid = os.path.join(self.path, "valid_info")

        # Ensure specific directories exist
        #os.makedirs(self.path_valid, exist_ok=True)
        #os.makedirs(self.path_train, exist_ok=True)
        
    #----------------------------
    # Update: Performance Metrics 
    #----------------------------

    def log_metrics(self, metrics_dict, step = None):
        logging.debug("Writer | logging metric dictionary")
        if step is None:
            step = len(self.metrics)
        self.metrics.append(metrics_dict)

    #----------------------------
    # Logging: Results
    #----------------------------

    def log_results(self, results, epoch, mode, count = 5, name = None):
        logging.debug("Writer | logging and saving results")

        if epoch is not None:
            name = name + "_" + str(epoch).zfill(count) + ".pkl"
        else:
            name = name + ".pkl"

        if(mode == "train"):
            path = self.path_train
        else:
            path = self.path_valid

        path_save = os.path.join(path, name)
        pickle.dump(results, open(path_save, "wb"))

    #----------------------------
    # Saving: Metrics
    #----------------------------

    def save(self):
        logging.debug("Writer | saving metric dictionary")

        last_m = {}
        for m in self.metrics:
            last_m.update(m)
        metrics_keys = list(last_m.keys())

        with open(self.path_metrics, "w", newline="") as f:
            self.writer = csv.DictWriter(f, fieldnames=metrics_keys)
            self.writer.writeheader()
            self.writer.writerows(self.metrics)

#--------------------------------
# Initialize: Experiment Logger
#--------------------------------

class Logger(Logger):
    def __init__(self, save_dir, name = "default", version = None, prefix = "", fold_idx=None):
        super().__init__()
        logging.debug("custom_logger.py - Initializing Logger")

        self._name = name 
        self._prefix = prefix
        self._experiment = None
        self._version = version
        self.fold_idx = fold_idx
        self._save_dir = save_dir
        self.fold_idx = fold_idx

        # Ensure directories exist
        #os.makedirs(self._save_dir, exist_ok=True)

    #----------------------------
    # Gather: Path (Root Folder)
    #----------------------------
    
    @property
    def root_dir(self) -> str:
        #if not self.name:
        #    return self.save_dir

        #return os.path.join(self.save_dir, self.name)
        return self.save_dir

    #----------------------------
    # Gather: Path (Log Folder)
    #----------------------------

    @property
    def log_dir(self):

        #version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        #log_dir = os.path.join(self.root_dir, version)

        return self.root_dir

    #----------------------------
    # Gather: Path (Save Folder)
    #----------------------------

    @property
    def save_dir(self):

        return self._save_dir

    #----------------------------
    # Gather: Experiment Version
    #----------------------------

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    #----------------------------
    # Gather: Experiment Title
    #----------------------------

    @property
    def name(self):
        return self._name

    #----------------------------
    # Log: Performance Metrics
    #----------------------------

    @rank_zero_only
    def log_metrics(self, metrics, step):
        self.experiment.log_metrics(metrics, step)

    #----------------------------
    # Log: Model Hyperparameters
    #----------------------------

    @rank_zero_only
    def log_hyperparams(self, params):

        pass

    #----------------------------
    # Initialize: Saver Actions
    #----------------------------

    @rank_zero_only
    def save(self):

        super().save()
        self.experiment.save()

    #----------------------------
    # Run: Post Training Code
    #----------------------------

    @rank_zero_only
    def finalize(self, status):
        self.save()

    #----------------------------
    # Run: Logger-Writer Object
    #----------------------------

    @property
    @rank_zero_experiment
    def experiment(self):

        if(self._experiment):

            return self._experiment

        #os.makedirs(self.root_dir, exist_ok=True)

        self._experiment = Writer(path = self.log_dir, fold_idx=self.fold_idx)

        return self._experiment