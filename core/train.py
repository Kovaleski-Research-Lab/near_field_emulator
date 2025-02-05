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
from sklearn.model_selection import KFold
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping, Callback
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm
from pytorch_lightning.plugins.environments import SLURMEnvironment
from tqdm.auto import tqdm

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import datamodule, custom_logger
from core.datamodule import MLPProcessor, TemporalProcessor
from utils import model_loader, mapping
from conf.schema import load_config

# debugging
#logging.basicConfig(level=logging.DEBUG)

#--------------------------------
# Utilities
#--------------------------------

# pipeline function mappings
PIPELINE_PROCESSORS = {
    "MLPProcessor": lambda: __import__('core.datamodule', fromlist=['MLPProcessor']).MLPProcessor(),
    "TemporalProcessor": lambda: __import__('core.datamodule', fromlist=['TemporalProcessor']).TemporalProcessor(),
}

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()
    
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
    Lightning’s built‐in early stopping logic does not interfere.
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

           
def configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar):
    """Create and return a configured Trainer instance."""
    trainer_kwargs = {
        'logger': logger,
        'max_epochs': conf.trainer.num_epochs,
        'deterministic': True,
        'enable_progress_bar': True,
        'enable_model_summary': True,
        'default_root_dir': conf.paths.root,
        'callbacks': [checkpoint_callback, early_stopping, progress_bar],
        'check_val_every_n_epoch': conf.trainer.valid_rate,
        'num_sanity_val_steps': 1,
        'log_every_n_steps': 1
    }

    if torch.cuda.is_available():
        trainer_kwargs.update({
            'accelerator': 'cuda',
        })
        logging.debug("Training with GPUs")
    else:
        trainer_kwargs.update({
            'accelerator': 'cpu'
        })
        logging.debug("Training with CPUs")

    return Trainer(**trainer_kwargs)

def save_best_model(conf, best_model_path, phase_name=None, n_splits=None):
    """Save the best model checkpoint and clean up temporary ones."""
    if best_model_path:
        results_dir = conf.paths.results
        if phase_name:
            results_dir = os.path.join(results_dir, phase_name)
        #os.makedirs(results_dir, exist_ok=True)
        checkpoint_path = os.path.join(results_dir, 'model.ckpt')
        
        best_model = torch.load(best_model_path)
        torch.save(best_model, checkpoint_path)
        logging.info(f"Saved best overall model to {checkpoint_path}")
        
        # If cross-validation was used, remove fold checkpoints
        if n_splits is not None:
            for fold in range(n_splits):
                temp_fold_ckpt = os.path.join(conf.paths.results, f"model_fold{fold + 1}.ckpt")
                if os.path.exists(temp_fold_ckpt):
                    os.remove(temp_fold_ckpt)
                    
def record_split_info(fold_idx, train_idx, val_idx, results_dir):
    """
    Records index info for the best performing fold for later reference during evaluation.
    """
    fold_num = fold_idx + 1
    split_info = {
        "best_fold": fold_num,
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist()
    }
    with open(os.path.join(results_dir, "split_info.yaml"), 'w') as f:
        yaml.dump(split_info, f)


#--------------------------------
# Training Pipelines
#--------------------------------

def train_phase(conf, data_module, phase_name, custom_processor=None, fold_idx=None):
    """
    Train one phase with the provided datamodule and configuration.
    
    Optionally, if custom_processor is provided (a class or function that returns a processor),
    update the datamodule to use it before calling setup().
    
    Returns the trained model instance.
    """
    # optionally update the datamodule's processor
    if custom_processor is not None:
        data_module.processor = custom_processor
    # Re-setup the datamodule (raw data is cached, so this re-formats the dataset)
    print(f"\nIN TRAIN_PHASE, calling data_module.setup(stage='fit')")
    print(f"the phase name is {phase_name}")
    data_module.setup(stage='fit')
    
    
    # select model instance according to the updated configuration
    model_instance = model_loader.select_model(conf.model)
    # create the results directory
    save_dir = os.path.join(conf.paths.results, phase_name)
    os.makedirs(save_dir, exist_ok=True)
    
    logger = custom_logger.Logger(
        save_dir=save_dir,
        name=f"{conf.model.model_id}_{phase_name}",
        version=0
    )

    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=False
    )
    early_stopping = CustomEarlyStopping(
        monitor='val_loss',
        patience=conf.trainer.patience,
        min_delta=conf.trainer.min_delta,
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )
    progress_bar = CustomProgressBar(fold_idx, None) if fold_idx is not None else CustomProgressBar()

    trainer = configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar)
    
    '''# debugging
    print(f"checking the contents of a data_module.train_dataloader() batch...")
    batch = next(iter(data_module.train_dataloader()))
    item1, item2 = batch
    
    print(f"We're checking here in train_phase for phase_name: {phase_name}")
    print(f"Near fields (or samples) shape: {item1.shape}")
    print(f"Radii (or labels) shape: {item2.shape}")'''
    
    trainer.fit(model_instance, data_module)
    
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_state = torch.load(best_model_path)['state_dict']
        model_instance.load_state_dict(best_state)
        
    # save the best model with phase name using save_best_model
    save_best_model(conf, best_model_path, phase_name, n_splits=None)
    
    return model_instance

def pipeline_train(conf):
    """Run a multi-phase training pipeline using the pre-constructed pipeline config"""
    dm = datamodule.select_data(conf)
    
    conf_copy = conf.copy()
    trained_models = {}
    
    for phase in conf.pipeline:
        print(f"=== Starting phase: {phase.phase_name} ===")
        # update configuration with the desired model architecture for this phase
        conf_copy.model.arch = phase.model_arch
        
        # Get the actual processor
        processor = None
        if phase.processor == 'MLPProcessor':
            processor = MLPProcessor()
        elif phase.processor == 'TemporalProcessor':
            processor = TemporalProcessor()
        else:
            raise ValueError(f"Unknown processor: {phase.processor}")
        '''if phase.processor:
            processor_callable = PIPELINE_PROCESSORS.get(phase.processor)
            if processor_callable is None:
                raise ValueError(f"Unknown processor: {phase.processor}")
            processor = processor_callable()'''
            
        # train this phase
        model = train_phase(conf_copy, dm, phase.phase_name, 
                          custom_processor=processor)
        trained_models[phase.phase_name] = model
        
    # restore
    conf_copy.model.arch = 'mlp-lstm'
        
    return trained_models, dm

# --------------------------------
# Legacy Single-Phase Training Functions
# --------------------------------
                  
def train_once(conf, data_module):
    """
    Train without cross-validation, utilizing the train/valid split originally
    established during data preprocessing (should be 80/20) 
    The split: core/preprocess_data.py --> separate_datasets()
    Tagged in core/datamodule.py --> load_pickle_data()
    """
    #data_module.setup_og()

    model_instance = model_loader.select_model(conf.model)
    logger = custom_logger.Logger(
        save_dir=conf.paths.results,
        name=conf.model.model_id,
        version=0
    )

    # Checkpoint and EarlyStopping
    checkpoint_path = conf.paths.results
    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=False
    )

    early_stopping = CustomEarlyStopping(
        monitor='val_loss',
        patience=conf.trainer.patience,
        min_delta=conf.trainer.min_delta,
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    progress_bar = CustomProgressBar()

    trainer = configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar)

    # Train
    trainer.fit(model_instance, data_module)
    
    # Save best model
    best_model_path = checkpoint_callback.best_model_path
    save_best_model(conf, best_model_path, n_splits=None)


def train_with_cross_validation(conf, data_module):
    """Train using K-Fold Cross Validation."""
    full_dataset = data_module.dataset
    n_splits = conf.data.n_folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=conf.seed_value)

    best_val_loss = float('-inf') if conf.model.objective_function == 'psnr' else float('inf')
    best_model_path = None

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        logging.info(f"Fold {fold_idx + 1}/{n_splits}")
        if fold_idx > 0:
            clear_memory()

        model_instance = model_loader.select_model(conf.model, fold_idx)
        data_module.setup_fold(train_idx, val_idx)

        logger = custom_logger.Logger(
            save_dir=conf.paths.results,
            name=f"{conf.model.model_id}_fold{fold_idx + 1}", 
            version=0, 
            fold_idx=fold_idx
        )

        checkpoint_path = conf.paths.results
        filename = f'model_fold{fold_idx + 1}'
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_path,
            filename=filename,
            save_top_k=1,
            monitor='val_loss',
            mode='min' if conf.model.objective_function == 'mse' else 'max',
            verbose=True
        )
        
        early_stopping = CustomEarlyStopping(
            monitor='val_loss',
            patience=conf.trainer.patience,
            min_delta=conf.trainer.min_delta,
            mode='min' if conf.model.objective_function == 'mse' else 'max',
            verbose=True
        )

        progress_bar = CustomProgressBar(fold_idx, n_splits)
        trainer = configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar)

        # Training
        trainer.fit(model_instance, data_module)

        current_val_loss = checkpoint_callback.best_model_score.item()

        if conf.model.objective_function == 'psnr':
            if current_val_loss > best_val_loss:
                best_val_loss = current_val_loss
                best_model_path = checkpoint_callback.best_model_path
                logging.info(f"New best model found in fold {fold_idx + 1} with PSNR: {best_val_loss:.6f}")
                record_split_info(fold_idx, train_idx, val_idx, os.path.dirname(best_model_path))
        else:
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_path = checkpoint_callback.best_model_path
                logging.info(f"New best model found in fold {fold_idx + 1} with val loss: {best_val_loss:.6f}")
                record_split_info(fold_idx, train_idx, val_idx, os.path.dirname(best_model_path))

        # Testing if needed
        if conf.trainer.include_testing:
            trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
            fold_results.append(model_instance.test_results)
        else:
            base_path = os.path.dirname(best_model_path)
            # remove train_info and valid_info dirs from base_path #TODO: cleaner to have this in logger
            shutil.rmtree(os.path.join(base_path, 'train_info'))
            shutil.rmtree(os.path.join(base_path, 'valid_info'))

    # After all folds, save the best model
    save_best_model(conf, best_model_path, n_splits)

#--------------------------------
# Main Training Entry Point
#--------------------------------

def run(conf, data_module=None, pipeline=None):
    logging.debug("train.py() | running training")
    
    # save exact config file for later use
    os.makedirs(conf.paths.results, exist_ok=True)
    # copy args.config to results folder
    shutil.copy('conf/config.yaml', os.path.join(conf.paths.results, 'config.yaml'))

    if conf.model.arch == 'mlp-lstm':
        trained_models, data_module = pipeline_train(conf)
        return trained_models, data_module
    else:

        # Initialize: The datamodule
        if data_module is None:
            data_module = datamodule.select_data(conf)
            #data_module.prepare_data()
            data_module.setup(stage='fit')
        #else: # 2nd training pass
        #TODO: determine what/if things need to be recalled
        
        # run training
        if(conf.trainer.cross_validation):
            train_with_cross_validation(conf, data_module)
        else: # run once
            train_once(conf, data_module)
            
        trained_models = None
            
        return trained_models, data_module