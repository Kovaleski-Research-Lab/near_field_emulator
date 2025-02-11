# actual driver for evaluation - produces metrics/plots/etc/
import torch
import matplotlib.pyplot as plt
import sys
import os
import yaml
import numpy as np
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping
import gc
import warnings

warnings.filterwarnings("ignore")

sys.path.append('../')
import evaluation.evaluation as eval
import evaluation.inference as inference
import utils.model_loader as model_loader
from core import datamodule, custom_logger, train, modes
from core.datamodule import MLPProcessor, TemporalProcessor
from conf.schema import load_config

# pipeline function mappings
'''PIPELINE_PROCESSORS = {
    "MLPProcessor": lambda: __import__('core.datamodule', fromlist=['MLPProcessor']).MLPProcessor(),
    "TemporalProcessor": lambda: __import__('core.datamodule', fromlist=['TemporalProcessor']).TemporalProcessor(),
}'''

def plotting(conf, test_results, results_dir, fold_num=None):
    """
    The generation of a variety of plots and performance metrics
    """
    # plot training and validation loss from recorded loss.csv once
    if not os.path.exists(os.path.join(results_dir, "loss_plots", "loss.pdf")):
        os.makedirs(os.path.join(results_dir, "loss_plots"), exist_ok=True)
        print("Created directory: loss_plots")
        print("\nGenerating loss plots...")
        eval.plot_loss(conf, save_dir=results_dir, save_fig=True)
        
    # Create subdirectories for different types of results
    wl = str(conf.data.wv_eval)
    plots_dir = os.path.join(results_dir, f"eval_{wl}")
    os.makedirs(plots_dir, exist_ok=True)
    metrics_dir = os.path.join(plots_dir, "performance_metrics")
    dft_dir = os.path.join(plots_dir, "dft_plots")
    flipbook_dir = os.path.join(plots_dir, "flipbooks")
    
    for directory in [metrics_dir, dft_dir, flipbook_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # determine model type
    model_type = conf.model.arch
        
    # compute relevant metrics across folds
    if model_type != 'autoencoder':
        plot_mse = True if conf.model.arch != 'mlp' and conf.model.arch != 'cvnn' else False
        print("\nComputing and saving metrics...")
        eval.metrics(test_results, dataset='train', save_fig=True, save_dir=plots_dir, plot_mse=plot_mse)
        eval.metrics(test_results, dataset='valid', save_fig=True, save_dir=plots_dir, plot_mse=plot_mse)
    
    # visualize performance with DFT fields
    print("\nGenerating DFT field plots...")
    eval.plot_dft_fields(test_results, resub=True, sample_idx=10, save_fig=True, 
                         save_dir=plots_dir, arch=model_type, format='polar',
                         fold_num=fold_num)
    #if model_type == 'lstm' or model_type == 'convlstm':
    eval.plot_absolute_difference(test_results, resub=True, sample_idx=10, 
                                  save_fig=True, save_dir=plots_dir,
                                  arch=model_type, fold_num=fold_num)
    
    # visualize performance with animation
    if model_type != 'autoencoder' and model_type != 'mlp':
        print("\nGenerating field animations...")
        eval.animate_fields(test_results, dataset='valid', 
                            seq_len=conf.model.seq_len, save_dir=plots_dir)
    
    print(f"\nEvaluation complete. All results saved to: {plots_dir}")
    
def process_svd_results(results_arr, P, mean_vec):
    """denormalizes, decodes, and returns in numpy"""
    result = results_arr * 180 / np.pi
    result_tensor = torch.from_numpy(result)
    result_tensor = modes.decode_dataset(result_tensor.squeeze(3), P, mean_vec)
    return result_tensor.detach().cpu().numpy()

#--------------------------------
# Evaluation Process Functions
#--------------------------------

def evaluate_model(model_instance, data_module, conf, phase_name):
    """Evaluate a single model and return its results"""
    logger = None  # empty logger to not interfere with existing logs
    
    checkpoint_path = os.path.join(conf.paths.results, phase_name)
    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )
    
    early_stopping = train.CustomEarlyStopping(
        monitor='val_loss',
        patience=conf.trainer.patience,
        min_delta=conf.trainer.min_delta,
        mode='min' if conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )
    
    progress_bar = train.CustomProgressBar()
    
    # Reset test results
    model_instance.test_results = {
        'train': {'nf_pred': [], 'nf_truth': []},
        'valid': {'nf_pred': [], 'nf_truth': []}
    }
    
    trainer = train.configure_trainer(conf, logger, checkpoint_callback, 
                                   early_stopping, progress_bar)
    
    '''# debugging
    print(f"checking the contents of a data_module.train_dataloader() batch...")
    batch = next(iter(data_module.train_dataloader()))
    item1, item2 = batch
    
    print(f"Near fields (or samples) shape: {item1.shape}")
    print(f"Radii (or labels) shape: {item2.shape}")'''
    
    # Run evaluation
    trainer.test(model_instance, 
                dataloaders=[data_module.val_dataloader(), 
                            data_module.train_dataloader()])
    
    return model_instance.test_results

def pipeline_eval(conf, data_module, trained_models):
    """Evaluate a multi-phase pipeline, handling data dependencies between phases"""
    results_dir = conf.paths.results
    
    print(f"Evaluating MLP")
    
    # First evaluate MLP with MLP processor
    mlp_model = trained_models['mlp']
    mlp_processor = MLPProcessor()
    data_module.processor = mlp_processor
    data_module.setup(stage='test')
    # print dataset near fields size
    
    conf_copy = conf.copy()
    conf_copy.model.arch = 'mlp'
    mlp_results = evaluate_model(mlp_model, data_module, conf_copy, 'mlp')
    
    # call the plotting handler for MLP
    plotting(conf, mlp_results, os.path.join(results_dir, 'mlp'))
    
    print(f"Evaluating Time-Series Model")
    
    # Convert MLP results to tensor format for LSTM input
    mlp_preds = {
        'train': torch.from_numpy(mlp_results['train']['nf_pred']),
        'valid': torch.from_numpy(mlp_results['valid']['nf_pred'])
    }
    
    # Then evaluate LSTM with temporal processor
    lstm_model = trained_models['lstm']
    temporal_processor = TemporalProcessor()
    data_module.processor = temporal_processor
    # Update the datamodule's data with MLP predictions
    data_module.update_near_fields(mlp_preds)
    conf_copy.model.arch = 'lstm'
    lstm_results = evaluate_model(lstm_model, data_module, conf_copy, 'lstm')
    
    # call the plotting handler for LSTM
    plotting(conf, lstm_results, os.path.join(results_dir, 'lstm'))

def single_model_eval(conf, data_module=None):
    # use current config to get results directory
    results_dir = conf.paths.results
    
    # init datamodule
    if data_module is None:
        # setup new parameter manager based on saved parameters
        saved_conf = load_config(os.path.join(results_dir, 'config.yaml'))
        # update select parameters to match current run
        saved_conf.data.wv_eval = conf.data.wv_eval
        
        # init datamodule
        data_module = datamodule.select_data(saved_conf)
        #data_module.prepare_data()
        data_module.setup(stage='test')
    else:
        saved_conf = conf
        
    # Load model checkpoint
    model_path = os.path.join(results_dir, 'model.ckpt')
    model_instance = model_loader.select_model(saved_conf.model)
    model_instance.load_state_dict(torch.load(model_path)['state_dict'])

    # empty logger so as not to mess with loss.csv
    logger = None

    # Checkpoint, EarlyStopping, ProgressBar
    checkpoint_path = saved_conf.paths.results
    filename = 'model'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename=filename,
        save_top_k=1,
        monitor='val_loss',
        mode='min' if saved_conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    early_stopping = train.CustomEarlyStopping(
        monitor='val_loss',
        patience=saved_conf.trainer.patience,
        min_delta=saved_conf.trainer.min_delta,
        mode='min' if saved_conf.model.objective_function == 'mse' else 'max',
        verbose=True
    )

    progress_bar = train.CustomProgressBar()
    
    # ensure test results are empty so we can populate them
    model_instance.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                                    'valid': {'nf_pred': [], 'nf_truth': []}}
    
    # setup the trainer
    trainer = train.configure_trainer(saved_conf, logger, checkpoint_callback, early_stopping, progress_bar)
    
    if (saved_conf.trainer.cross_validation):
        with open(os.path.join(results_dir, "split_info.yaml"), 'r') as f:
            split_info = yaml.safe_load(f)
        train_idx = split_info["train_idx"]
        val_idx = split_info["val_idx"]
        data_module.setup_fold(train_idx, val_idx)
    
    # perform testing
    trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
    
    # evaluate
    if saved_conf.model.arch == 'modelstm':
        # fetch decoding params
        P = data_module.P
        mean_vec = data_module.mean_vec
        
        # Process both 'train' and 'valid' splits for both prediction and truth.
        for split in ['train', 'valid']:
            for key in ['nf_pred', 'nf_truth']:
                model_instance.test_results[split][key] = process_svd_results(
                    model_instance.test_results[split][key], P, mean_vec
                )
        
    if conf.model.full_pipeline:
        if conf.model.arch == 'mlp':
            # stack pred train and valid together into one tensor and save to a torch .pt file
            pred_train = torch.from_numpy(model_instance.test_results['train']['nf_pred'])
            pred_valid = torch.from_numpy(model_instance.test_results['valid']['nf_pred'])
            pred = torch.cat((pred_train, pred_valid), dim=0)
            torch.save(pred, os.path.join(results_dir, 'preds.pt'))

    plotting(saved_conf, model_instance.test_results, 
            results_dir)
    
#--------------------------------
# Main Evaluation Entry Point
#--------------------------------
    
def run(conf, data_module=None, trained_models=None):
    """Main evaluation entry point"""
    if conf.model.arch == 'mlp-lstm':
        pipeline_eval(conf, data_module, trained_models)
    else:
        single_model_eval(conf, data_module)
    