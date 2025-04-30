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
import utils.mapping as mapping
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
    
    #print(f"test_results: valid truth shape {test_results['valid']['nf_truth'].shape}")
    #print(f"test_results valid pred shape: {test_results['valid']['nf_pred'].shape}") 
    
    # determine model type
    model_type = conf.model.arch
    
    # plot training and validation loss from recorded loss.csv once
    if not os.path.exists(os.path.join(results_dir, "loss_plots", "combined_loss.pdf")):
        os.makedirs(os.path.join(results_dir, "loss_plots"), exist_ok=True)
        print("Created directory: loss_plots")
        print("\nGenerating loss plot...")
        eval.plot_loss(conf, save_dir=results_dir, save_fig=True)
        
    # de-standardize results if necessary
    if conf.data.standardize:
        stats_path = os.path.join(conf.paths.data, 'preprocessed_data', 'global_stats.pt')
        for set in ['train', 'valid']:
            test_results[set]['nf_truth'] = mapping.destandardize(test_results[set]['nf_truth'], stats_path)
            test_results[set]['nf_pred'] = mapping.destandardize(test_results[set]['nf_pred'], stats_path)
        print("Destandardized test results.")
        
    # Create subdirectories for different types of results
    wl = str(conf.data.wv_eval)
    plots_dir = os.path.join(results_dir, f"eval_{wl}")
    os.makedirs(plots_dir, exist_ok=True)
    metrics_dir = os.path.join(plots_dir, "performance_metrics")
    dft_dir = os.path.join(plots_dir, "dft_plots")
    flipbook_dir = os.path.join(plots_dir, "flipbooks")
    misc_dir = os.path.join(plots_dir, "misc_plots")
    
    if model_type == "inverse":
        # Use specialized inverse model analysis
        train_stats, valid_stats = eval.analyze_inverse_results(test_results, save_fig=True, save_dir=plots_dir)
        if conf.model.inverse_strategy == 0: # results are design parameter predictions only
            return train_stats, valid_stats    
    
    for directory in [metrics_dir, dft_dir, flipbook_dir, misc_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    if conf.trainer.plot_ssim_corr:
        print("\n Computing Correlation Plots...")
        eval.analyze_field_correlations(test_results, resub=True,save_fig=True, 
                                        save_dir=plots_dir, arch=model_type, fold_num=fold_num)
        eval.analyze_field_correlations(test_results, resub=False, save_fig=True, 
                                        save_dir=plots_dir, arch=model_type, fold_num=fold_num)
        
    # compute relevant metrics across folds
    if model_type != 'autoencoder':
        plot_mse = True if conf.model.arch != 'mlp' and conf.model.arch != 'cvnn' else False
        print("\nComputing and saving metrics...")
        eval.metrics(test_results, dataset='train', save_fig=True, save_dir=plots_dir, plot_mse=plot_mse)
        eval.metrics(test_results, dataset='valid', save_fig=True, save_dir=plots_dir, plot_mse=plot_mse)
        # compute SSIM #TODO combine this with the above
        eval.compute_field_ssim(test_results, resub=True, save_fig=True, save_dir=plots_dir, arch=model_type)
        eval.compute_field_ssim(test_results, resub=False, save_fig=True, save_dir=plots_dir, arch=model_type)
    
    # visualize performance with DFT fields
    print("\nGenerating DFT field plots...")
    eval.plot_dft_fields(test_results, resub=True, sample_idx=10, save_fig=True, 
                         save_dir=plots_dir, arch=model_type, format='polar',
                         fold_num=fold_num)
    #if model_type == 'lstm' or model_type == 'convlstm':
    eval.plot_absolute_difference(test_results, resub=True, sample_idx=10, 
                                  save_fig=True, save_dir=plots_dir,
                                  arch=model_type, fold_num=fold_num, fixed_scale=True)
    
    # visualize performance with animation
    if model_type not in ['autoencoder', 'mlp', 'cvnn', 'inverse']:
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
    phase1_name = conf.pipeline[0].phase_name
    phase1_model = trained_models[phase1_name]
    processor1 = conf.pipeline[0].processor
    data_module.processor = MLPProcessor() if processor1 == 'MLPProcessor' else TemporalProcessor()
    data_module.setup(stage='test')
    # print dataset near fields size
    
    conf_copy = conf.copy()
    conf_copy.model.arch = conf.pipeline[0].model_arch
    phase1_results = evaluate_model(phase1_model, data_module, conf_copy, conf.pipeline[0].model_arch)
    
    # call the plotting handler for phase1 model
    plotting(conf, phase1_results, os.path.join(results_dir, conf.pipeline[0].model_arch))
    
    print(f"Evaluating Time-Series Model")
    
    # Convert results to tensor format for insertion into phase2's valid data
    phase1_preds = {
        'train': torch.from_numpy(phase1_results['train']['nf_pred']),
        'valid': torch.from_numpy(phase1_results['valid']['nf_pred'])
    }
    
    # Then evaluate phase2 model
    phase2_name = conf.pipeline[1].phase_name
    phase2_model = trained_models[phase2_name]
    processor2 = conf.pipeline[1].processor
    data_module.processor = TemporalProcessor() if processor2 == 'TemporalProcessor' else MLPProcessor()
    # Update the datamodule's data with MLP predictions
    data_module.update_near_fields(phase1_preds, phase1_name)
    conf_copy.model.arch = conf.pipeline[1].model_arch
    if conf.pipeline[0].model_arch == conf.pipeline[1].model_arch == 'mlp':
        conf_copy.model.mlp_strategy = 3 # doubled up MLP's mean's network 2, which means mlp_strategy = 3
    phase2_results = evaluate_model(phase2_model, data_module, conf_copy, conf.pipeline[1].model_arch)
    
    # call the plotting handler for LSTM
    plotting(conf, phase2_results, os.path.join(results_dir, conf.pipeline[1].model_arch))

def single_model_eval(conf, data_module=None):
    # use current config to get results directory
    results_dir = conf.paths.results
    
    # init datamodule
    if data_module is None:
        #TODO temporarily not using the saved one
        '''# setup new parameter manager based on saved parameters
        saved_conf = load_config(os.path.join(results_dir, 'config.yaml'))
        # update select parameters to match current run
        saved_conf.data.wv_eval = conf.data.wv_eval
        
        # init datamodule
        data_module = datamodule.select_data(saved_conf)'''

        data_module = datamodule.select_data(conf)
        #data_module.prepare_data()
        data_module.setup(stage='test')
    #else:
    #    saved_conf = conf
        
    # Load model checkpoint
    model_path = os.path.join(results_dir, 'model.ckpt')
    model_instance = model_loader.select_model(conf.model)
    model_instance.load_state_dict(torch.load(model_path)['state_dict'])

    # empty logger so as not to mess with loss.csv
    logger = None

    # Checkpoint, EarlyStopping, ProgressBar
    checkpoint_path = conf.paths.results
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
    
    # ensure test results are empty so we can populate them
    model_instance.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                                    'valid': {'nf_pred': [], 'nf_truth': []}}
    
    # setup the trainer
    trainer = train.configure_trainer(conf, logger, checkpoint_callback, early_stopping, progress_bar)
    
    if (conf.trainer.cross_validation):
        with open(os.path.join(results_dir, "split_info.yaml"), 'r') as f:
            split_info = yaml.safe_load(f)
        train_idx = split_info["train_idx"]
        val_idx = split_info["val_idx"]
        data_module.setup_fold(train_idx, val_idx)
    
    # perform testing
    trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
    
    # evaluate
    if conf.model.arch == 'modelstm':
        # fetch decoding params
        P = data_module.P
        mean_vec = data_module.mean_vec
        
        # Process both 'train' and 'valid' splits for both prediction and truth.
        for split in ['train', 'valid']:
            for key in ['nf_pred', 'nf_truth']:
                model_instance.test_results[split][key] = process_svd_results(
                    model_instance.test_results[split][key], P, mean_vec
                )

    plotting(conf, model_instance.test_results, 
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
    