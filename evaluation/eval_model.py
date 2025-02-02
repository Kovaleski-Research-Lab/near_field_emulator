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
from conf.schema import load_config

def plotting(conf, test_results, results_dir, fold_num=None):
    """
    The generation of a variety of plots and performance metrics
    """
    # plot training and validation loss from recorded loss.csv once
    if not os.path.exists(os.path.join(conf.paths.results, "loss_plots", "loss.pdf")):
        os.makedirs(os.path.join(conf.paths.results, "loss_plots"), exist_ok=True)
        print("Created directory: loss_plots")
        print("\nGenerating loss plots...")
        eval.plot_loss(conf, save_fig=True)
        
    # Create subdirectories for different types of results
    wl = str(conf.data.wv_eval)
    results_dir = os.path.join(results_dir, f"eval_{wl}")
    os.makedirs(results_dir, exist_ok=True)
    metrics_dir = os.path.join(results_dir, "performance_metrics")
    dft_dir = os.path.join(results_dir, "dft_plots")
    flipbook_dir = os.path.join(results_dir, "flipbooks")
    
    for directory in [metrics_dir, dft_dir, flipbook_dir]:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # determine model type
    model_type = conf.model.arch
        
    # compute relevant metrics across folds
    if model_type != 'autoencoder':
        plot_mse = True if conf.model.arch != 'mlp' and conf.model.arch != 'cvnn' else False
        print("\nComputing and saving metrics...")
        eval.metrics(test_results, dataset='train', save_fig=True, save_dir=results_dir, plot_mse=plot_mse)
        eval.metrics(test_results, dataset='valid', save_fig=True, save_dir=results_dir, plot_mse=plot_mse)
    
    # visualize performance with DFT fields
    print("\nGenerating DFT field plots...")
    eval.plot_dft_fields(test_results, resub=True, sample_idx=10, save_fig=True, 
                         save_dir=results_dir, arch=model_type, format='polar',
                         fold_num=fold_num)
    #if model_type == 'lstm' or model_type == 'convlstm':
    eval.plot_absolute_difference(test_results, resub=True, sample_idx=10, 
                                  save_fig=True, save_dir=results_dir,
                                  arch=model_type, fold_num=fold_num)
    
    # visualize performance with animation
    if model_type != 'autoencoder' and model_type != 'mlp':
        print("\nGenerating field animations...")
        eval.animate_fields(test_results, dataset='valid', 
                            seq_len=conf.model.seq_len, save_dir=results_dir)
    
    print(f"\nEvaluation complete. All results saved to: {results_dir}")

def run(conf):
    # use current config to get results directory
    results_dir = conf.paths.results
    
    # setup new parameter manager based on saved parameters
    saved_conf = load_config(os.path.join(results_dir, 'config.yaml'))
    
    # update select parameters to match current run
    saved_conf.data.wv_eval = conf.data.wv_eval
    saved_conf.directive = conf.directive # just to be safe
        
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
    
    # init datamodule
    data_module = datamodule.select_data(saved_conf)
    data_module.prepare_data()
    data_module.setup(stage='test')
    
    if (saved_conf.trainer.cross_validation):
        with open(os.path.join(results_dir, "split_info.yaml"), 'r') as f:
            split_info = yaml.safe_load(f)
        train_idx = split_info["train_idx"]
        val_idx = split_info["val_idx"]
        data_module.setup_fold(train_idx, val_idx)
    else: # cross validation was not conducted
        data_module.setup_og()
    
    # perform testing
    trainer.test(model_instance, dataloaders=[data_module.val_dataloader(), data_module.train_dataloader()])
    
    # evaluate
    if saved_conf.model.arch == 'modelstm':
        
        # print a bunch of stuff
        print(f"model_instance.test_results['valid']['nf_pred']: {model_instance.test_results['valid']['nf_pred']}")
        print(f"model_instance.test_results['valid']['nf_truth'].shape: {model_instance.test_results['valid']['nf_truth'].shape}")
        print(f"model_instance.test_results['train']['nf_pred'].shape: {model_instance.test_results['train']['nf_pred'].shape}")
        print(f"model_instance.test_results['train']['nf_truth'].shape: {model_instance.test_results['train']['nf_truth'].shape}")
        print(f"data_module.P.shape: {data_module.P.shape}")
        print(f"data_module.mean_vec.shape: {data_module.mean_vec.shape}")
        
        nf_pred_valid = model_instance.test_results['valid']['nf_pred'] * 180 / np.pi
        nf_truth_valid = model_instance.test_results['valid']['nf_truth'] * 180 / np.pi
        nf_pred_train = model_instance.test_results['train']['nf_pred'] * 180 / np.pi
        nf_truth_train = model_instance.test_results['train']['nf_truth'] * 180 / np.pi    
        
        print(f"nf_pred_valid.shape before decode_dataset called: {nf_pred_valid.shape}")
        print(f"nf_pred_valid.dtype: {nf_pred_valid.dtype}")
        
        # decode the data
        P = data_module.P
        mean_vec = data_module.mean_vec
        
        nf_pred_valid = torch.from_numpy(nf_pred_valid)
        nf_truth_valid = torch.from_numpy(nf_truth_valid)
        nf_pred_train = torch.from_numpy(nf_pred_train)
        nf_truth_train = torch.from_numpy(nf_truth_train)
        
        nf_pred_valid = modes.decode_dataset(nf_pred_valid.squeeze(3), P, mean_vec)
        nf_truth_valid = modes.decode_dataset(nf_truth_valid.squeeze(3), P, mean_vec)
        nf_pred_train = modes.decode_dataset(nf_pred_train.squeeze(3), P, mean_vec)
        nf_truth_train = modes.decode_dataset(nf_truth_train.squeeze(3), P, mean_vec)

        print(f"nf_pred_valid.shape after decode_dataset called: {nf_pred_valid.shape}")
        
        # update the test results
        model_instance.test_results['valid']['nf_pred'] = nf_pred_valid.detach().cpu().numpy()
        model_instance.test_results['valid']['nf_truth'] = nf_truth_valid.detach().cpu().numpy()
        model_instance.test_results['train']['nf_pred'] = nf_pred_train.detach().cpu().numpy()
        model_instance.test_results['train']['nf_truth'] = nf_truth_train.detach().cpu().numpy()
        
        
        '''svd_params_valid = model_instance.test_results['valid']
        svd_params_train = model_instance.test_results['train']
        model_instance.test_results['valid']['nf_pred'] = modes.reconstruct_full_dataset(svd_params_valid['nf_pred'], saved_conf)
        model_instance.test_results['valid']['nf_truth'] = modes.reconstruct_full_dataset(svd_params_valid['nf_truth'], saved_conf)
        model_instance.test_results['train']['nf_pred'] = modes.reconstruct_full_dataset(svd_params_train['nf_pred'], saved_conf)
        model_instance.test_results['train']['nf_truth'] = modes.reconstruct_full_dataset(svd_params_train['nf_truth'], saved_conf)
        print(model_instance.test_results['valid']['nf_pred'].shape)'''
        
    if conf.model.full_pipeline:
        if conf.model.arch == 'mlp':
            # stack pred train and valid together into one tensor and save to a torch .pt file
            pred_train = torch.from_numpy(model_instance.test_results['train']['nf_pred'])
            pred_valid = torch.from_numpy(model_instance.test_results['valid']['nf_pred'])
            pred = torch.cat((pred_train, pred_valid), dim=0)
            torch.save(pred, os.path.join(results_dir, 'preds.pt'))

    plotting(saved_conf, model_instance.test_results, 
            results_dir)
    
    