import pickle
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import scipy.stats as stats
import yaml
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

sys.path.append('../')
import utils.mapping as mapping
import utils.visualize as viz
fontsize = 8
font = FontProperties()
colors = ['darkgreen','purple','#4e88d9'] 
model_identifier = ""

def is_csv_empty(path):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return True
    except pd.errors.EmptyDataError:
        return True

    return False

def load_losses(fold_num, folder_path):
    """Loads losses for a specific fold"""
    path = os.path.join(folder_path, f"loss_fold{fold_num+1}.csv")
    if is_csv_empty(path):
        print("Empty CSV file")
        return False
    losses = pd.read_csv(path)
    return losses


def get_results(folder_path, n_folds, fold_num=1):

    train_path = os.path.join(folder_path, "train_info")
    valid_path = os.path.join(folder_path, "valid_info")

    losses = load_losses(fold_num, folder_path)

    train_results = pickle.load(open(os.path.join(train_path, f"fold{fold_num}", f"results_fold{fold_num}.pkl"), "rb"))
    valid_results = pickle.load(open(os.path.join(valid_path, f"fold{fold_num}", f"results_fold{fold_num}.pkl"), "rb"))
    
    return losses, train_results, valid_results

def get_all_results(folder_path, n_folds, resub=False):
    fold_results = []

    # Loop over all folds
    for fold_num in range(n_folds):
        # Define paths for the current fold
        if resub:
            train_path = os.path.join(folder_path, "train_info", f"fold{fold_num+1}", f"results.pkl")
            train_results = pickle.load(open(train_path, "rb"))
        else:
            train_results = None
        valid_path = os.path.join(folder_path, "valid_info", f"fold{fold_num+1}", f"results.pkl")
        
        # Load the results for this fold
        valid_results = pickle.load(open(valid_path, "rb"))

        # Load the losses for this fold (assuming your losses are saved in a similar way)
        losses = load_losses(fold_num, folder_path)

        fold_results.append({
            'train': train_results,    # Contains 'nf_truth' and 'nf_pred' for training
            'valid': valid_results,    # Contains 'nf_truth' and 'nf_pred' for validation
            'losses': losses           # Contains 'epoch', 'train_loss', and 'val_loss'
        })

    return fold_results
        
def save_eval_item(save_dir, eval_item, file_name, type):
    """Save metrics or plot(s) to a specified file."""
    if 'metrics' in type or type == 'evo':
        save_path = os.path.join(save_dir, "performance_metrics")
    elif type == 'loss':
        save_path = os.path.join(save_dir, "loss_plots")
    elif type == 'dft':
        save_path = os.path.join(save_dir, "dft_plots")
    else:
        return NotADirectoryError
    #os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, file_name)
    std_metrics = ["RMSE_First_Slice", "RMSE_Final_Slice"]
    if 'metrics' in type:
        with open(save_path, 'w') as file:
            for metric, value in eval_item.items():
                if metric in std_metrics:
                    file.write(f"{metric}: {value}\n")
                else:
                    file.write(f"{metric}: {value:.4f}\n")
    else:
        eval_item.savefig(save_path)
    print(f"Generated evaluation item: {type}")
    
def get_model_identifier(conf):
    """Construct a model identifier string for the plot title based on model parameters."""
    model_type = conf.model.arch
    title = conf.model.model_id
    lr = conf.model.learning_rate
    optimizer = conf.model.optimizer
    lr_scheduler = conf.model.lr_scheduler
    batch_size = conf.trainer.batch_size
    
    if model_type in ['mlp', 'cvnn']:
        mlp_layers = conf.model.mlp_real['layers']
        return f'{title} - lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, batch: {batch_size}, mlp_layers: {mlp_layers}'
    elif model_type in ['lstm', 'ae-lstm']:
        lstm_num_layers = conf.model.lstm.num_layers
        lstm_i_dims = conf.model.lstm.i_dims
        lstm_h_dims = conf.model.lstm.h_dims
        seq_len = conf.model.seq_len
        return (f'{title} - lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, '
                f'batch: {batch_size}, lstm_layers: {lstm_num_layers}, i_dims: {lstm_i_dims}, '
                f'h_dims: {lstm_h_dims}, seq_len: {seq_len}')
    elif model_type in ['convlstm', 'ae-convlstm']:
        in_channels = conf.model.convlstm.in_channels
        out_channels = conf.model.convlstm.out_channels
        kernel_size = conf.model.convlstm.kernel_size
        padding = conf.model.convlstm.padding
        return (f'{title} - lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, '
                f'batch: {batch_size}, in_channels: {in_channels}, out_channels: {out_channels}, '
                f'kernel_size: {kernel_size}, padding: {padding}')
    elif model_type == 'modelstm':
        method = conf.model.modelstm.method
        lstm_num_layers = conf.model.modelstm.num_layers
        lstm_i_dims = conf.model.modelstm.spatial * conf.model.modelstm.k * 2 + conf.model.modelstm.k
        lstm_h_dims = conf.model.modelstm.h_dims
        seq_len = conf.model.seq_len
        return (f'{title} - encoding: {method}, lr: {lr}, lr_scheduler: {lr_scheduler}, optimizer: {optimizer}, '
                f'batch: {batch_size}, lstm_layers: {lstm_num_layers}, i_dims: {lstm_i_dims}, '
                f'h_dims: {lstm_h_dims}, seq_len: {seq_len}')
    elif model_type == 'autoencoder':
        latent_dim = conf.model.autoencoder.latent_dim
        method = conf.model.autoencoder.method
        return (f'{title} - encoding: {method}, lr: {lr}, lr_scheduler: {lr_scheduler}, '
                f'optimizer: {optimizer}, batch: {batch_size}, latent_dim: {latent_dim}')
    else:
        return f'{title} - lr: {lr}, optimizer: {optimizer}, batch: {batch_size}'

def clean_loss_df(df):
    """
    Clean up the loss DataFrame which may have each epoch split into two lines.
    We group by epoch and take max() since one of train_loss/val_loss lines will be NaN on one row.
    """
    df = df.dropna(how='all')  # Drop completely empty rows
    # Convert columns to numeric
    for col in ['val_loss', 'epoch', 'train_loss']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Group by epoch and aggregate
    df = df.groupby('epoch', as_index=False).agg({'val_loss': 'max', 'train_loss': 'max'})
    df = df.set_index('epoch')
    df = df.sort_index()
    return df

def plot_loss(conf, save_dir, min_list=[None, None], max_list=[None, None], save_fig=False):
    model_identifier = get_model_identifier(conf)
    
    if conf.trainer.cross_validation:
        losses_path = os.path.join(save_dir, "losses")
        if not os.path.exists(losses_path):
            print(f"No losses directory found at {losses_path}.")
            return
        
        fold_files = [f for f in os.listdir(losses_path) if f.startswith('fold')]
        if not fold_files:
            print("No fold loss files found.")
            return
        
        train_losses = []
        val_losses = []
        for f in fold_files:
            path = os.path.join(losses_path, f)
            if os.path.getsize(path) == 0:
                print(f"Empty CSV file: {path}")
                continue
            df = pd.read_csv(path)
            df = clean_loss_df(df)

            train_losses.append(df['train_loss'])
            val_losses.append(df['val_loss'])
        
        if not train_losses or not val_losses:
            print("No valid training/validation losses to plot after cleaning.")
            return
        
        # Align on epochs
        train_loss_df = pd.concat(train_losses, axis=1)
        val_loss_df = pd.concat(val_losses, axis=1)
        
        mean_train_loss = train_loss_df.mean(axis=1, skipna=True)
        std_train_loss = train_loss_df.std(axis=1, skipna=True)
        
        mean_val_loss = val_loss_df.mean(axis=1, skipna=True)
        std_val_loss = val_loss_df.std(axis=1, skipna=True)
        
        plt.style.use("ggplot")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        
        # Plot training mean and std
        ax[0].plot(mean_train_loss.index, mean_train_loss.values, color='red', label='Training Mean')
        ax[0].fill_between(mean_train_loss.index,
                           mean_train_loss.values - std_train_loss.values,
                           mean_train_loss.values + std_train_loss.values,
                           color='red', alpha=0.3, label='Training Std Dev')
        ax[0].set_ylabel(f"{conf.model.objective_function} Loss", fontsize=10)
        ax[0].set_xlabel("Epoch", fontsize=10)
        ax[0].set_title("Training Loss", fontsize=12)
        ax[0].set_ylim([min_list[0], max_list[0]])
        ax[0].legend()
        
        # Plot validation mean and std
        ax[1].plot(mean_val_loss.index, mean_val_loss.values, color='blue', label='Validation Mean')
        ax[1].fill_between(mean_val_loss.index,
                           mean_val_loss.values - std_val_loss.values,
                           mean_val_loss.values + std_val_loss.values,
                           color='blue', alpha=0.3, label='Validation Std Dev')
        ax[1].set_ylabel(f"{conf.model.objective_function} Loss", fontsize=10)
        ax[1].set_xlabel("Epoch", fontsize=10)
        ax[1].set_title("Validation Loss", fontsize=12)
        ax[1].set_ylim([min_list[1], max_list[1]])
        ax[1].legend()
        
        fig.suptitle(model_identifier)
        fig.tight_layout()
        
        if save_fig:
            save_eval_item(save_dir, fig, 'loss.pdf', 'loss')
        else:
            plt.show()
    
    else:
        # Single run scenario
        loss_file = os.path.join(save_dir, "loss.csv")
        if not os.path.exists(loss_file) or os.path.getsize(loss_file) == 0:
            print("No loss.csv found or it's empty.")
            return
        
        df = pd.read_csv(loss_file)
        df = clean_loss_df(df)  # Clean the df to properly align train and val loss per epoch
        
        if 'train_loss' not in df.columns or 'val_loss' not in df.columns:
            print("train_loss or val_loss columns not found in cleaned DataFrame.")
            return
        
        plt.style.use("ggplot")
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

        ax[0].plot(df.index, df['train_loss'].values, color='red', label='Training Loss')
        ax[0].set_ylabel(f"{conf.model.objective_function} Loss", fontsize=10)
        ax[0].set_xlabel("Epoch", fontsize=10)
        ax[0].set_title("Training Loss", fontsize=12)
        ax[0].set_ylim([min_list[0], max_list[0]])
        ax[0].legend()

        ax[1].plot(df.index, df['val_loss'].values, color='blue', label='Validation Loss')
        ax[1].set_ylabel(f"{conf.model.objective_function} Loss", fontsize=10)
        ax[1].set_xlabel("Epoch", fontsize=10)
        ax[1].set_title("Validation Loss", fontsize=12)
        ax[1].set_ylim([min_list[1], max_list[1]])
        ax[1].legend()

        fig.suptitle(model_identifier)
        fig.tight_layout()

        if save_fig:
            save_eval_item(save_dir, fig, 'loss.pdf', 'loss')
        else:
            plt.show()

def calculate_metrics(truth, pred):
    """
    Calculate various metrics between ground truth and predictions.
    Also compute MSE at each slice if it's a 5D shape (N, T, R, X, Y).
    """      
    truth_torch = torch.tensor(truth) if not isinstance(truth, torch.Tensor) else truth
    pred_torch  = torch.tensor(pred)  if not isinstance(pred, torch.Tensor)  else pred
    # compute metrics
    mae = np.mean(np.abs(truth - pred))
    rmse = np.sqrt(np.mean((truth - pred) ** 2))
    correlation = np.corrcoef(truth.flatten(), pred.flatten())[0, 1]

    psnr = PeakSignalNoiseRatio(data_range=1.0)(pred_torch, truth_torch)
    try:
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0)(pred_torch, truth_torch)
    except:
        ssim = torch.tensor(0.0)

    # Initialize placeholders
    rmse_first_slice = None
    rmse_final_slice = None
    final_slice_std  = None
    
    # If 5D shape: (N, T, R, X, Y)
    if truth.ndim == 5:   
        # Compute final slice MSE across the batch
        final_slice_errors = (truth[:, -1] - pred[:, -1])**2  
        final_slice_mse_per_sample = np.mean(final_slice_errors, axis=(1,2,3))  
        rmse_final_slice = np.sqrt(np.mean(final_slice_mse_per_sample))

        # standard deviation among the batch for the final slice MSE
        final_slice_std = np.sqrt(np.var(final_slice_mse_per_sample))
        
        # same for first slice
        first_slice_errors = (truth[:, 0] - pred[:, 0])**2  
        first_slice_mse_per_sample = np.mean(first_slice_errors, axis=(1,2,3))  
        rmse_first_slice = np.sqrt(np.mean(first_slice_mse_per_sample))
        first_slice_std = np.sqrt(np.var(first_slice_mse_per_sample))

    # Build dictionary
    out = {
        'MAE': mae,
        'RMSE': rmse,
        'Correlation': correlation,
        'PSNR': psnr.item(),
        'SSIM': ssim.item()
    }
    if rmse_first_slice is not None:
        out['RMSE_First_Slice'] = f"{rmse_first_slice:.4f} +/- {first_slice_std:.4f}"
    if rmse_final_slice is not None:
        # Add a string with +/- if you like
        out['RMSE_Final_Slice'] = f"{rmse_final_slice:.4f} +/- {final_slice_std:.4f}"

    return out

def metrics(test_results, fold_idx=None, dataset='valid', 
                  save_fig=False, save_dir=None, plot_mse=False):
    """Print metrics for a specific fold and dataset (train or valid)."""
    if dataset not in test_results:
        raise ValueError(f"Dataset '{dataset}' not found in test_results.")

    truth = test_results[dataset]['nf_truth']
    pred = test_results[dataset]['nf_pred']
    
    std_metrics = ["RMSE_First_Slice", "RMSE_Final_Slice"]

    metrics = calculate_metrics(truth, pred)
    print(f"Metrics for {dataset.capitalize()} Dataset:")
    for metric, value in metrics.items():
        if metric in std_metrics:
            print(f"{metric}: {value}")
        else:
            print(f"{metric}: {value:.4f}")

    # save to file if requested
    if save_fig:
        if not save_dir:
            raise ValueError("Please specify a save directory")
        file_name = f'{dataset}_metrics.txt'
        save_eval_item(save_dir, metrics, file_name, 'metrics')
        
        # 3) Optionally compute MSE vs. time slice and plot
    if plot_mse:
        # compute MSE(t) for each slice
        mse_means, mse_stds = compute_mse_per_slice(truth, pred)

        # Plot
        plot_mse_evolution(
            mse_means, 
            mse_stds, 
            title=f"{dataset.capitalize()} - MSE Across Slices",
            save_fig=save_fig,
            save_dir=save_dir
        )

def compute_mse_per_slice(truth, pred):
    """
    Compute the mean and std-dev of the MSE across the batch dimension
    for each timestep in [0..seq_len-1].
    
    Args:
        truth: np.ndarray or torch.Tensor of shape (batch, seq_len, r_i, xdim, ydim)
        pred:  np.ndarray or torch.Tensor of the same shape
    
    Returns:
        mse_means: np.ndarray of shape (seq_len,) - average MSE at each time t
        mse_stds:  np.ndarray of shape (seq_len,) - std dev of MSE across the batch at each time t
    """
    # Ensure we have numpy arrays
    if isinstance(truth, torch.Tensor):
        truth = truth.detach().cpu().numpy()
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    
    # If shape is (batch, seq_len, r_i, xdim, ydim), let's extract dimensions
    batch_size, seq_len = truth.shape[0], truth.shape[1]
    
    # Arrays to store results
    mse_means = np.zeros(seq_len)
    mse_stds  = np.zeros(seq_len)
    
    # For each timestep t, compute MSE for each sample in the batch
    for t in range(seq_len):
        # shape of errors: (batch_size, r_i, xdim, ydim)
        errors_t = (truth[:, t] - pred[:, t])**2  # still shape (N, r_i, xdim, ydim)
        
        # MSE per sample = average across spatial dimensions
        # shape: (batch_size,)
        mse_per_sample = np.mean(errors_t, axis=(1, 2, 3))  
        
        # Now compute mean, std across batch dimension
        mse_means[t] = np.mean(mse_per_sample)  
        mse_stds[t]  = np.std(mse_per_sample)
    
    return mse_means, mse_stds

def plot_mse_evolution(mse_means, mse_stds=None, title="MSE Across Slices", save_fig=None, save_dir=None):
    """
    Plot the MSE vs timestep (with optional std-dev shading or error bars).
    
    Args:
        mse_means: np.ndarray of shape (T,) - the mean MSE at each timestep
        mse_stds:  (optional) np.ndarray of shape (T,) - the std dev of MSE
                   if None, we won't plot error shading.
        title:     title of the plot
        save_fig: if True, saves the plot to a file
        save_dir: if provided, saves the plot to this path
    """
    timesteps = np.arange(len(mse_means))

    fig = plt.figure(figsize=(8, 5))
    # Plot the mean
    ax = fig.add_subplot(111)
    ax.plot(timesteps, mse_means, label="Mean MSE", color='blue')
    
    # Optionally add error shading or error bars
    if mse_stds is not None:
        # Shaded region
        plt.fill_between(
            timesteps,
            mse_means - mse_stds,
            mse_means + mse_stds,
            color='blue',
            alpha=0.2,
            label="Std Dev"
        )
        # Or you could do error bars instead:
        # plt.errorbar(timesteps, mse_means, yerr=mse_stds, fmt='o-', ecolor='lightblue')
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    if save_fig:
        if not save_dir:
            raise ValueError("Please specify a save directory")
        file_name = f'mse_evolution.pdf'
        save_eval_item(save_dir, fig, file_name, 'evo')
    else:
        plt.show()

def plot_dft_fields(test_results, resub=False,
                    sample_idx=0, save_fig=False, save_dir=None,
                    arch='mlp', format='polar', fold_num=False):
    """
    Parameters:
    - test_results: List of dictionaries containing train and valid results
    - sample_idx: Index of the sample to plot
    - save_fig: Whether to save the plot to a file
    - save_dir: Directory to save the plot to
    - arch: "mlp" or "lstm"
    - format: "cartesian" or "polar"
    - fold_num: the fold # of the selected fold being plotted (if cross val)
    """
    def plot_single_set(results, title, format, save_path, sample_idx):
        if arch == 'mlp' or arch == 'cvnn' or arch == 'autoencoder':
            # extract and convert to tensors
            truth_real = torch.from_numpy(results['nf_truth'][sample_idx, 0, :, :])
            truth_imag = torch.from_numpy(results['nf_truth'][sample_idx, 1, :, :])
            pred_real = torch.from_numpy(results['nf_pred'][sample_idx, 0, :, :])
            pred_imag = torch.from_numpy(results['nf_pred'][sample_idx, 1, :, :])

            # determine which coordinate format to plot
            if format == 'polar':
                component_1 = "Magnitude"
                component_2 = "Phase"
                # convert to magnitude and phase
                truth_component_1, truth_component_2 = mapping.cartesian_to_polar(truth_real, truth_imag)
                pred_component_1, pred_component_2 = mapping.cartesian_to_polar(pred_real, pred_imag)
            else:
                component_1 = "Real"
                component_2 = "Imaginary"
                truth_component_1, truth_component_2 = truth_real, truth_imag
                pred_component_1, pred_component_2 = pred_real, pred_imag

            # 4 subplots (2x2 grid)
            fig, ax = plt.subplots(2, 2, figsize=(8, 8))
            fig.suptitle(title, fontsize=16)
            fig.text(0.5, 0.92, model_identifier, ha='center', fontsize=12)

            # real part of the truth
            ax[0, 0].imshow(truth_component_1, cmap='viridis')
            ax[0, 0].set_title(f'True {component_1} Component')
            ax[0, 0].axis('off')

            # real part of the prediction
            ax[0, 1].imshow(pred_component_1, cmap='viridis')
            ax[0, 1].set_title(f'Predicted {component_1} Component')
            ax[0, 1].axis('off')

            # imaginary part of the truth
            ax[1, 0].imshow(truth_component_2, cmap='twilight_shifted')
            ax[1, 0].set_title(f'True {component_2} Component')
            ax[1, 0].axis('off')

            # imaginary part of the prediction
            ax[1, 1].imshow(pred_component_2, cmap='twilight_shifted')  
            ax[1, 1].set_title(f'Predicted {component_2} Component')
            ax[1, 1].axis('off')
                
        else:
            # extract and convert to tensors
            truth_real = torch.from_numpy(results['nf_truth'][sample_idx, :, 0, :, :])
            truth_imag = torch.from_numpy(results['nf_truth'][sample_idx, :, 1, :, :])
            pred_real = torch.from_numpy(results['nf_pred'][sample_idx, :, 0, :, :])
            pred_imag = torch.from_numpy(results['nf_pred'][sample_idx, :, 1, :, :])
            
            
            # determine which coordinate format to plot
            if format == 'polar':
                component_1 = "Magnitude"
                component_2 = "Phase"
                # convert to magnitude and phase
                truth_component_1, truth_component_2 = mapping.cartesian_to_polar(truth_real, truth_imag)
                pred_component_1, pred_component_2 = mapping.cartesian_to_polar(pred_real, pred_imag)
            else:
                component_1 = "Real"
                component_2 = "Imaginary"
                truth_component_1, truth_component_2 = truth_real, truth_imag
                pred_component_1, pred_component_2 = pred_real, pred_imag
            
            seq_len = truth_component_1.shape[0]
            
            # Create figure WITHOUT creating subplots
            fig = plt.figure(figsize=(4*seq_len + 2, 16))
            
            # Create gridspec with space for labels and column headers
            gs = fig.add_gridspec(5, seq_len + 1,  # 5 rows: header + 4 data rows
                                width_ratios=[0.3] + [1]*seq_len,
                                height_ratios=[0.05] + [1]*4,
                                hspace=0.1,
                                wspace=0.1)
            
            # Create axes for column headers
            header_axs = [fig.add_subplot(gs[0, j]) for j in range(1, seq_len + 1)]
            
            # Create axes for images
            axs = [[fig.add_subplot(gs[i+1, j]) for j in range(1, seq_len + 1)] 
                for i in range(4)]
            
            # Create axes for row labels
            label_axs = [fig.add_subplot(gs[i+1, 0]) for i in range(4)]
            
            fig.suptitle(title, fontsize=24, y=0.95, fontweight='bold')
            fig.text(0.5, 0.94, model_identifier, ha='center', fontsize=16)
            
            # Add column headers
            for t, ax in enumerate(header_axs):
                ax.axis('off')
                ax.text(0.5, 0.3,
                    f't={t+1}',
                    ha='center',
                    va='center',
                    fontsize=20,
                    fontweight='bold')
            
            # Add row labels
            row_labels = [f'Ground Truth\n{component_1}',
                        f'Predicted\n{component_1}',
                        f'Ground Truth\n{component_2}',
                        f'Predicted\n{component_2}']
            
            for ax, label in zip(label_axs, row_labels):
                ax.axis('off')
                ax.text(0.95, 0.5, 
                    label,
                    ha='right',
                    va='center',
                    fontsize=20,
                    fontweight='bold')
            
            # Plot sequence
            for t in range(seq_len):
                axs[0][t].imshow(truth_component_1[t], cmap='viridis')
                axs[0][t].axis('off')
                
                axs[1][t].imshow(pred_component_1[t], cmap='viridis')
                axs[1][t].axis('off')
                
                axs[2][t].imshow(truth_component_2[t], cmap='twilight_shifted')
                axs[2][t].axis('off')
                
                axs[3][t].imshow(pred_component_2[t], cmap='twilight_shifted')
                axs[3][t].axis('off')
                
        #fig.tight_layout()
        
        # save the plot if specified
        if save_fig:
            if not save_dir:
                raise ValueError("Please specify a save directory")
            file_name = f'{title}_dft_sample_idx_{sample_idx}_{format}.pdf'
            save_eval_item(save_dir, fig, file_name, 'dft')

        plt.show()

    if fold_num:
        title = f'Cross Val - Fold {fold_num}'
    else:
        title = 'Default Split'
    # Plot both training and validation results
    if resub:
        plot_single_set(test_results['train'], f"{title} - Random Training Sample - {format}", format, save_dir, sample_idx)
    plot_single_set(test_results['valid'], f"{title} - Random Validation Sample - {format}", format, save_dir, sample_idx)
    
def plot_absolute_difference(test_results, resub=False, sample_idx=0, 
                             save_fig=False, save_dir=None, arch='mlp', fold_num=None):
    """
    Plot a sequence of absolute difference between predicted and ground truth fields
    
    Args:
        test_results (list): List of dictionaries containing train and valid results
        resub (bool): Whether to plot a random training sample instead of a random validation sample
        sample_idx (int): Index of sample to plot
        save_fig (bool): Whether to save the figure
        save_dir (str): Directory to save figure if save_fig is True
        arch (str): Architecture type ('mlp' or 'lstm')
        fold_num: the fold # of the selected fold being plotted (if cross val)

    """
    def plot_single_set(results, title, sample_idx):
        abs_diff = calculate_absolute_difference(results, sample_idx)
        
        if arch == 'mlp' or arch == 'cvnn' or arch == 'autoencoder':
            # Extract real and imaginary differences
            real_diff = abs_diff[0, :, :]
            imag_diff = abs_diff[1, :, :]
            
            # Convert to magnitude and phase differences
            #mag_diff, phase_diff = mapping.cartesian_to_polar(real_diff, imag_diff)
            
            # Create a single column plot
            fig, ax = plt.subplots(2, 1, figsize=(6, 12))
            fig.suptitle(title, fontsize=16)
            
            # Plot magnitude difference
            im_mag = ax[0].imshow(real_diff, cmap='magma')
            ax[0].set_title('Real Difference')
            ax[0].axis('off')
            
            # Plot phase difference
            im_phase = ax[1].imshow(imag_diff, cmap='magma')
            ax[1].set_title('Imaginary Difference')
            ax[1].axis('off')
            
            # Add colorbars
            fig.colorbar(im_mag, ax=ax[0], orientation='vertical', fraction=0.046, pad=0.04)
            fig.colorbar(im_phase, ax=ax[1], orientation='vertical', fraction=0.046, pad=0.04)
        
        else:
            # Extract real and imaginary differences
            real_diff = abs_diff[:, 0, :, :]
            imag_diff = abs_diff[:, 1, :, :]
            
            # Convert to magnitude and phase differences
            mag_diff, phase_diff = mapping.cartesian_to_polar(real_diff, imag_diff)
            
            seq_len = mag_diff.shape[0]
            
            # Create figure with space for colorbar
            fig = plt.figure(figsize=(4*seq_len, 9))
            gs = fig.add_gridspec(3, seq_len, height_ratios=[1, 1, 0.1])
            
            axs_top = [fig.add_subplot(gs[0, i]) for i in range(seq_len)]
            axs_bottom = [fig.add_subplot(gs[1, i]) for i in range(seq_len)]
            cax = fig.add_subplot(gs[2, :])
            
            fig.suptitle(title, fontsize=16)
            
            for t in range(seq_len):
                im_mag = axs_top[t].imshow(mag_diff[t], cmap='magma')
                axs_top[t].axis('off')
                axs_top[t].set_title(f't={t+1}')
                
                im_phase = axs_bottom[t].imshow(phase_diff[t], cmap='magma')
                axs_bottom[t].axis('off')
            
            # Add single colorbar at the bottom
            cbar = plt.colorbar(im_mag, cax=cax, orientation='horizontal')
            cbar.set_label('Absolute Difference')
        
        if save_fig:
            if not save_dir:
                raise ValueError("Please specify a save directory")
            file_name = f'abs_diff_{title}.pdf'
            save_eval_item(save_dir, fig, file_name, 'dft')
        else:
            plt.show()

    if fold_num:
        title = f'Cross Val - Fold {fold_num}'
    else:
        title = 'Default Split'
    plot_single_set(test_results['valid'], f"{title} - Validation", sample_idx)
    if resub:
        plot_single_set(test_results['train'], f"{title} - Training", sample_idx)

def calculate_absolute_difference(results, sample_idx=0):
    """Generate absolute difference data for a given sample"""
    truth = torch.from_numpy(results['nf_truth'][sample_idx, :])
    pred = torch.from_numpy(results['nf_pred'][sample_idx, :])
    return torch.abs(truth - pred)
    
def animate_fields(test_results, dataset, sample_idx=0, seq_len=5, save_dir=None): 
    results = test_results[dataset]
    truth_real = torch.from_numpy(results['nf_truth'][sample_idx, :, 0, :, :])
    truth_imag = torch.from_numpy(results['nf_truth'][sample_idx, :, 1, :, :])
    pred_real = torch.from_numpy(results['nf_pred'][sample_idx, :, 0, :, :])
    pred_imag = torch.from_numpy(results['nf_pred'][sample_idx, :, 1, :, :])
    truth_real = truth_real.permute(1, 2, 0)
    truth_imag = truth_imag.permute(1, 2, 0)
    pred_real = pred_real.permute(1, 2, 0)
    pred_imag = pred_imag.permute(1, 2, 0)

    truth_mag, truth_phase = mapping.cartesian_to_polar(truth_real, truth_imag)
    pred_mag, pred_phase = mapping.cartesian_to_polar(pred_real, pred_imag)

    flipbooks_dir = os.path.join(save_dir, "flipbooks")
    os.makedirs(flipbooks_dir, exist_ok=True)

    # intensity
    truth_anim = viz.animate_fields(truth_mag, "True Intensity", 
                                save_path=os.path.join(flipbooks_dir, f"{dataset}_sample_{sample_idx}_mag_groundtruth.gif"), 
                                frames=seq_len,
                                interval=250)
    pred_anim = viz.animate_fields(pred_mag, "Predicted Intensity", 
                                save_path=os.path.join(flipbooks_dir, f"{dataset}_sample_{sample_idx}_mag_prediction.gif"), 
                                frames=seq_len,
                                interval=250)

    # phase
    truth_phase_anim = viz.animate_fields(truth_phase, "True Phase", 
                                save_path=os.path.join(flipbooks_dir, f"{dataset}_sample_{sample_idx}_phase_groundtruth.gif"), 
                                cmap='twilight_shifted',
                                frames=seq_len,
                                interval=250)
    pred_phase_anim = viz.animate_fields(pred_phase, "Predicted Phase", 
                                save_path=os.path.join(flipbooks_dir, f"{dataset}_sample_{sample_idx}_phase_prediction.gif"), 
                                cmap='twilight_shifted',
                                frames=seq_len,
                                interval=250)

def construct_results_table(model_names, model_types):
    # Define the metrics to extract
    metrics_to_extract = ["RMSE", "Correlation", "PSNR"]
    
    # Initialize a dictionary to store results
    results = {model_type: {model_name: {"resub": {}, "testing": {}} for model_name in model_names} for model_type in model_types}
    
    # Base path for metrics files
    base_path = "/develop/results/meep_meep"
    
    # Iterate over each model type and model name
    for model_type in model_types:
        for model_name in model_names:
            # Construct paths for train and valid metrics files
            train_metrics_path = os.path.join(base_path, model_type, f"model_{model_name}", "performance_metrics", "train_metrics.txt")
            valid_metrics_path = os.path.join(base_path, model_type, f"model_{model_name}", "performance_metrics", "valid_metrics.txt")            
            # Read and parse the train metrics file
            with open(train_metrics_path, 'r') as file:
                for line in file:
                    for metric in metrics_to_extract:
                        if line.startswith(metric):
                            value = line.split(":")[1].strip().split("±")[0].strip()
                            results[model_type][model_name]["resub"][metric] = value
            
            # Read and parse the valid metrics file
            with open(valid_metrics_path, 'r') as file:
                for line in file:
                    for metric in metrics_to_extract:
                        if line.startswith(metric):
                            value = line.split(":")[1].strip().split("±")[0].strip()
                            results[model_type][model_name]["testing"][metric] = value
    
    # Print the results table to the command line
    print("Results Table:")
    for model_type in model_types:
        print(f"\nModel Type: {model_type}")
        print(f"{'Model Name':<20} {'Metric':<15} {'Resub':<10} {'Testing':<10}")
        print("-" * 60)
        for model_name in model_names:
            for metric in metrics_to_extract:
                resub_value = results[model_type][model_name]["resub"].get(metric, "N/A")
                testing_value = results[model_type][model_name]["testing"].get(metric, "N/A")
                print(f"{model_name:<20} {metric:<15} {resub_value:<10} {testing_value:<10}")
    
    # Generate LaTeX-friendly table
    latex_table = "\\begin{table}[h!]\n\\centering\n\\caption{Model Performance Metrics}\n\\begin{tabular}{|l|l|l|l|l|}\n\\hline\n"
    latex_table += "Model Type & Model Name & Metric & Resub & Testing \\\\\n\\hline\n"
    for model_type in model_types:
        for model_name in model_names:
            # Escape underscores in model names for LaTeX compatibility
            latex_model_name = model_name.replace("_", "\\_")
            for metric in metrics_to_extract:
                resub_value = results[model_type][model_name]["resub"].get(metric, "N/A")
                testing_value = results[model_type][model_name]["testing"].get(metric, "N/A")
                latex_table += f"{model_type} & {latex_model_name} & {metric} & {resub_value} & {testing_value} \\\\\n"
    latex_table += "\\hline\n\\end{tabular}\n\\end{table}"
    
    print("\nLaTeX Table:")
    print(latex_table)
    
if __name__ == "__main__":
    # fetch the model names from command line args
    import argparse
    parser = argparse.ArgumentParser(description="Construct a results table for a given set of models")
    parser.add_argument("--model_names", nargs="+", required=True, help="List of model names to include in the table")
    parser.add_argument("--model_types", nargs="+", required=True, help="List of model types to include in the table")
    args = parser.parse_args()
    
    construct_results_table(args.model_names, args.model_types)