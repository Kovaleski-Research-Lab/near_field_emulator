import matplotlib
matplotlib.use('Agg')  # or try 'Qt5Agg' if TkAgg doesn't work
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os

def analyze_tensorboard_logs(log_dir):
    """Load and analyze TensorBoard logs"""
    print(f"Analyzing logs in: {log_dir}")
    
    # Check if directory exists
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
        
    # List contents of directory
    print("Contents of log directory:")
    print(os.listdir(log_dir))
    
    # Initialize event accumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available tags (metrics)
    tags = event_acc.Tags()['scalars']
    print("\nAvailable metrics:", tags)
    
    # Create DataFrame with all metrics
    all_steps = set()
    metrics = {}
    
    # First pass: collect all steps
    for tag in tags:
        events = event_acc.Scalars(tag)
        steps = [event.step for event in events]
        all_steps.update(steps)
    
    # Convert to sorted list
    all_steps = sorted(list(all_steps))
    print(f"\nTotal number of steps: {len(all_steps)}")
    
    # Second pass: create dictionary with NaN for missing steps
    for tag in tags:
        events = event_acc.Scalars(tag)
        # Create dict of step -> value
        step_to_value = {event.step: event.value for event in events}
        
        # Fill in values for all steps, using NaN for missing steps
        metrics[tag] = [step_to_value.get(step, float('nan')) for step in all_steps]
        print(f"Loaded {len(events)} values for {tag}")
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics, index=all_steps)
    
    # If epoch is available in metrics, use it to create a proper index
    if 'epoch' in df.columns:
        # Create epoch index
        df['epoch'] = df['epoch'].fillna(method='ffill')
        df = df.set_index('epoch')
    
    # Optionally, forward fill NaN values
    df = df.fillna(method='ffill')
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs')
    args = parser.parse_args()
    
    print("Starting analysis...")
    
    try:
        metrics_df = analyze_tensorboard_logs(args.log_dir)
        print("\nDataFrame shape:", metrics_df.shape)
        print("\nDataFrame columns:", metrics_df.columns.tolist())
        
        # Create figure
        plt.figure(figsize=(15, 5))
        print("\nCreated figure")

        # Loss plot
        plt.subplot(131)
        if 'train_loss' in metrics_df.columns and 'val_loss' in metrics_df.columns:
            plt.plot(metrics_df.index, metrics_df['train_loss'], label='Train')
            plt.plot(metrics_df.index, metrics_df['val_loss'], label='Validation')
            plt.title('MSE Loss over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.legend()
        print("Created loss plot")

        # PSNR plot
        plt.subplot(132)
        if 'train_psnr' in metrics_df.columns and 'val_psnr' in metrics_df.columns:
            plt.plot(metrics_df.index, metrics_df['train_psnr'], label='Train')
            plt.plot(metrics_df.index, metrics_df['val_psnr'], label='Validation')
            plt.title('PSNR over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('PSNR')
            plt.legend()
        print("Created PSNR plot")

        # SSIM plot
        plt.subplot(133)
        if 'train_ssim' in metrics_df.columns and 'val_ssim' in metrics_df.columns:
            plt.plot(metrics_df.index, metrics_df['train_ssim'], label='Train')
            plt.plot(metrics_df.index, metrics_df['val_ssim'], label='Validation')
            plt.title('SSIM over epochs')
            plt.xlabel('Epoch')
            plt.ylabel('SSIM')
            plt.legend()
        print("Created SSIM plot")

        plt.tight_layout()
        
        # Save the plot
        save_path = 'training_metrics.png'
        plt.savefig(os.path.join(args.log_dir, save_path))
        print(f"\nSaved plot to {save_path}")

        # Print summary statistics
        print("\nFinal Metrics:")
        for column in metrics_df.columns:
            if column != 'epoch':  # Skip epoch column in statistics
                print(f"{column}:")
                print(f"  Best: {metrics_df[column].max():.4f}")
                print(f"  Final: {metrics_df[column].iloc[-1]:.4f}")
                print(f"  Mean: {metrics_df[column].mean():.4f}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
