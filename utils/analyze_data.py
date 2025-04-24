def analyze_data_split(data_path: str, save_dir: str = None) -> None:
    """
    Analyze and visualize the data split in a .pt file.
    
    Parameters
    ----------
    data_path: str
        Path to the .pt file containing the data dictionary
    save_dir: str, optional
        Directory to save the plots. If None, plots will be saved in the current directory.
    """
    # Load the data
    data = torch.load(data_path)
    
    # Get the tag tensor and calculate split statistics
    tags = data['tag']
    total_samples = len(tags)
    train_samples = (tags == 1).sum().item()
    val_samples = (tags == 0).sum().item()
    
    # Calculate percentages
    train_percent = (train_samples / total_samples) * 100
    val_percent = (val_samples / total_samples) * 100
    
    # Print statistics
    print("\nData Split Analysis")
    print("=" * 50)
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_samples} ({train_percent:.1f}%)")
    print(f"Validation samples: {val_samples} ({val_percent:.1f}%)")
    
    # Analyze near fields
    near_fields = data['near_fields']
    print("\nNear Fields Statistics")
    print("=" * 50)
    print(f"Shape: {near_fields.shape}")
    print(f"Training set mean: {near_fields[tags == 1].mean():.4f}")
    print(f"Validation set mean: {near_fields[tags == 0].mean():.4f}")
    
    # Analyze refidx
    refidx = data['refidx']
    print("\nRefractive Index Statistics")
    print("=" * 50)
    print(f"Shape: {refidx.shape}")
    print(f"Training set mean: {refidx[tags == 1].mean():.4f}")
    print(f"Validation set mean: {refidx[tags == 0].mean():.4f}")
    
    # Create a pie chart of the split
    plt.figure(figsize=(10, 5))
    
    # Pie chart
    plt.subplot(121)
    plt.pie([train_samples, val_samples], 
            labels=['Training', 'Validation'],
            autopct='%1.1f%%',
            colors=['#2ecc71', '#e74c3c'])
    plt.title('Data Split Distribution')
    
    # Bar plot of means
    plt.subplot(122)
    means = [near_fields[tags == 1].mean().item(), 
             near_fields[tags == 0].mean().item(),
             refidx[tags == 1].mean().item(),
             refidx[tags == 0].mean().item()]
    labels = ['Near Fields\n(Train)', 'Near Fields\n(Val)',
              'Refidx\n(Train)', 'Refidx\n(Val)']
    plt.bar(labels, means, color=['#2ecc71', '#e74c3c', '#2ecc71', '#e74c3c'])
    plt.title('Mean Values by Split')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    if save_dir is None:
        save_dir = '.'
    plt.savefig(f'{save_dir}/data_split_analysis.png')
    plt.close()
    
    # Print a few sample values
    print("\nSample Values")
    print("=" * 50)
    print("First 5 training samples:")
    for i in range(5):
        if tags[i] == 1:
            print(f"Sample {i}:")
            print(f"  Near Fields: {near_fields[i, 0, 0, 0, 0]:.4f} (first value)")
            print(f"  Refidx: {refidx[i, 0]:.4f}")
    
    print("\nFirst 5 validation samples:")
    val_count = 0
    i = 0
    while val_count < 5 and i < len(tags):
        if tags[i] == 0:
            print(f"Sample {i}:")
            print(f"  Near Fields: {near_fields[i, 0, 0, 0, 0]:.4f} (first value)")
            print(f"  Refidx: {refidx[i, 0]:.4f}")
            val_count += 1
        i += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze data split in a .pt file')
    parser.add_argument('data_path', type=str, help='Path to the .pt file')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save plots')
    args = parser.parse_args()
    
    analyze_data_split(args.data_path, args.save_dir) 