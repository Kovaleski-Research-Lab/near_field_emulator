#!/bin/bash

# Array of config files to run
configs=(
    "config_comp_ssim.yaml"
    "config_comp_mse.yaml"
    "config_large_ssim.yaml"
    "config_large_mse.yaml"
)

# Base path for the configs
config_dir="/develop/code/near_field_emulator/conf"
main_script="/develop/code/near_field_emulator/main.py"

# Create or clear the timing log file
echo "config,start_time,end_time,duration_seconds" > training_times.csv

# Loop through each config file
for config in "${configs[@]}"; do
    echo "=========================================="
    echo "Starting training with config: $config"
    echo "=========================================="
    
    # Record start time
    start_time=$(date +%s)
    start_time_readable=$(date -d @$start_time)
    echo "Start time: $start_time_readable"
    
    # Run the training command
    python $main_script --config "$config_dir/$config"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        # Record end time and calculate duration
        end_time=$(date +%s)
        end_time_readable=$(date -d @$end_time)
        duration=$((end_time - start_time))
        
        # Convert duration to hours, minutes, seconds
        hours=$((duration / 3600))
        minutes=$(( (duration % 3600) / 60 ))
        seconds=$((duration % 60))
        
        echo "Successfully completed training with $config"
        echo "End time: $end_time_readable"
        echo "Duration: ${hours}h ${minutes}m ${seconds}s"
        
        # Append to CSV file
        echo "$config,$start_time_readable,$end_time_readable,$duration" >> training_times.csv
    else
        echo "Error occurred while running $config"
        exit 1  # Exit if there's an error
    fi
    
    echo "=========================================="
    echo "Finished training with config: $config"
    echo "=========================================="
    
    # Optional: Add a small delay between runs
    sleep 5
done

echo "All configurations have been processed"
echo "Timing information saved to training_times.csv"