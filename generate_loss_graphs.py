#!/usr/bin/env python3
"""
Generate Loss Curves for Experiments

This script generates static images or animated GIFs of training and validation loss curves
for completed experiments.

Usage:
  python generate_loss_graphs.py                # Process all completed experiments
  python generate_loss_graphs.py --recent N     # Process only the N most recent experiments
  python generate_loss_graphs.py --experiment exp_id_123  # Process a specific experiment
  python generate_loss_graphs.py --all          # Process all experiments (default)
  python generate_loss_graphs.py --animated     # Generate animated GIFs instead of static images
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from models.common.utils import safe_json_load

# Set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# File paths
COMPLETED_EXPERIMENTS_FILE = 'data/completed_experiments.json'
OUTPUT_DIR = 'data/loss_graphs'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate loss curves for experiments')
    
    # Filter options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--recent', type=int, help='Process only N most recent experiments')
    group.add_argument('--experiment', type=str, help='Process a specific experiment ID or UID')
    group.add_argument('--all', action='store_true', help='Process all experiments (default)')
    
    # Output options
    parser.add_argument('--animated', action='store_true', help='Generate animated GIFs instead of static images')
    
    return parser.parse_args()

def plot_static_loss_curves(experiment, output_path):
    """
    Generate a static plot of training and validation loss curves.
    
    Args:
        experiment: Experiment data dictionary
        output_path: Path to save the output image
        
    Returns:
        bool: True if image was created successfully
    """
    # Check if metrics are available
    if 'metrics' not in experiment or 'train_loss' not in experiment['metrics'] or 'val_loss' not in experiment['metrics']:
        print(f"No loss data found for experiment {experiment.get('experiment_id', 'unknown')}")
        return False
    
    try:
        # Extract loss data
        train_losses = experiment['metrics']['train_loss']
        val_losses = experiment['metrics']['val_loss']
        epochs = range(1, len(train_losses) + 1)
        
        # Get experiment details for title
        exp_id = experiment.get('experiment_id', 'unknown')
        
        # Get hyperparameters for subtitle
        hyperparams = experiment.get('hyperparams', {})
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        num_encoder_layers = hyperparams.get('num_encoder_layers', 'N/A')
        dim_feedforward = hyperparams.get('dim_feedforward', 'N/A')
        learning_rate = hyperparams.get('learning_rate', 'N/A')
        
        # Extract final validation accuracy if available
        val_accuracy = experiment['metrics'].get('val_accuracy', [])
        if val_accuracy:
            accuracy_info = f"Final Accuracy: {val_accuracy[-1]:.4f}"
        else:
            accuracy_info = ""
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the curves
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        
        # Add labels and title
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves for {exp_id}\n' +
                 f'd_model={d_model}, nhead={nhead}, layers={num_encoder_layers}, ff={dim_feedforward}, lr={learning_rate}\n' +
                 accuracy_info)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        print(f"Created loss graph: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating loss graph for {experiment.get('experiment_id', 'unknown')}: {e}")
        return False

def create_animated_loss_gif(experiment, output_path):
    """
    Create an animated GIF showing the progression of loss curves.
    
    Args:
        experiment: Experiment data dictionary
        output_path: Path to save the output GIF
        
    Returns:
        bool: True if GIF was created successfully
    """
    # Check if metrics are available
    if 'metrics' not in experiment or 'train_loss' not in experiment['metrics'] or 'val_loss' not in experiment['metrics']:
        print(f"No loss data found for experiment {experiment.get('experiment_id', 'unknown')}")
        return False
    
    try:
        # Extract loss data
        train_losses = experiment['metrics']['train_loss']
        val_losses = experiment['metrics']['val_loss']
        total_epochs = len(train_losses)
        
        # Get experiment details for title
        exp_id = experiment.get('experiment_id', 'unknown')
        
        # Get hyperparameters for subtitle
        hyperparams = experiment.get('hyperparams', {})
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        num_encoder_layers = hyperparams.get('num_encoder_layers', 'N/A')
        dim_feedforward = hyperparams.get('dim_feedforward', 'N/A')
        learning_rate = hyperparams.get('learning_rate', 'N/A')
        
        # Temporary directory for frames
        temp_dir = os.path.join(OUTPUT_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Find y-axis limits to keep them consistent
        min_loss = min(min(train_losses), min(val_losses))
        max_loss = max(max(train_losses), max(val_losses))
        
        # Add some padding to the limits
        y_range = max_loss - min_loss
        y_min = max(0, min_loss - 0.05 * y_range)  # Don't go below 0
        y_max = max_loss + 0.05 * y_range
        
        # Create frames
        frame_files = []
        
        # Determine how many frames to create (animation should show ~10-20 steps)
        frame_step = max(1, total_epochs // 20)
        
        for frame_idx, epoch in enumerate(range(1, total_epochs + 1, frame_step)):
            # Show all data up to this epoch
            current_epochs = list(range(1, epoch + 1))
            current_train_losses = train_losses[:epoch]
            current_val_losses = val_losses[:epoch]
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Plot the curves
            plt.plot(current_epochs, current_train_losses, 'b-', label='Training Loss')
            plt.plot(current_epochs, current_val_losses, 'r-', label='Validation Loss')
            
            # Set consistent y-axis limits
            plt.ylim(y_min, y_max)
            
            # Add labels and title
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curves for {exp_id} - Epoch {epoch}/{total_epochs}\n' +
                    f'd_model={d_model}, nhead={nhead}, layers={num_encoder_layers}, ff={dim_feedforward}, lr={learning_rate}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Save the frame
            frame_file = os.path.join(temp_dir, f'frame_{frame_idx:03d}.png')
            plt.tight_layout()
            plt.savefig(frame_file, dpi=100)
            plt.close()
            
            frame_files.append(frame_file)
        
        # Create GIF
        images = [imageio.imread(file) for file in frame_files]
        imageio.mimsave(output_path, images, duration=0.3)
        
        # Clean up temporary files
        for file in frame_files:
            if os.path.exists(file):
                os.remove(file)
        
        print(f"Created animated loss GIF: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating loss animation for {experiment.get('experiment_id', 'unknown')}: {e}")
        return False

def generate_loss_graph(experiment, output_dir=OUTPUT_DIR, animated=False):
    """
    Generate a loss graph for an experiment.
    
    Args:
        experiment: Experiment data dictionary
        output_dir: Directory to save the output
        animated: Whether to create an animated GIF instead of a static image
        
    Returns:
        bool: True if graph was created, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if experiment has required data
    if 'metrics' not in experiment or 'uid' not in experiment:
        print(f"Insufficient data for experiment {experiment.get('experiment_id', 'unknown')}")
        return False
    
    # Set up output file path
    if animated:
        output_filename = f"{experiment['uid']}_loss_animated.gif"
    else:
        output_filename = f"{experiment['uid']}_loss.png"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Skip if file already exists
    if os.path.exists(output_path):
        print(f"Graph already exists for {experiment['uid']}")
        return False
    
    # Generate the appropriate type of graph
    if animated:
        return create_animated_loss_gif(experiment, output_path)
    else:
        return plot_static_loss_curves(experiment, output_path)

def process_experiments(args):
    """Process experiments based on command line arguments."""
    # Load all completed experiments
    experiments = safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    
    if not experiments:
        print("No completed experiments found.")
        return
    
    # Filter experiments based on arguments
    if args.experiment:
        # Filter by specific experiment ID or UID
        target = args.experiment
        filtered_exps = [exp for exp in experiments if 
                         exp.get('experiment_id') == target or 
                         exp.get('uid') == target]
        
        if not filtered_exps:
            print(f"No experiment found with ID or UID: {target}")
            return
        
        print(f"Processing experiment: {target}")
        experiments = filtered_exps
    
    elif args.recent:
        # Sort by timestamp if available
        try:
            sorted_exps = sorted(
                experiments, 
                key=lambda x: x.get('training_time', ''), 
                reverse=True
            )
            experiments = sorted_exps[:args.recent]
            print(f"Processing {len(experiments)} most recent experiments")
        except Exception as e:
            print(f"Error sorting experiments by time: {e}")
            print("Processing all experiments instead")
    
    else:
        # Process all
        print(f"Processing all {len(experiments)} completed experiments")
    
    # Generate graphs for each experiment
    success_count = 0
    for exp in experiments:
        if generate_loss_graph(exp, animated=args.animated):
            success_count += 1
    
    print(f"Generated {success_count} new {'animated' if args.animated else 'static'} loss graphs")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

def main():
    args = parse_arguments()
    process_experiments(args)

if __name__ == "__main__":
    main()