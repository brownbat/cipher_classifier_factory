#!/usr/bin/env python3
"""
Generate Confusion Matrix GIFs

This script generates animated GIFs of confusion matrices for completed experiments.
It handles visualization separately from the training process.

Usage:
  python generate_gifs.py               # Process all completed experiments
  python generate_gifs.py --recent N    # Process only the N most recent experiments
  python generate_gifs.py --experiment exp_id_123  # Process a specific experiment
  python generate_gifs.py --all         # Process all experiments (default)
"""

import os
import sys
import json
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio.v2 as imageio
from models.common.utils import safe_json_load

# Set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# File paths
COMPLETED_EXPERIMENTS_FILE = 'data/completed_experiments.json'
GIF_OUTPUT_DIR = 'data/cm'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate confusion matrix GIFs')
    
    # Filter options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--recent', type=int, help='Process only N most recent experiments')
    group.add_argument('--experiment', type=str, help='Process a specific experiment ID or UID')
    group.add_argument('--all', action='store_true', help='Process all experiments (default)')
    
    return parser.parse_args()


def plot_confusion_matrix(conf_matrix, class_names, epoch, total_epochs, output_file):
    """
    Plot a single confusion matrix and save it as an image.
    
    Args:
        conf_matrix: The confusion matrix to plot
        class_names: List of class names
        epoch: Current epoch number
        total_epochs: Total number of epochs
        output_file: File path to save the image
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', 
                xticklabels=class_names, yticklabels=class_names)
    
    plt.title(f'Confusion Matrix - Epoch {epoch+1}/{total_epochs}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file)
    plt.close()
    
    return output_file


def create_gif(image_files, output_file, duration=0.5):
    """
    Create a GIF from a list of image files.
    
    Args:
        image_files: List of image file paths
        output_file: Output GIF file path
        duration: Duration of each frame in seconds
    """
    images = [imageio.imread(file) for file in image_files]
    imageio.mimsave(output_file, images, duration=duration)
    
    # Clean up temporary image files
    for file in image_files:
        if os.path.exists(file):
            os.remove(file)


def generate_confusion_matrix_gif(experiment, output_dir=GIF_OUTPUT_DIR):
    """
    Generate a GIF of confusion matrices from an experiment.
    
    Args:
        experiment: Experiment data dictionary
        output_dir: Directory to save the GIF
    
    Returns:
        bool: True if GIF was created, False otherwise
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if experiment has confusion matrices
    if 'metrics' not in experiment or 'conf_matrix' not in experiment['metrics']:
        print(f"No confusion matrices found for experiment {experiment.get('experiment_id', 'unknown')}")
        return False
    
    # Check if UID is available
    if 'uid' not in experiment:
        print(f"No UID found for experiment {experiment.get('experiment_id', 'unknown')}")
        return False
    
    # Set up output file path
    gif_filename = f"{output_dir}/{experiment['uid']}_conf_matrix.gif"
    
    # Skip if GIF already exists
    if os.path.exists(gif_filename):
        print(f"GIF already exists for {experiment['uid']}")
        return False
    
    print(f"Generating GIF for {experiment['uid']}...")
    
    # Extract confusion matrices and other data
    conf_matrices = experiment['metrics']['conf_matrix']
    total_epochs = len(conf_matrices)
    
    # Try to get class names
    try:
        # First check metadata file
        metadata_file = experiment.get('metadata_filename')
        if metadata_file and os.path.exists(metadata_file):
            import pickle
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                classes = metadata.get('label_encoder_classes', [])
        else:
            # Fall back to cipher names from the experiment
            classes = experiment.get('data_params', {}).get('ciphers', [[]])[0]
    except Exception as e:
        print(f"Error getting class names: {e}")
        classes = [f"Class {i}" for i in range(conf_matrices[0].shape[0])]
    
    # Generate images for each epoch
    image_files = []
    for epoch, conf_matrix in enumerate(conf_matrices):
        if isinstance(conf_matrix, list):
            conf_matrix = np.array(conf_matrix)
        
        # Create temporary file for this frame
        tmp_file = f"{output_dir}/tmp_{experiment['uid']}_epoch_{epoch+1}.png"
        plot_confusion_matrix(conf_matrix, classes, epoch, total_epochs, tmp_file)
        image_files.append(tmp_file)
    
    # Create the GIF from the image files
    create_gif(image_files, gif_filename)
    print(f"Created GIF: {gif_filename}")
    
    return True


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
    
    # Generate GIFs for each experiment
    success_count = 0
    for exp in experiments:
        if generate_confusion_matrix_gif(exp):
            success_count += 1
    
    print(f"Generated {success_count} new GIFs")


def main():
    args = parse_arguments()
    process_experiments(args)


if __name__ == "__main__":
    main()