#!/usr/bin/env python3
"""
Generate Trend Matrix Visualization

This script creates a specialized visualization that combines a confusion matrix with
embedded trend lines for each cell, showing how classification patterns evolved during training.

Usage:
  python generate_trend_matrix.py --experiment EXP_ID  # Generate for a specific experiment
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from models.common.utils import safe_json_load

# Set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# File paths
COMPLETED_EXPERIMENTS_FILE = 'data/completed_experiments.json'
OUTPUT_DIR = 'data/trend_matrix'

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate trend matrix visualization')
    parser.add_argument('--experiment', type=str, required=True, 
                        help='Experiment ID or UID to visualize')
    return parser.parse_args()

def find_experiment(target_id):
    """Find experiment by ID or UID."""
    # Direct JSON loading for large files to avoid memory issues
    try:
        print(f"Loading experiments from {COMPLETED_EXPERIMENTS_FILE}...")
        with open(COMPLETED_EXPERIMENTS_FILE, 'r') as file:
            experiments = json.load(file)
            print(f"Successfully loaded {len(experiments)} experiments")
    except Exception as e:
        print(f"Error loading experiments: {e}")
        return None
    
    if not experiments:
        print("No completed experiments found.")
        return None
    
    # Try to find the experiment by ID or UID
    for exp in experiments:
        if exp.get('experiment_id') == target_id or exp.get('uid') == target_id:
            print(f"Found experiment {target_id}")
            return exp
    
    print(f"No experiment found with ID or UID: {target_id}")
    return None

def get_class_names(experiment):
    """Get class names from experiment data."""
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
        first_matrix = experiment['metrics']['conf_matrix'][0]
        if isinstance(first_matrix, list):
            first_matrix = np.array(first_matrix)
        num_classes = first_matrix.shape[0]
        classes = [f"Class {i}" for i in range(num_classes)]
    
    return classes

def create_trend_matrix(experiment, output_path):
    """
    Create a visualization that combines a confusion matrix with trend lines.
    
    Args:
        experiment: Experiment data dictionary
        output_path: Path to save the output image
        
    Returns:
        bool: True if visualization was created successfully
    """
    # Check if confusion matrices are available
    if 'metrics' not in experiment or 'conf_matrix' not in experiment['metrics']:
        print(f"No confusion matrices found for experiment {experiment.get('experiment_id', 'unknown')}")
        return False
    
    try:
        # Extract confusion matrices
        conf_matrices = experiment['metrics']['conf_matrix']
        if not conf_matrices:
            print(f"Empty confusion matrix list for {experiment.get('experiment_id', 'unknown')}")
            return False
        
        # Convert to numpy arrays if they are lists
        conf_matrices = [np.array(cm) if isinstance(cm, list) else cm for cm in conf_matrices]
        
        # Get the last confusion matrix (final state)
        final_matrix = conf_matrices[-1]
        num_classes = final_matrix.shape[0]
        
        # Get class names
        class_names = get_class_names(experiment)
        if len(class_names) != num_classes:
            print(f"Warning: Class names count ({len(class_names)}) doesn't match matrix dimensions ({num_classes})")
            class_names = [f"Class {i}" for i in range(num_classes)]
        
        # Get experiment details for title
        exp_id = experiment.get('experiment_id', 'unknown')
        uid = experiment.get('uid', 'unknown')
        epochs = len(conf_matrices)
        
        # Extract accuracy information if available
        val_accuracy = experiment['metrics'].get('val_accuracy', [])
        final_accuracy = val_accuracy[-1] if val_accuracy else None
        
        # Create the trend matrix visualization
        fig = plt.figure(figsize=(num_classes * 1.5 + 2, num_classes * 1.5 + 2))
        
        # Create a grid layout with main title space
        outer_grid = gridspec.GridSpec(2, 1, height_ratios=[1, 20], figure=fig)
        
        # Add title in the top section
        title_ax = fig.add_subplot(outer_grid[0])
        title_text = f"Trend Matrix for {exp_id} ({uid})\n"
        if final_accuracy is not None:
            title_text += f"Final Accuracy: {final_accuracy:.4f}"
        title_ax.text(0.5, 0.5, title_text, ha='center', va='center', fontsize=14, fontweight='bold')
        title_ax.axis('off')
        
        # Create the main grid for the confusion matrix
        main_grid = gridspec.GridSpecFromSubplotSpec(
            num_classes, num_classes, subplot_spec=outer_grid[1], 
            wspace=0.1, hspace=0.1
        )
        
        # Calculate the normalization factor for colors
        norm_factor = np.max(final_matrix)
        
        # Define colormap for the cell backgrounds
        cmap = plt.cm.Blues
        
        # Track min/max trend values for consistent scaling
        trend_min = float('inf')
        trend_max = float('-inf')
        
        # Extract trend data for each cell
        trend_data = np.zeros((num_classes, num_classes, epochs))
        for i in range(num_classes):
            for j in range(num_classes):
                for epoch in range(epochs):
                    trend_data[i, j, epoch] = conf_matrices[epoch][i, j]
                
                # Update global min/max
                cell_min = np.min(trend_data[i, j])
                cell_max = np.max(trend_data[i, j])
                trend_min = min(trend_min, cell_min)
                trend_max = max(trend_max, cell_max)
        
        # Create each cell with a trend line
        for i in range(num_classes):
            for j in range(num_classes):
                cell_ax = fig.add_subplot(main_grid[i, j])
                
                # Get the trend data for this cell
                cell_trend = trend_data[i, j]
                
                # Set cell background color based on final value
                cell_value = final_matrix[i, j]
                color_intensity = cell_value / norm_factor
                cell_color = cmap(color_intensity)
                cell_ax.patch.set_facecolor(cell_color)
                cell_ax.patch.set_alpha(0.3)  # Make it lighter to see the trend line
                
                # Draw the trend line
                x = range(1, epochs + 1)
                cell_ax.plot(x, cell_trend, 'k-', linewidth=1.5)
                
                # Add the final value as text
                cell_ax.text(0.5, 0.5, str(int(cell_value)), 
                           ha='center', va='center', transform=cell_ax.transAxes,
                           fontsize=10, fontweight='bold')
                
                # Clean up the subplot
                cell_ax.set_xticks([])
                cell_ax.set_yticks([])
                
                # Only show spines for diagonal cells
                if i == j:
                    for spine in cell_ax.spines.values():
                        spine.set_visible(True)
                        spine.set_color('green')
                        spine.set_linewidth(2)
                else:
                    for spine in cell_ax.spines.values():
                        spine.set_visible(False)
                
                # Add class labels to the first row and first column
                if i == 0:
                    cell_ax.set_title(class_names[j], fontsize=8, rotation=45, ha='left')
                if j == 0:
                    cell_ax.set_ylabel(class_names[i], fontsize=8)
        
        # Add axes labels
        fig.text(0.5, 0.02, 'Predicted', ha='center', fontsize=12)
        fig.text(0.02, 0.5, 'True', va='center', rotation='vertical', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path, dpi=150)
        plt.close()
        
        print(f"Created trend matrix visualization: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error creating trend matrix for {experiment.get('experiment_id', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    try:
        args = parse_arguments()
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Find the experiment
        experiment = find_experiment(args.experiment)
        if not experiment:
            print("Failed to find experiment. Please check the experiment ID/UID and try again.")
            return
        
        # Set up output file path
        uid = experiment.get('uid', 'unknown')
        output_path = os.path.join(OUTPUT_DIR, f"{uid}_trend_matrix.png")
        
        # Create the trend matrix visualization
        success = create_trend_matrix(experiment, output_path)
        
        if success:
            print(f"Successfully created trend matrix visualization!")
            print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
        else:
            print("Failed to create trend matrix visualization.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()