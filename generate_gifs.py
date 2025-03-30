#!/usr/bin/env python3
"""
Generate Animated Confusion Matrix GIFs for Experiments

This script generates animated GIFs of confusion matrices over epochs for completed experiments.
It reads the path to the confusion matrix history (.npy file) from completed_experiments.json.

Usage:
  python generate_gifs.py                # Process all completed experiments (Default)
  python generate_gifs.py --recent N     # Process only the N most recent experiments
  python generate_gifs.py --experiment YYYYMMDD-N  # Process a specific experiment
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns # Use seaborn for consistent plotting style

# Import utils for safe_json_load and paths
import models.common.utils as utils
from pathlib import Path

# --- Constants ---
# Use utils constants for file paths
COMPLETED_EXPERIMENTS_FILE = utils.COMPLETED_EXPERIMENTS_FILE
# Define output dir relative to project root found by utils
OUTPUT_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'cm') # Save GIFs in data/cm
# Animation Speed
ANIMATION_FPS = 2

# Ensure output directory exists at the start
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Argument Parsing ---
def parse_arguments():
    """Parse command line arguments for experiment selection."""
    parser = argparse.ArgumentParser(description='Generate animated confusion matrix GIFs')
    # Filter options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--recent', type=int, help='Process only N most recent experiments (based on completion order)')
    group.add_argument('--experiment', type=str, help='Process a specific experiment ID (e.g., YYYYMMDD-N)')
    # '--all' is the default behavior if no filter is specified
    return parser.parse_args()

# --- Metadata Loading (no changes needed) ---
def get_class_names_from_metadata(exp_id):
    """Loads class names from the experiment's metadata JSON file."""
    metadata_filename = f"data/models/{exp_id}_metadata.json"
    metadata_path = os.path.join(utils._PROJECT_ROOT, metadata_filename)
    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found for {exp_id} at {metadata_path}. Cannot get class names.")
        return None
    try:
        with open(metadata_path, 'r') as f: metadata = json.load(f)
        classes = metadata.get('label_encoder_classes')
        if classes and isinstance(classes, list): return classes
        else: print(f"Warning: 'label_encoder_classes' not found or invalid in {metadata_path}."); return None
    except Exception as e:
        print(f"Error reading metadata file {metadata_path}: {e}"); return None

# --- Helper to get experiment details ---
def get_experiment_details_for_gif(experiment):
    """Extracts key parameters for display, formatted into two lines."""
    line1_parts = []
    line2_parts = []
    exp_id = experiment.get('experiment_id', 'N/A') # Keep ID for main title

    # Data parameters -> Line 1
    data_params = experiment.get('data_params', {})
    num_samples = data_params.get('num_samples', 'N/A')
    sample_length = data_params.get('sample_length', 'N/A')
    line1_parts.append(f"Data: {num_samples}x{sample_length}")

    # Hyperparameters -> Line 1
    hyperparams = experiment.get('hyperparams', {})
    d_model = hyperparams.get('d_model', 'N/A')
    nhead = hyperparams.get('nhead', 'N/A')
    layers = hyperparams.get('num_encoder_layers', 'N/A')
    ff = hyperparams.get('dim_feedforward', 'N/A')
    lr = hyperparams.get('learning_rate', 'N/A')
    lr_str = f"{lr:.1e}" if isinstance(lr, float) else str(lr)
    line1_parts.append(f"Model: d={d_model}, h={nhead}, lyr={layers}, ff={ff}, lr={lr_str}")

    # Metrics summary -> Line 2
    metrics = experiment.get('metrics', {})
    best_acc = metrics.get('best_val_accuracy')
    best_epoch = metrics.get('best_epoch')
    status = "Stopped Early" if metrics.get('stopped_early') else "Completed"
    epochs_run = metrics.get('epochs_completed', 'N/A')
    if best_acc is not None and best_epoch is not None:
         line2_parts.append(f"Result: Best Acc {best_acc:.4f} @ Ep {best_epoch} ({status}, {epochs_run} epochs)")
    else:
         line2_parts.append(f"Result: {status} ({epochs_run} epochs)")

    # Combine lines with a newline character
    return " | ".join(line1_parts) + "\n" + " | ".join(line2_parts)


def create_animated_cm_gif(exp_id, experiment_details, cm_history_3d_array, class_names, output_path):
    """
    Creates an animated GIF of confusion matrices over epochs from a 3D array.

    Args:
        exp_id (str): The experiment ID for titling.
        experiment_details (str): Formatted string of key parameters/results.
        cm_history_3d_array (np.ndarray): 3D array of shape (epochs, classes, classes).
        class_names (list): List of strings for class labels.
        output_path (str): Full path to save the output GIF.
    """
    num_epochs, num_classes, _ = cm_history_3d_array.shape

    # Adjust figure size dynamically, add more height for titles
    fig_width = max(9, num_classes * 0.9)
    fig_height = max(8, num_classes * 0.8 + 1.5) # Add extra height for titles
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.24, right=0.95) # Adjust margins

    # Add main title and experiment details (static)
    fig.suptitle(f"Confusion Matrix History: {exp_id}", fontsize=14, y=0.97)
    fig.text(0.5, 0.90, experiment_details, ha='center', va='top', fontsize=9, wrap=True)

    # Find max value across all matrices for consistent color scaling
    max_val = cm_history_3d_array.max()
    if max_val == 0: max_val = 1

    def update(frame):
        # Updates the plot for a single frame of the animation.
        ax.clear() # Clear axes content for the new frame
        epoch_num = frame + 1
        cm_numeric = cm_history_3d_array[frame]

        # Calculate accuracy for this epoch
        correct = np.diag(cm_numeric).sum()
        total = cm_numeric.sum()
        accuracy = correct / total if total > 0 else 0

        # Generate custom annotation labels (hide zeros)
        annot_labels = np.where(cm_numeric > 0, cm_numeric.astype(str), "")

        # Use seaborn for plotting
        sns.heatmap(cm_numeric, annot=annot_labels, fmt="", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.2, linecolor='lightgrey', # Thinner, lighter grid
                    cbar=False,
                    ax=ax, vmin=0, vmax=max_val,
                    annot_kws={"size": 8})

        # Add axis labels and per-frame title
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10, labelpad=15)
        ax.set_title(f"Epoch {epoch_num}/{num_epochs} | Accuracy: {accuracy:.4f}", fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    try:
        # Create animation
        ani = FuncAnimation(fig, update, frames=num_epochs, repeat=False)
        print(f"Saving animation for {exp_id} ({num_epochs} frames, {ANIMATION_FPS} fps)...")
        writer = PillowWriter(fps=ANIMATION_FPS) # Use constant
        ani.save(output_path, writer=writer)
        print(f"Created animated CM GIF: {output_path}")
        return True
    except Exception as e:
        print(f"Error during animation creation/saving for {exp_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        plt.close(fig)
        plt.close('all')

# --- Refactored Processing Function ---
def process_experiment_for_gif(experiment):
    """Loads data and generates GIF for a single experiment."""
    exp_id = experiment.get('experiment_id')
    if not exp_id:
        print("Skipping experiment: Missing 'experiment_id'")
        return False

    output_filename = f"{exp_id}_conf_matrix.gif"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    if os.path.exists(output_path):
        return False # Nothing new generated

    cm_history_filename_rel = experiment.get('cm_history_filename')
    if not cm_history_filename_rel:
        print(f"Skipping GIF for {exp_id}: 'cm_history_filename' not found.")
        return False

    cm_history_path_abs = os.path.join(utils._PROJECT_ROOT, cm_history_filename_rel)
    if not os.path.exists(cm_history_path_abs):
        print(f"Skipping GIF for {exp_id}: CM History file not found at {cm_history_path_abs}")
        return False

    try:
        # Load the 3D numeric array
        cm_history_3d_array = np.load(cm_history_path_abs)
        # Basic validation
        if not isinstance(cm_history_3d_array, np.ndarray) or cm_history_3d_array.ndim != 3:
             print(f"ERROR: Loaded CM history for {exp_id} is not a 3D array (shape: {cm_history_3d_array.shape}).")
             return False
        if not np.issubdtype(cm_history_3d_array.dtype, np.number):
             print(f"ERROR: Loaded CM history for {exp_id} does not have a numeric dtype (dtype: {cm_history_3d_array.dtype}).")
             return False
        if cm_history_3d_array.shape[0] == 0:
             print(f"Skipping GIF for {exp_id}: Loaded CM history has zero epochs.")
             return False
    except Exception as e:
        print(f"ERROR loading or validating CM history for {exp_id}: {e}")
        return False

    # Get class names from metadata
    class_names = get_class_names_from_metadata(exp_id)
    num_classes = cm_history_3d_array.shape[1]
    if not class_names or len(class_names) != num_classes:
        print(f"Warning: Class names mismatch/missing for {exp_id}. Using generic labels.")
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Get experiment details string
    details_str = get_experiment_details_for_gif(experiment)

    # Create the GIF
    return create_animated_cm_gif(exp_id, details_str, cm_history_3d_array, class_names, output_path)


# --- Main Execution ---
def main():
    """Main execution function."""
    print(f"Using project root: {utils._PROJECT_ROOT}")
    print(f"Output directory set to: {OUTPUT_DIR}")
    args = parse_arguments() # Parse args for experiment selection

    experiments = utils.safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    if not experiments:
        print(f"No completed experiments found in {COMPLETED_EXPERIMENTS_FILE}.")
        return

    # Filter experiments
    experiments_to_process = experiments # Default to all
    if args.experiment:
        target_id = args.experiment
        filtered_exps = [exp for exp in experiments if exp.get('experiment_id') == target_id]
        if not filtered_exps:
            print(f"No experiment found with ID: {target_id}")
            return
        print(f"Processing specific experiment: {target_id}")
        experiments_to_process = filtered_exps
    elif args.recent:
        num_recent = args.recent
        if num_recent <= 0:
             print("Number of recent experiments must be positive.")
             return
        # Ensure slicing doesn't go out of bounds if fewer experiments exist
        experiments_to_process = experiments[-min(num_recent, len(experiments)):]
        print(f"Processing {len(experiments_to_process)} most recent experiments")
    else:
        # Processing all (or already filtered by --experiment)
        if not args.experiment: # Only print if not specific experiment
            print(f"Processing all {len(experiments)} completed experiments")


    if not experiments_to_process: # Check if filtering resulted in empty list
         print("No experiments selected for processing after filtering.")
         return

    # Generate GIFs
    success_count = 0
    processed_count = 0
    for exp in experiments_to_process:
        processed_count += 1
        # Pass experiment data to the processing function
        if process_experiment_for_gif(exp):
            success_count += 1

    print(f"\nProcessed {processed_count} experiments.")
    print(f"Generated {success_count} new animated CM GIFs at {ANIMATION_FPS} fps.")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
