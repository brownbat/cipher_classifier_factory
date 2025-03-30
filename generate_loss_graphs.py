#!/usr/bin/env python3
"""
Generate Loss Curves for Experiments

This script generates static images or animated GIFs of training and validation loss curves
for completed experiments based on the canonical experiment_id.

Usage:
  python generate_loss_graphs.py                # Process all completed experiments (Default)
  python generate_loss_graphs.py --recent N     # Process only the N most recent experiments
  python generate_loss_graphs.py --experiment YYYYMMDD-N  # Process a specific experiment
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
# Import utils for safe_json_load and paths
import models.common.utils as utils
from pathlib import Path # Use pathlib for path manipulation

# Use utils constants for file paths
COMPLETED_EXPERIMENTS_FILE = utils.COMPLETED_EXPERIMENTS_FILE
# Define output dir relative to project root found by utils
OUTPUT_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'loss_graphs')


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate loss curves for experiments')

    # Filter options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--recent', type=int, help='Process only N most recent experiments (based on completion order)')
    group.add_argument('--experiment', type=str, help='Process a specific experiment ID (e.g., YYYYMMDD-N)')
    # '--all' is the default behavior if no filter specified

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
    exp_id = experiment.get('experiment_id', 'unknown_id')

    # Check metrics using keys saved by train.py
    metrics = experiment.get('metrics', {})
    train_losses = metrics.get('train_loss_curve')
    val_losses = metrics.get('val_loss_curve')

    if not train_losses or not val_losses:
        print(f"No sufficient loss curve data found for experiment {exp_id}")
        return False

    try:
        epochs = range(1, len(train_losses) + 1)

        # Get hyperparameters for subtitle
        hyperparams = experiment.get('hyperparams', {})
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        num_encoder_layers = hyperparams.get('num_encoder_layers', 'N/A')
        dim_feedforward = hyperparams.get('dim_feedforward', 'N/A')
        learning_rate = hyperparams.get('learning_rate', 'N/A')
        lr_str = f"{learning_rate:.1e}" if isinstance(learning_rate, float) else str(learning_rate)


        # Get accuracy info from metrics
        accuracy_info = ""
        best_accuracy = metrics.get('best_val_accuracy')
        if best_accuracy is not None:
             accuracy_info = f"Best Accuracy: {best_accuracy:.4f}"
        else:
             val_accuracy_curve = metrics.get('val_accuracy_curve', [])
             if val_accuracy_curve:
                 accuracy_info = f"Final Accuracy: {val_accuracy_curve[-1]:.4f}"

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss Curves for {exp_id}\n' +
                 f'd={d_model}, h={nhead}, lyr={num_encoder_layers}, ff={dim_feedforward}, lr={lr_str}\n' +
                 accuracy_info)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()

        print(f"Created loss graph: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating static loss graph for {exp_id}: {e}")
        return False
    finally:
        plt.close('all') # Ensure figures are closed

def create_animated_loss_gif(experiment, output_path):
    """
    Create an animated GIF showing the progression of loss curves.

    Args:
        experiment: Experiment data dictionary
        output_path: Path to save the output GIF

    Returns:
        bool: True if GIF was created successfully
    """
    exp_id = experiment.get('experiment_id', 'unknown_id')
    metrics = experiment.get('metrics', {})
    train_losses = metrics.get('train_loss_curve')
    val_losses = metrics.get('val_loss_curve')

    if not train_losses or not val_losses:
        print(f"No sufficient loss curve data found for experiment {exp_id}")
        return False

    try:
        total_epochs = len(train_losses)
        hyperparams = experiment.get('hyperparams', {})
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        num_encoder_layers = hyperparams.get('num_encoder_layers', 'N/A')
        dim_feedforward = hyperparams.get('dim_feedforward', 'N/A')
        learning_rate = hyperparams.get('learning_rate', 'N/A')
        lr_str = f"{learning_rate:.1e}" if isinstance(learning_rate, float) else str(learning_rate)

        temp_dir = os.path.join(OUTPUT_DIR, f'temp_{exp_id}')
        os.makedirs(temp_dir, exist_ok=True)

        all_losses = [l for l in train_losses + val_losses if l is not None and np.isfinite(l)]
        if not all_losses:
             print(f"Warning: All loss values are None/NaN/Inf for {exp_id}. Cannot generate animation.")
             return False
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        y_range = max_loss - min_loss if max_loss > min_loss else 1.0
        y_min = max(0, min_loss - 0.05 * y_range)
        y_max = max_loss + 0.05 * y_range

        frame_files = []
        frame_step = max(1, total_epochs // 30)

        for frame_idx, epoch_end in enumerate(range(frame_step, total_epochs + frame_step, frame_step)):
            epoch_end = min(epoch_end, total_epochs)
            current_epochs = list(range(1, epoch_end + 1))
            current_train_losses = train_losses[:epoch_end]
            current_val_losses = val_losses[:epoch_end]

            plt.figure(figsize=(10, 6))
            plt.plot(current_epochs, current_train_losses, 'b-', label='Training Loss')
            plt.plot(current_epochs, current_val_losses, 'r-', label='Validation Loss')
            plt.ylim(y_min, y_max)
            plt.xlim(1, total_epochs)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Loss Curves for {exp_id} - Epoch {epoch_end}/{total_epochs}\n' +
                    f'd={d_model}, h={nhead}, lyr={num_encoder_layers}, ff={dim_feedforward}, lr={lr_str}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            frame_file = os.path.join(temp_dir, f'frame_{frame_idx:03d}.png')
            plt.tight_layout()
            plt.savefig(frame_file, dpi=100)
            plt.close()
            frame_files.append(frame_file)

        if not frame_files:
             print(f"No frames generated for {exp_id}. Skipping GIF creation.")
             try: os.rmdir(temp_dir)
             except OSError: pass
             return False

        images = [imageio.imread(file) for file in frame_files]
        imageio.mimsave(output_path, images, duration=0.2, loop=0)

        for file in frame_files:
            try: os.remove(file)
            except OSError as e: print(f"Warning: Could not remove frame file {file}: {e}")
        try: os.rmdir(temp_dir)
        except OSError as e: print(f"Warning: Could not remove temp directory {temp_dir}: {e}")

        print(f"Created animated loss GIF: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating loss animation for {exp_id}: {e}")
        if 'temp_dir' in locals() and os.path.exists(temp_dir):
             import shutil
             try: shutil.rmtree(temp_dir)
             except Exception as clean_e: print(f"Error cleaning temp dir: {clean_e}")
        return False
    finally:
        plt.close('all')

def generate_loss_graph(experiment, output_dir=OUTPUT_DIR, animated=False):
    """Generates a static or animated loss graph for an experiment."""
    os.makedirs(output_dir, exist_ok=True)
    exp_id = experiment.get('experiment_id')
    if not exp_id:
        print(f"Skipping experiment: Missing 'experiment_id'")
        return False
    if 'metrics' not in experiment:
        print(f"Insufficient metric data for experiment {exp_id}")
        return False

    output_filename = f"{exp_id}_loss{'_animated' if animated else ''}.{ 'gif' if animated else 'png'}"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        return False # Indicate nothing new was generated

    if animated:
        return create_animated_loss_gif(experiment, output_path)
    else:
        return plot_static_loss_curves(experiment, output_path)

def process_experiments(args):
    """Loads and filters experiments, then generates graphs."""
    experiments = utils.safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    if not experiments:
        print(f"No completed experiments found in {COMPLETED_EXPERIMENTS_FILE}.")
        return

    experiments_to_process = experiments # Default to all
    if args.experiment:
        target_id = args.experiment
        filtered_exps = [exp for exp in experiments if exp.get('experiment_id') == target_id]
        if not filtered_exps: print(f"No experiment found with ID: {target_id}"); return
        print(f"Processing specific experiment: {target_id}")
        experiments_to_process = filtered_exps
    elif args.recent:
        num_recent = args.recent
        if num_recent <= 0: print("Number of recent experiments must be positive."); return
        experiments_to_process = experiments[-min(num_recent, len(experiments)):]
        print(f"Processing {len(experiments_to_process)} most recent experiments")
    else:
        if not args.experiment: print(f"Processing all {len(experiments)} completed experiments")

    if not experiments_to_process:
        print("No experiments selected for processing after filtering.")
        return

    success_count = 0
    processed_count = 0
    for exp in experiments_to_process:
        processed_count += 1
        if generate_loss_graph(exp, output_dir=OUTPUT_DIR, animated=args.animated):
            success_count += 1

    print(f"\nProcessed {processed_count} experiments.")
    print(f"Generated {success_count} new {'animated' if args.animated else 'static'} loss graphs.")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

def main():
    print(f"Using project root: {utils._PROJECT_ROOT}")
    print(f"Output directory set to: {OUTPUT_DIR}")
    args = parse_arguments()
    process_experiments(args)

if __name__ == "__main__":
    main()
