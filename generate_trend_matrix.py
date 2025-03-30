#!/usr/bin/env python3
"""
Generate Trend Matrix Visualization

Creates visualizations showing how classification counts evolved during training
for specified experiments, loading data from associated _cm_history.npy files.

Usage:
  python generate_trend_matrix.py --experiment YYYYMMDD-N  # Single experiment
  python generate_trend_matrix.py --all                   # All completed experiments
  python generate_trend_matrix.py --recent N              # N most recent experiments
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# Import utils
import models.common.utils as utils
from pathlib import Path

# Use utils constants/paths
COMPLETED_EXPERIMENTS_FILE = utils.COMPLETED_EXPERIMENTS_FILE
OUTPUT_DIR = os.path.join(utils._PROJECT_ROOT, 'data', 'trend_matrix')

# Ensure output directory exists at the start
os.makedirs(OUTPUT_DIR, exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate trend matrix visualization')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--experiment', type=str,
                       help='Generate for a specific experiment ID (e.g., YYYYMMDD-N)')
    group.add_argument('--all', action='store_true',
                       help='Generate for ALL completed experiments.')
    group.add_argument('--recent', type=int,
                       help='Generate for the N most recent completed experiments.')
    return parser.parse_args()

def find_experiment(target_id):
    """Find experiment configuration by ID, ensuring it's completed."""
    print(f"Searching for experiment {target_id}...")
    exp_config = utils.get_experiment_config_by_id(target_id)
    if exp_config:
        # Check if necessary artifact paths exist in the record and files exist
        meta_file = exp_config.get('metadata_filename')
        cm_hist_file = exp_config.get('cm_history_filename')
        if meta_file and cm_hist_file:
             metadata_path = os.path.join(utils._PROJECT_ROOT, meta_file)
             cm_history_path = os.path.join(utils._PROJECT_ROOT, cm_hist_file)
             if os.path.exists(metadata_path) and os.path.exists(cm_history_path):
                  print(f"Found completed experiment {target_id} with required artifacts.")
                  return exp_config
             else: print(f"Exp {target_id} record found, but artifact file(s) missing.")
        else: print(f"Exp {target_id} record found, but missing metadata/cm_history paths.")
    else: print(f"Experiment {target_id} not found.")
    return None

def get_class_names_from_metadata(exp_id):
    """Loads class names from the experiment's metadata JSON file."""
    metadata_path = os.path.join(utils._PROJECT_ROOT, f"data/models/{exp_id}_metadata.json")
    if not os.path.exists(metadata_path): return None
    try:
        with open(metadata_path, 'r') as f: metadata = json.load(f)
        classes = metadata.get('label_encoder_classes')
        return classes if classes and isinstance(classes, list) else None
    except Exception as e: print(f"Error reading metadata {metadata_path}: {e}"); return None

def load_cm_history(experiment):
    """Loads the CM history array from the .npy file specified in the experiment record."""
    exp_id = experiment.get('experiment_id')
    cm_history_filename_rel = experiment.get('cm_history_filename')
    if not cm_history_filename_rel: print(f"ERROR: 'cm_history_filename' not found for {exp_id}."); return None
    cm_history_path_abs = os.path.join(utils._PROJECT_ROOT, cm_history_filename_rel)
    if not os.path.exists(cm_history_path_abs): print(f"ERROR: CM History file not found: {cm_history_path_abs}"); return None
    try:
        cm_history_3d_array = np.load(cm_history_path_abs)
        # Validation
        if not isinstance(cm_history_3d_array, np.ndarray) or cm_history_3d_array.ndim != 3: return None
        if not np.issubdtype(cm_history_3d_array.dtype, np.number): return None
        if cm_history_3d_array.shape[0] == 0: return None
        print(f"Loaded CM History shape: {cm_history_3d_array.shape}")
        return cm_history_3d_array
    except Exception as e: print(f"ERROR loading/validating CM history for {exp_id}: {e}"); return None

def create_trend_matrix_plot(exp_id, cm_history_3d_array, class_names, output_path):
    """
    Creates the trend matrix visualization with embedded line plots.

    Args:
        exp_id (str): Experiment ID for title.
        cm_history_3d_array (np.ndarray): 3D array (epochs, classes, classes).
        class_names (list): List of class name strings.
        output_path (str): Path to save the output image.

    Returns:
        bool: True if visualization was created successfully.
    """
    try:
        num_epochs, num_classes, _ = cm_history_3d_array.shape
        if num_epochs == 0 or num_classes == 0: return False
        final_matrix = cm_history_3d_array[-1]
        if not class_names or len(class_names) != num_classes:
            print(f"Warning: Using generic class names for {exp_id}.")
            class_names = [f"Class {i}" for i in range(num_classes)]

        # --- Layout Adjustments ---
        # Increase base width/height and multipliers slightly for labels
        fig_width = max(10, num_classes * 1.1)
        fig_height = max(9, num_classes * 1.0 + 2.0) # Add more height for top/bottom elements
        fig = plt.figure(figsize=(fig_width, fig_height))

        # --- Use subplots_adjust for overall margins ---
        # Provide ample space on left and bottom, adjust top
        plt.subplots_adjust(left=0.2, right=0.96, bottom=0.2, top=0.90, wspace=0.15, hspace=0.15)

        # --- Title (using fig.suptitle for positioning) ---
        fig.suptitle(f"Confusion Matrix Trends for {exp_id}\n({num_epochs} Epochs)",
                     fontsize=14, fontweight='bold', y=0.97) # Position near the top margin

        # --- Axis Meta-Labels (using fig.text) ---
        # Position relative to the figure edges
        fig.text(0.02, 0.55, "True Class", ha='left', va='center', fontsize=12, rotation=90)
        fig.text(0.5, 0.04, "Predicted Class", ha='center', va='bottom', fontsize=12)

        # Calculate normalization and color scale
        row_sums = final_matrix.sum(axis=1); norm_factors = np.where(row_sums > 0, row_sums, 1)
        cmap = plt.cm.Blues
        global_max_trend_value = max(1, cm_history_3d_array.max())

        # Create grid for the plots themselves
        # This grid lives within the margins defined by subplots_adjust
        inner_grid = gridspec.GridSpec(num_classes, num_classes, figure=fig,
                                      wspace=0.1, hspace=0.1) # Let subplots_adjust handle overall position

        # Create each cell plot
        for i in range(num_classes): # True Class (Row)
            for j in range(num_classes): # Predicted Class (Col)
                # Use inner_grid indices
                cell_ax = fig.add_subplot(inner_grid[i, j])

                # Get trend data and final value
                cell_trend = cm_history_3d_array[:, i, j]
                final_cell_value = final_matrix[i, j]

                # Set background color
                color_intensity = final_cell_value / norm_factors[i]
                cell_color = cmap(color_intensity)
                cell_ax.patch.set_facecolor(cell_color); cell_ax.patch.set_alpha(0.4)

                # Plot trend line
                if num_epochs > 1:
                    x = range(1, num_epochs + 1)
                    cell_ax.plot(x, cell_trend, 'k-', linewidth=1.0)
                    cell_ax.set_ylim(0, global_max_trend_value * 1.05)
                else:
                     cell_ax.plot([],[]); cell_ax.set_ylim(0, global_max_trend_value * 1.05)

                # Add final count text (hide zeros)
                if final_cell_value > 0:
                     cell_ax.text(0.5, 0.5, str(int(final_cell_value)), ha='center', va='center',
                                  transform=cell_ax.transAxes, fontsize=9, fontweight='bold',
                                  color='white' if color_intensity > 0.6 else 'black')

                # --- Clean up axes and add class labels ---
                cell_ax.set_xticks([]); cell_ax.set_yticks([]) # Hide ticks

                # Add class labels ONLY on the edges of the entire grid
                # Bottom edge (Predicted): Add labels below the last row
                if i == (num_classes - 1):
                    cell_ax.set_xlabel(class_names[j], fontsize=8, rotation=45, ha='right', va='top', labelpad=2)
                # Left edge (True): Add labels left of the first column
                if j == 0:
                    cell_ax.set_ylabel(class_names[i], fontsize=8, rotation=0, ha='right', va='center', labelpad=5)

                # Diagonal spines
                if i == j:
                    for spine in cell_ax.spines.values(): spine.set_visible(True); spine.set_color('green'); spine.set_linewidth(1.5)
                else:
                    for spine in cell_ax.spines.values(): spine.set_visible(False)

        # Save the figure
        plt.savefig(output_path, dpi=150) # Remove bbox_inches='tight' as it conflicts with manual layout
        print(f"Created trend matrix plot: {output_path}")
        return True

    except Exception as e:
        print(f"Error creating trend matrix plot for {exp_id}: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        plt.close('all')

def main():
    """Main execution function."""
    print(f"Using project root: {utils._PROJECT_ROOT}")
    print(f"Output directory set to: {OUTPUT_DIR}")
    args = parse_arguments()

    all_experiments = utils.safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    if not all_experiments:
        print(f"No completed experiments found in {COMPLETED_EXPERIMENTS_FILE}.")
        return

    # Determine which experiments to process
    experiments_to_process = []
    if args.experiment:
        exp = find_experiment(args.experiment)
        if exp: experiments_to_process.append(exp)
    elif args.recent:
        num_recent = args.recent
        if num_recent <= 0: print("Number of recent experiments must be positive."); return
        # Filter for valid completed experiments first before taking recent
        valid_completed = [e for e in all_experiments if find_experiment(e.get('experiment_id'))]
        experiments_to_process = valid_completed[-min(num_recent, len(valid_completed)):]
        print(f"Processing {len(experiments_to_process)} most recent valid experiments.")
    elif args.all:
        print(f"Processing ALL {len(all_experiments)} completed experiments (if valid)...")
        experiments_to_process = [e for e in all_experiments if find_experiment(e.get('experiment_id'))]
    else: # Default behavior
        print(f"Processing ALL {len(all_experiments)} completed experiments (Default)...")
        experiments_to_process = [e for e in all_experiments if find_experiment(e.get('experiment_id'))]

    if not experiments_to_process:
         print("No valid experiments selected for processing.")
         return

    # Generate plots
    success_count = 0
    processed_count = 0
    for experiment in experiments_to_process:
        processed_count += 1
        exp_id = experiment.get('experiment_id')
        if not exp_id: continue

        output_filename = f"{exp_id}_trend_matrix.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        if os.path.exists(output_path): continue

        cm_history_3d_array = load_cm_history(experiment)
        if cm_history_3d_array is None: continue
        class_names = get_class_names_from_metadata(exp_id)

        if create_trend_matrix_plot(exp_id, cm_history_3d_array, class_names, output_path):
            success_count += 1

    print(f"\nProcessed {processed_count} experiments.")
    print(f"Generated {success_count} new trend matrix plots.")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
