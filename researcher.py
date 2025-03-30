import datetime
import itertools
import json
import notifications
import os
import torch
import numpy as np
import signal
import sys
import time
import argparse

# Import from our new modular structure
from models import train_model, get_data # Assuming models.__init__ exposes these
from models.common.utils import safe_json_load, convert_ndarray_to_list
from ciphers import _get_cipher_names # Assuming ciphers.py is accessible

# Set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Global flag for signal handling
should_continue = True

# Default parameters - these are used ONLY if nothing is specified via manage_queue.py
# It's generally better to define experiments explicitly via manage_queue.py
default_params = {
    'ciphers': [_get_cipher_names()],
    'num_samples': [100000],
    'sample_length': [500],

    # Transformer hyperparameters
    'd_model': [128],
    'nhead': [8],
    'num_encoder_layers': [2],
    'dim_feedforward': [512],
    'batch_size': [64],
    'dropout_rate': [0.1],
    'learning_rate': [1e-4],
    'early_stopping_patience': [10] # Default patience for early stopping
}

# --- Signal Handling ---
def signal_handler(sig, frame):
    """Handle Ctrl+C or kill signals gracefully."""
    global should_continue
    print('\nSignal received. Attempting graceful shutdown...')
    print('Current experiment checkpoint (if any) should be saved by train.py.')
    print('Run `python researcher.py` again later to resume queue processing.')
    should_continue = False
    # No need to clean checkpoints here, train.py handles its own cleanup on exit/completion
    sys.exit(0) # Exit cleanly

# --- Deprecated / Moved Functionality ---
# plot_confusion_matrices is complex and slow, better run separately via generate_gifs.py
# query_experiments_metrics is likely superseded by suggest_experiments.py analysis

# --- Core Functions ---

import math # Add import for checking NaN/inf if needed, although not strictly used here yet


def get_experiment_details(exp):
    """
    Returns experiment details as a multi-line formatted string,
    focusing on key parameters and best achieved metrics.
    """
    details = []
    details.append("--- Experiment Summary ---")
    exp_id = exp.get('experiment_id', 'N/A')
    details.append(f"ID: {exp_id}")

    # --- Data Parameters ---
    data_params = exp.get('data_params', {})
    num_samples = data_params.get('num_samples', 'N/A')
    sample_length = data_params.get('sample_length', 'N/A')

    # Correctly extract cipher list (handle potential list-of-lists)
    ciphers_list_of_lists = data_params.get('ciphers', []) # Default to empty list
    actual_ciphers = []
    if ciphers_list_of_lists and isinstance(ciphers_list_of_lists[0], list):
        # It's a list containing one list of ciphers
        actual_ciphers = ciphers_list_of_lists[0]
    elif ciphers_list_of_lists and isinstance(ciphers_list_of_lists[0], str):
        # It's already a flat list of strings (less likely based on defaults)
        actual_ciphers = ciphers_list_of_lists
    # Handle case where ciphers might be missing or empty
    ciphers_used_str = ', '.join(actual_ciphers) if actual_ciphers else 'N/A'

    details.append(f"Data: {num_samples} samples, length {sample_length}, ciphers: {ciphers_used_str}")

    # Hyperparameters
    hyperparams = exp.get('hyperparams', {})
    batch_size = hyperparams.get('batch_size', 'N/A')
    dropout = hyperparams.get('dropout_rate', 'N/A')
    lr = hyperparams.get('learning_rate', None)
    d_model = hyperparams.get('d_model', 'N/A')
    nhead = hyperparams.get('nhead', 'N/A')
    layers = hyperparams.get('num_encoder_layers', 'N/A')
    ff_dim = hyperparams.get('dim_feedforward', 'N/A')
    patience = hyperparams.get('early_stopping_patience', 'Default') # Keep displaying patience

    details.append(f"Model: d={d_model}, h={nhead}, lyr={layers}, ff={ff_dim}, drop={dropout}")

    # Format learning rate nicely
    lr_str = "N/A"
    if isinstance(lr, (int, float)):
        # ... (lr formatting logic) ...
        if 0 < abs(lr) < 1e-3:
            lr_str = f"{lr:.1e}"
        else:
            lr_str = f"{lr:.6f}".rstrip('0').rstrip('.')

    # Update Training line - remove MaxEpochs
    details.append(f"Training: BS={batch_size}, LR={lr_str}, Patience={patience}")

    # Metrics (Focus on Best Results)
    metrics = exp.get('metrics', {})
    if metrics:
        stopped_early = metrics.get('stopped_early', False)
        epochs_completed = metrics.get('epochs_completed', 'N/A')
        best_epoch = metrics.get('best_epoch', None)
        best_acc = metrics.get('best_val_accuracy', None)
        best_loss = metrics.get('best_val_loss', None)
        duration = metrics.get('training_duration', None)

        # Update Status line - remove /epochs_max
        status = f"Status: Completed {epochs_completed} epochs"
        if stopped_early:
            status += " (Stopped Early)"
        details.append(status)

        # Best performance line
        best_metrics_parts = []
        if best_epoch is not None and best_epoch > 0:
            best_metrics_parts.append(f"Best @ Ep {best_epoch}")
            if best_acc is not None: best_metrics_parts.append(f"Acc={best_acc:.4f}")
            if best_loss is not None: best_metrics_parts.append(f"Loss={best_loss:.4f}")
        if best_metrics_parts:
             details.append("  " + " | ".join(best_metrics_parts))
        elif epochs_completed != 'N/A' and epochs_completed > 0:
             # Fallback if best metrics weren't saved, show final curve values
             val_acc_curve = metrics.get('val_accuracy_curve', []) # Check if curves were saved
             val_loss_curve = metrics.get('val_loss_curve', [])
             fallback_parts = []
             if val_acc_curve: fallback_parts.append(f"Final Acc={val_acc_curve[-1]:.4f}")
             if val_loss_curve: fallback_parts.append(f"Final Loss={val_loss_curve[-1]:.4f}")
             if fallback_parts:
                 details.append("  " + " | ".join(fallback_parts) + " (No 'best' metric recorded)")
             else:
                 details.append("  Metrics recorded, but best/final values not found.")
        else:
             details.append("  No best metrics recorded (possibly failed early).")


        # Duration line
        if duration is not None:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = ""
            if hours > 0: duration_str += f"{hours}h "
            if minutes > 0 or hours > 0: duration_str += f"{minutes}m " # Show minutes if hours > 0
            duration_str += f"{seconds}s"
            details.append(f"Duration: {duration_str.strip()}")
        else:
            details.append("Duration: N/A")
    else:
        details.append("Metrics: Not available.")

    # --- File Information ---
    if 'model_filename' in exp:
        details.append(f"Model Saved: {exp['model_filename']}")
    if 'metadata_filename' in exp:
         details.append(f"Metadata Saved: {exp['metadata_filename']}")

    details.append("--------------------------") # Separator
    return '\n'.join(details)


def run_experiment(exp_config):
    """
    Loads data, runs one experiment using train_model, saves results (including CM history path).
    """
    experiment_id = exp_config.get('experiment_id', 'unknown_id')
    data_params = exp_config.get('data_params', {})
    hyperparams = exp_config.get('hyperparams', {})

    # Validate essential params
    if 'num_samples' not in data_params or 'sample_length' not in data_params:
        print(f"❌ ERROR: 'num_samples' or 'sample_length' missing in data_params for {experiment_id}")
        # Optionally mark as failed? For now, just return None.
        return None # Indicate failure at data validation stage

    print(f"\n--- Starting Experiment: {experiment_id} ---")
    print(f"Data Params: {data_params}")
    print(f"Hyperparams: {hyperparams}")

    # Prepare data
    try:
        data = get_data(data_params) # Assuming get_data handles loading based on params
    except Exception as e:
         print(f"❌ ERROR: Failed to load data for {experiment_id}: {e}")
         # Return None to indicate failure before training even starts
         return None

    # Add experiment_id to hyperparams for internal use (e.g., checkpoint naming)
    hyperparams['experiment_id'] = experiment_id

    # --- Run Training ---
    try:
        # <<< CHANGE: Unpack the 4 return values from train_model >>>
        model, final_metrics, model_metadata, cm_history_path = train_model(data, hyperparams)

        if model is None: # Handle potential failure within train_model
            print(f"❌ ERROR: train_model returned None for {experiment_id}. Skipping save.")
            # Note: train_model failure might indicate NaN loss, etc.
            # Consider if a specific 'failed' status should be logged here.
            return None # Indicate failure during training/model return

    except Exception as e:
         print(f"❌ ERROR: Exception during training call for {experiment_id}: {e}")
         import traceback
         traceback.print_exc()
         # Indicate failure if exception occurs during the train_model call itself
         return None

    print(f"✅ Experiment {experiment_id} training phase completed.")

    # --- Process and Store Results ---
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    completed_exp_data = exp_config.copy() # Start with the original config
    completed_exp_data['run_timestamp'] = run_timestamp
    completed_exp_data['uid'] = experiment_id # Use experiment_id as uid (deprecating uid later?)

    # Ensure metrics are JSON serializable (handles numpy arrays etc. in curves)
    serializable_metrics = convert_ndarray_to_list(final_metrics)
    completed_exp_data['metrics'] = serializable_metrics

    # Save the final trained model (state_dict)
    model_filename = f'data/models/{experiment_id}.pt'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    try:
        # Save the model state_dict for better compatibility
        if isinstance(model, torch.nn.DataParallel):
             torch.save(model.module.state_dict(), model_filename)
        else:
             torch.save(model.state_dict(), model_filename)
        completed_exp_data['model_filename'] = model_filename
        print(f"   Model state_dict saved to: {model_filename}")
    except Exception as e:
         print(f"❌ ERROR: Failed to save model for {experiment_id}: {e}")
         # Consider returning None or marking as partial success if model save fails?
         # For now, continue to save metadata/metrics if available.


    # Save model metadata (token dict, label encoder, etc.)
    metadata_filename = f'data/models/{experiment_id}_metadata.json' # Use JSON for metadata
    os.makedirs(os.path.dirname(metadata_filename), exist_ok=True)
    try:
        # Ensure metadata is serializable (should be, but good practice)
        serializable_metadata = convert_ndarray_to_list(model_metadata)
        with open(metadata_filename, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        completed_exp_data['metadata_filename'] = metadata_filename
        print(f"   Metadata saved to: {metadata_filename}")
    except Exception as e:
         print(f"❌ ERROR: Failed to save metadata for {experiment_id}: {e}")

    # <<< CHANGE: Add the cm_history filename path to the record >>>
    # cm_history_path will be None if saving failed or history was empty
    completed_exp_data['cm_history_filename'] = cm_history_path
    if cm_history_path:
         print(f"   CM History path recorded: {cm_history_path}")
    else:
         print(f"   No CM History path recorded (save failed or history empty).")

    return completed_exp_data # Return the enriched experiment data


def append_to_experiment_file(file_path, experiment_data):
    """Safely appends a completed experiment to the JSON log file."""
    if experiment_data is None: # Don't append failed experiments
        return
    # Convert any remaining non-serializable types (just in case)
    serializable_experiment = convert_ndarray_to_list(experiment_data)

    # Read existing data, append, and write back
    existing_data = safe_json_load(file_path)
    existing_data.append(serializable_experiment)
    try:
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2) # Use indent for readability
    except Exception as e:
         print(f"❌ ERROR: Failed to write updated experiments to {file_path}: {e}")


def rewrite_experiment_file(file_path, experiments):
    """Writes a list of experiments to a JSON file, overwriting existing content."""
    serializable_experiments = convert_ndarray_to_list(experiments)
    try:
        with open(file_path, 'w') as f:
            json.dump(serializable_experiments, f, indent=2)
    except Exception as e:
         print(f"❌ ERROR: Failed to rewrite experiment file {file_path}: {e}")


def get_experiment_keys(file_path):
    """Loads experiments and returns a set of their unique keys."""
    experiments = safe_json_load(file_path)
    # Use the unique experiment_id directly as the key now
    return {exp.get('experiment_id') for exp in experiments if exp.get('experiment_id')}


def get_pending_experiments(pending_file='data/pending_experiments.json'):
    """Loads pending experiments from the queue file."""
    return safe_json_load(pending_file)


def argument_parser():
    """Defines command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments from the queue.")
    parser.add_argument('--immediate', action='store_true',
                      help="Run a single experiment immediately, bypassing the queue. "
                           "Requires specifying all parameters.")
    # Add arguments for immediate mode (optional, could read from a config)
    parser.add_argument('--ciphers', nargs='*', default=None)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--sample_length', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--nhead', type=int, default=None)
    parser.add_argument('--num_encoder_layers', type=int, default=None)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--dropout_rate', type=float, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None, help="Early stopping patience")


    return parser.parse_args()


def create_immediate_experiment(args):
    """Creates a single experiment config from command-line args for immediate mode."""
    # Check required args for immediate mode
    required = ['num_samples', 'sample_length', 'd_model', 'nhead',
                'num_encoder_layers', 'dim_feedforward', 'batch_size',
                'dropout_rate', 'learning_rate']
    if any(getattr(args, req) is None for req in required):
        print("ERROR: For --immediate mode, all hyperparameter arguments must be provided.")
        sys.exit(1)

    # Use 'all' ciphers if not specified
    ciphers_list = args.ciphers if args.ciphers else _get_cipher_names()

    data_params = {
        'ciphers': [ciphers_list], # Ensure it's nested list
        'num_samples': args.num_samples,
        'sample_length': args.sample_length
    }

    hyperparams = {
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_encoder_layers': args.num_encoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'batch_size': args.batch_size,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        # Add optional patience if provided
        'early_stopping_patience': args.patience if args.patience is not None else default_params['early_stopping_patience'][0]
    }

    # Generate a unique ID for immediate run
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    experiment = {
        'experiment_id': f'immediate_{timestamp}',
        'data_params': data_params,
        'hyperparams': hyperparams,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return experiment


def main():
    global should_continue
    args = argument_parser()

    # Setup environment info
    print(f"Using PyTorch Version: {torch.__version__}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Running on Device: {device_name}")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Immediate Mode ---
    if args.immediate:
        print("--- Running in Immediate Mode ---")
        experiment_config = create_immediate_experiment(args)
        completed_exp = run_experiment(experiment_config)
        if completed_exp:
            print(get_experiment_details(completed_exp))
            append_to_experiment_file('data/completed_experiments.json', completed_exp)
            print("Immediate experiment saved to completed log.")
        else:
             print("Immediate experiment failed to complete.")
        return # Exit after immediate run

    # --- Queue Processing Mode ---
    print("\n--- Running in Queue Processing Mode ---")
    pending_file = 'data/pending_experiments.json'
    completed_file = 'data/completed_experiments.json'

    pending_experiments = get_pending_experiments(pending_file)
    completed_keys = get_experiment_keys(completed_file)

    if not pending_experiments:
        print("\n📋 Experiment queue is empty.")
        print("Use manage_queue.py to add experiments.")
        return

    print(f"Found {len(pending_experiments)} experiments in the queue.")
    original_pending_count = len(pending_experiments)
    processed_count = 0

    # Process queue sequentially
    while pending_experiments and should_continue:
        exp_config = pending_experiments.pop(0) # Get the next experiment
        experiment_id = exp_config.get('experiment_id', 'MISSING_ID')

        # Check if already completed
        if experiment_id in completed_keys:
            print(f"Skipping {experiment_id}: Already found in completed experiments.")
            # Update the pending file immediately to reflect removal
            rewrite_experiment_file(pending_file, pending_experiments)
            continue

        # Run the experiment
        completed_exp_data = run_experiment(exp_config)
        processed_count += 1

        if completed_exp_data:
            # Add to completed log and update completed keys
            append_to_experiment_file(completed_file, completed_exp_data)
            completed_keys.add(experiment_id)

            # Print summary and notify
            details = get_experiment_details(completed_exp_data)
            print(details)
            try:
                 notifications.send_discord_notification(f"✅ Experiment Completed: {experiment_id}\n" + details)
            except Exception as e:
                 print(f"Warning: Failed to send Discord notification: {e}")

        else:
            # Handle failed experiment (optional: move to a failed queue?)
            print(f"❌ Experiment {experiment_id} failed or returned no data.")
            try:
                notifications.send_discord_notification(f"❌ Experiment Failed: {experiment_id}")
            except Exception as e:
                 print(f"Warning: Failed to send Discord notification: {e}")
            # Keep it removed from pending queue for now

        # Update the pending experiments file after each run (or potential skip)
        rewrite_experiment_file(pending_file, pending_experiments)
        remaining_count = len(pending_experiments)
        print(f"Queue status: {remaining_count} remaining.")

        # Estimate remaining time (optional, based on recent durations)
        # (Consider adding the duration calculation logic back here if desired)

    # --- Post-Queue Processing ---
    if not should_continue:
        print("\nQueue processing interrupted by signal.")
    elif not pending_experiments:
        print("\n🎉 Experiment queue processing complete.")
        try:
            notifications.send_discord_notification("🎉 All pending experiments have been processed!")
        except Exception as e:
            print(f"Warning: Failed to send final Discord notification: {e}")
    else:
        # This case shouldn't normally be reached if loop completes
        print("\nQueue processing finished with items remaining (unexpected).")

    # Final cleanup (optional, maybe run separately?)
    # print("\nRunning final file cleanup...")
    # clean_up_files(test_mode=False)

    print("--- Researcher finished ---")


if __name__ == "__main__":
    main()
