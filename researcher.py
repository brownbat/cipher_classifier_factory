# researcher.py
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
import math # For isnan/isinf checks if needed

# Import from project structure
# Assuming models.__init__ exposes train_model and get_data
try:
    # Make sure your models/__init__.py has:
    # from .transformer.train import train_model
    # from .common.data import get_data
    from models import train_model, get_data
    from models.common.utils import safe_json_load, convert_ndarray_to_list
    from models.transformer.train import clean_old_checkpoints # Import cleanup for potential error handling
    from ciphers import _get_cipher_names
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Ensure researcher.py is run from the project root and models are structured correctly.")
    sys.exit(1)

# --- Global Flag for Signal Handling ---
should_continue = True

# --- Signal Handling ---
def signal_handler(sig, frame):
    """Handle Ctrl+C or kill signals gracefully by setting a flag."""
    global should_continue
    if not should_continue: # If already trying to exit, ignore subsequent signals
        print("\nShutdown already in progress. Please wait or force quit (Ctrl+\\).")
        # Optionally force exit if pressed multiple times:
        # sys.exit(1)
        return
    print('\nSignal received. Finishing current step/epoch and shutting down gracefully...')
    print('Latest checkpoint will be saved by train.py if applicable.')
    print('Run `python researcher.py` again later to resume queue processing.')
    should_continue = False
    # --- REMOVED sys.exit(0) --- It's now cooperative

# --- Core Functions ---

def get_experiment_details(exp):
    """
    Returns experiment details as a multi-line formatted string,
    focusing on key parameters and best achieved metrics.
    (Function content remains the same as before)
    """
    details = []
    details.append("--- Experiment Summary ---")
    exp_id = exp.get('experiment_id', 'N/A')
    details.append(f"ID: {exp_id}")

    # --- Data Parameters ---
    data_params = exp.get('data_params', {})
    num_samples = data_params.get('num_samples', 'N/A')
    sample_length = data_params.get('sample_length', 'N/A')

    ciphers_list_of_lists = data_params.get('ciphers', [])
    actual_ciphers = []
    if ciphers_list_of_lists and isinstance(ciphers_list_of_lists, list) and len(ciphers_list_of_lists) > 0:
        if isinstance(ciphers_list_of_lists[0], list):
             actual_ciphers = ciphers_list_of_lists[0]
        elif isinstance(ciphers_list_of_lists[0], str):
             actual_ciphers = ciphers_list_of_lists
    ciphers_used_str = ', '.join(actual_ciphers) if actual_ciphers else 'N/A'
    details.append(f"Data: {num_samples} samples, length {sample_length}, ciphers: {ciphers_used_str}")

    # Hyperparameters
    hyperparams = exp.get('hyperparams', {})
    batch_size = hyperparams.get('batch_size', 'N/A')
    dropout = hyperparams.get('dropout_rate', 'N/A')
    d_model = hyperparams.get('d_model', 'N/A')
    nhead = hyperparams.get('nhead', 'N/A')
    layers = hyperparams.get('num_encoder_layers', 'N/A')
    ff_dim = hyperparams.get('dim_feedforward', 'N/A')
    base_patience = hyperparams.get('base_patience', '?')
    details.append(f"Model: d={d_model}, h={nhead}, lyr={layers}, ff={ff_dim}, drop={dropout}")

    # Training details
    metadata = exp.get('model_metadata', {})
    initial_lr_str = "N/A"
    scheduler_str = "ReduceLROnPlateau (Default Params)"
    es_patience_str = "?"
    lr_patience_str = "?"

    if metadata:
        initial_lr = metadata.get('initial_learning_rate')
        if initial_lr is not None: initial_lr_str = f"{initial_lr:.2e}" # Reads the actual initial LR used

        es_patience_meta = metadata.get('early_stopping_patience')
        lr_patience_meta = metadata.get('lr_scheduler_patience')
        if es_patience_meta is not None: es_patience_str = str(es_patience_meta)
        if lr_patience_meta is not None: lr_patience_str = str(lr_patience_meta)

        sched_type = metadata.get('lr_scheduler_type')
        if sched_type == 'ReduceLROnPlateau':
             factor = metadata.get('lr_scheduler_factor', '?')
             sched_patience_disp = lr_patience_meta if lr_patience_meta is not None else '?'
             min_lr = metadata.get('lr_scheduler_min_lr', '?')
             monitor = metadata.get('lr_scheduler_monitor', '?')
             min_lr_disp = f"{min_lr:.1e}" if isinstance(min_lr, float) else str(min_lr)
             scheduler_str = f"ReduceLROnPlateau (f={factor}, p={sched_patience_disp}, min={min_lr_disp}, m='{monitor}')"
        elif sched_type: scheduler_str = f"Scheduler: {sched_type}"

    details.append(f"Training: BS={batch_size}, Initial LR={initial_lr_str} (Auto), Base Patience={base_patience} -> (LR={lr_patience_str}, ES={es_patience_str})")
    details.append(f"Scheduler: {scheduler_str}")

    # Metrics
    metrics = exp.get('metrics', {})
    status_from_metadata = exp.get('status', None) # Get status if saved separately
    if metrics:
        stopped_early = metrics.get('stopped_early', False)
        epochs_completed = metrics.get('epochs_completed', 'N/A')
        best_epoch = metrics.get('best_epoch', None)
        best_acc = metrics.get('best_val_accuracy', None)
        best_loss = metrics.get('best_val_loss', None)
        duration = metrics.get('training_duration', None)

        # Determine status string
        status = f"Status: Completed {epochs_completed} epochs"
        if status_from_metadata == 'interrupted': # Check for explicit interrupt status
             status = "Status: Interrupted"
        elif status_from_metadata == 'failed':
             status = "Status: Failed (NaN/Inf)"
        elif stopped_early:
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
        elif status_from_metadata not in ['failed', 'interrupted'] and epochs_completed != 'N/A' and epochs_completed > 0 :
             val_acc_curve = metrics.get('val_accuracy_curve', [])
             val_loss_curve = metrics.get('val_loss_curve', [])
             fallback_parts = []
             if val_acc_curve: fallback_parts.append(f"Final Acc={val_acc_curve[-1]:.4f}")
             if val_loss_curve: fallback_parts.append(f"Final Loss={val_loss_curve[-1]:.4f}")
             if fallback_parts:
                 details.append("  " + " | ".join(fallback_parts) + " (No 'best' metric recorded)")
             else:
                 details.append("  Metrics recorded, but best/final values not found.")
        elif status_from_metadata not in ['failed', 'interrupted']:
             details.append("  No metrics recorded (possibly failed early).")

        # Duration line
        if duration is not None:
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            duration_str = ""
            if hours > 0: duration_str += f"{hours}h "
            if minutes > 0 or hours > 0: duration_str += f"{minutes}m "
            duration_str += f"{seconds}s"
            details.append(f"Duration: {duration_str.strip()}")
        else:
            details.append("Duration: N/A")
    elif status_from_metadata: # If metrics are missing but we have a status
         details.append(f"Status: {status_from_metadata.capitalize()}")
         details.append("Metrics: Not available.")
    else: # No metrics and no status
         details.append("Metrics: Not available.")

    # --- File Information ---
    if 'model_filename' in exp and exp['model_filename']:
        details.append(f"Model Saved: {exp['model_filename']}")
    else:
        details.append(f"Model Saved: No (Failed/Interrupted?)") # Indicate if missing
    if 'metadata_filename' in exp and exp['metadata_filename']:
         details.append(f"Metadata Saved: {exp['metadata_filename']}")
    if 'cm_history_filename' in exp and exp['cm_history_filename']:
         details.append(f"CM History Saved: {exp['cm_history_filename']}")

    details.append("--------------------------") # Separator
    return '\n'.join(details)


# --- MODIFIED run_experiment signature ---
def run_experiment(exp_config, check_continue_func):
    """
    Loads data, runs one experiment using train_model, saves results.
    Checks check_continue_func for interruption signals.
    """
    experiment_id = exp_config.get('experiment_id', 'unknown_id')
    data_params = exp_config.get('data_params', {})
    hyperparams = exp_config.get('hyperparams', {})

    # Validate essential params
    if 'num_samples' not in data_params or 'sample_length' not in data_params:
        print(f"❌ ERROR: 'num_samples' or 'sample_length' missing in data_params for {experiment_id}")
        return None # Indicate failure

    print(f"\n--- Starting Experiment: {experiment_id} ---")
    print(f"Data Params: {data_params}")
    print(f"Hyperparams: {hyperparams}")

    # Prepare data
    try:
        data = get_data(data_params)
        if data is None or data.empty:
             print(f"❌ ERROR: Failed to load data or data is empty for {experiment_id}")
             return None
    except Exception as e:
         print(f"❌ ERROR: Exception during data loading for {experiment_id}: {e}")
         return None

    # Add experiment_id to hyperparams for internal use
    hyperparams['experiment_id'] = experiment_id

    # --- Run Training ---
    model = None
    final_metrics = None
    model_metadata = None
    cm_history_path = None
    training_outcome_status = "unknown" # Track outcome

    try:
        # --- Pass the checker function down to train_model ---
        model, final_metrics, model_metadata, cm_history_path = train_model(
            data, hyperparams, check_continue_func
        )

        # Check return values to determine outcome
        if model is None or final_metrics is None or model_metadata is None:
            # Check if interrupted (train_model might update metadata even if interrupted)
            # This relies on train_model setting a status in metadata on interruption/failure
            # If train_model just returns None on interrupt/fail, we infer here
            # Let's assume train_model *doesn't* modify metadata on fail/interrupt and returns None
            print(f"   train_model returned None/incomplete results for {experiment_id}. Assuming Failure or Interruption.")
            # We can't easily distinguish failure vs interruption here without more info from train_model
            # Let's set a generic 'failed_or_interrupted' status for the summary
            training_outcome_status = 'failed_or_interrupted'
            # Return None to signify no complete results to save/process further
            return {'experiment_id': experiment_id, 'status': training_outcome_status} # Return minimal info

        else:
            # Training completed successfully or stopped early
            print(f"✅ Experiment {experiment_id} training phase completed successfully.")
            training_outcome_status = 'completed' # Or 'stopped_early' based on metrics
            if final_metrics.get('stopped_early', False):
                training_outcome_status = 'stopped_early'

    except Exception as e:
         print(f"❌ ERROR: Exception during train_model call for {experiment_id}: {e}")
         import traceback
         traceback.print_exc()
         # Attempt cleanup if possible (though train_model might handle its own)
         try:
             print("   Attempting checkpoint cleanup after exception in researcher...")
             clean_old_checkpoints(experiment_id, completed=True)
         except Exception as cleanup_e:
             print(f"   WARNING: Could not perform checkpoint cleanup after exception: {cleanup_e}")
         training_outcome_status = 'crashed'
         return {'experiment_id': experiment_id, 'status': training_outcome_status} # Return minimal info


    # --- Process and Store Results (only if training didn't fail/crash) ---
    # This block is only reached if train_model returned valid results

    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    completed_exp_data = exp_config.copy() # Start with the original config
    completed_exp_data['run_timestamp'] = run_timestamp
    completed_exp_data['uid'] = experiment_id # Keep UID for potential compatibility
    completed_exp_data['status'] = training_outcome_status # Add the outcome status

    # Ensure metrics are JSON serializable
    serializable_metrics = convert_ndarray_to_list(final_metrics)
    completed_exp_data['metrics'] = serializable_metrics

    # Save the final trained model (state_dict)
    model_filename = f'data/models/{experiment_id}.pt'
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    try:
        if isinstance(model, torch.nn.DataParallel):
             torch.save(model.module.state_dict(), model_filename)
        else:
             torch.save(model.state_dict(), model_filename)
        completed_exp_data['model_filename'] = model_filename
        print(f"   Model state_dict saved to: {model_filename}")
    except Exception as e:
         print(f"❌ ERROR: Failed to save model for {experiment_id}: {e}")
         completed_exp_data['model_filename'] = None # Mark as failed
         completed_exp_data['status'] = 'completed_save_failed' # Update status

    # Save model metadata
    metadata_filename = f'data/models/{experiment_id}_metadata.json'
    os.makedirs(os.path.dirname(metadata_filename), exist_ok=True)
    try:
        serializable_metadata = convert_ndarray_to_list(model_metadata)
        # Add status to metadata as well for self-contained record
        serializable_metadata['status'] = training_outcome_status
        with open(metadata_filename, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        completed_exp_data['metadata_filename'] = metadata_filename
        # Include the metadata directly in the completed exp record
        completed_exp_data['model_metadata'] = serializable_metadata
        print(f"   Metadata saved to: {metadata_filename}")
    except Exception as e:
         print(f"❌ ERROR: Failed to save metadata for {experiment_id}: {e}")
         completed_exp_data['metadata_filename'] = None # Mark as failed
         if completed_exp_data['status'] not in ['failed_or_interrupted', 'crashed']: # Don't overwrite failure status
            completed_exp_data['status'] = 'completed_save_failed'

    # Add the cm_history filename path
    completed_exp_data['cm_history_filename'] = cm_history_path
    if cm_history_path:
         print(f"   CM History path recorded: {cm_history_path}")
    else:
         print(f"   No CM History path recorded (or saving failed).")

    return completed_exp_data # Return the enriched experiment data


def append_to_experiment_file(file_path, experiment_data):
    """Safely appends a completed or partially completed experiment to the JSON log file."""
    if experiment_data is None: # Should not happen with new return logic, but safe check
        print("Warning: append_to_experiment_file called with None data.")
        return

    # Ensure the basic dict is serializable
    try:
        serializable_experiment = convert_ndarray_to_list(experiment_data)
    except Exception as e:
        print(f"❌ ERROR: Could not make experiment data serializable before appending: {e}")
        # Fallback: try to save just the ID and status if possible
        exp_id = experiment_data.get('experiment_id', 'unknown_serialization_error')
        status = experiment_data.get('status', 'serialization_error')
        serializable_experiment = {'experiment_id': exp_id, 'status': status, 'error': 'Serialization failed'}

    # Read existing data, append, and write back
    existing_data = safe_json_load(file_path)
    # Avoid duplicates if somehow run twice (e.g. after crash/resume)
    existing_ids = {exp.get('experiment_id') for exp in existing_data if exp.get('experiment_id')}
    if serializable_experiment.get('experiment_id') not in existing_ids:
        existing_data.append(serializable_experiment)
        try:
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
             print(f"❌ ERROR: Failed to write updated experiments to {file_path}: {e}")
    else:
        print(f"   Skipping append for {serializable_experiment.get('experiment_id')}: Already in {file_path}.")


def rewrite_experiment_file(file_path, experiments):
    """Writes a list of experiments to a JSON file, overwriting existing content."""
    try:
        serializable_experiments = convert_ndarray_to_list(experiments)
    except Exception as e:
        print(f"❌ ERROR: Could not make pending experiments serializable before writing: {e}")
        # Avoid writing corrupted data if serialization fails
        return

    try:
        with open(file_path, 'w') as f:
            json.dump(serializable_experiments, f, indent=2)
    except Exception as e:
         print(f"❌ ERROR: Failed to rewrite experiment file {file_path}: {e}")


def get_experiment_keys(file_path):
    """Loads experiments and returns a set of their unique experiment_ids."""
    experiments = safe_json_load(file_path)
    return {exp.get('experiment_id') for exp in experiments if exp.get('experiment_id')}


def get_pending_experiments(pending_file='data/pending_experiments.json'):
    """Loads pending experiments from the queue file."""
    return safe_json_load(pending_file)


def argument_parser():
    """Defines command-line arguments."""
    parser = argparse.ArgumentParser(description="Run experiments from the queue or immediately.")
    parser.add_argument('--immediate', action='store_true',
                      help="Run a single experiment immediately, bypassing the queue. "
                           "Requires specifying necessary parameters.")
    # Add arguments for immediate mode (optional, could read from a config)
    # Learning rate removed
    parser.add_argument('--ciphers', nargs='*', default=None, help="List of ciphers (e.g., Vigenere Beaufort)")
    parser.add_argument('--num_samples', type=int, default=None, help="Number of text samples")
    parser.add_argument('--sample_length', type=int, default=None, help="Length of each text sample")
    parser.add_argument('--d_model', type=int, default=None, help="Model dimension")
    parser.add_argument('--nhead', type=int, default=None, help="Number of attention heads")
    parser.add_argument('--num_encoder_layers', type=int, default=None, help="Number of encoder layers")
    parser.add_argument('--dim_feedforward', type=int, default=None, help="Dimension of feedforward layers")
    parser.add_argument('--batch_size', type=int, default=None, help="Training batch size")
    parser.add_argument('--dropout_rate', type=float, default=None, help="Dropout rate")
    # Renamed --patience to --base_patience for clarity
    parser.add_argument('--base_patience', type=int, default=None, help="Base patience for LR scheduler and Early Stopping")

    return parser.parse_args()


def create_immediate_experiment(args):
    """Creates a single experiment config from command-line args for immediate mode."""
    # Define defaults here for immediate mode if not provided
    default_base_patience = 5 # Example default

    # Check required model architecture args
    required_model = ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 'dropout_rate']
    if any(getattr(args, req) is None for req in required_model):
        print("ERROR: For --immediate mode, all required model hyperparameter arguments must be provided:")
        print(f"Required: {', '.join(required_model)}")
        sys.exit(1)

    # Check required data args
    required_data = ['num_samples', 'sample_length']
    if any(getattr(args, req) is None for req in required_data):
        print("ERROR: For --immediate mode, all required data arguments must be provided:")
        print(f"Required: {', '.join(required_data)}")
        sys.exit(1)

    # Check required training args
    required_training = ['batch_size'] # base_patience is optional below
    if any(getattr(args, req) is None for req in required_training):
        print("ERROR: For --immediate mode, all required training arguments must be provided:")
        print(f"Required: {', '.join(required_training)}")
        sys.exit(1)

    # Use 'all' ciphers if not specified
    ciphers_list = args.ciphers if args.ciphers else _get_cipher_names()

    data_params = {
        'ciphers': [ciphers_list], # Expects list of lists? Double check get_data needs
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
        # Add base_patience if provided, otherwise use default
        'base_patience': args.base_patience if args.base_patience is not None else default_base_patience
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
    project_root = os.path.dirname(os.path.abspath(__file__)) # Assuming researcher is at root
    print(f"Project Root detected: {project_root}")
    data_dir = os.path.join(project_root, 'data')
    print(f"Data Directory detected: {data_dir}") # Used for file paths below

    print(f"Using PyTorch Version: {torch.__version__}")
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    print(f"Running on Device: {device_name}")
    if gpu_available:
        print(f"CUDA Devices: {torch.cuda.device_count()}")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Define file paths relative to data_dir
    pending_file = os.path.join(data_dir, 'pending_experiments.json')
    completed_file = os.path.join(data_dir, 'completed_experiments.json')

    # Define a simple function to check the global flag - passed down to train_model
    def check_continue_flag():
        return should_continue

    # --- Immediate Mode ---
    if args.immediate:
        if not should_continue: # Check flag even before starting immediate
             print("Shutdown signal detected before starting immediate run.")
             return

        print("--- Running in Immediate Mode ---")
        experiment_config = create_immediate_experiment(args)

        # Run the single experiment, passing the checker function
        completed_exp = run_experiment(experiment_config, check_continue_flag)

        if completed_exp:
            print("\n--- Immediate Experiment Results ---")
            # Use get_experiment_details for summary
            # Ensure metadata is loaded if needed for display (run_experiment should return it)
            details = get_experiment_details(completed_exp)
            print(details)
            # Append result to completed log
            append_to_experiment_file(completed_file, completed_exp)
            print("Immediate experiment result saved to completed log.")
        else:
             # run_experiment returns minimal dict on failure/interrupt
             status = completed_exp.get('status', 'failed_unknown') if completed_exp else 'failed_unknown'
             print(f"Immediate experiment did not complete successfully (Status: {status}). Check logs.")
             # Still append minimal failure info
             if completed_exp:
                 append_to_experiment_file(completed_file, completed_exp)


        return # Exit after immediate run

    # --- Queue Processing Mode ---
    print("\n--- Running in Queue Processing Mode ---")

    pending_experiments = get_pending_experiments(pending_file)
    completed_keys = get_experiment_keys(completed_file)

    if not pending_experiments:
        print("\n📋 Experiment queue is empty.")
        print("Use manage_queue.py or suggest_experiments.py to add experiments.")
        return

    print(f"Found {len(pending_experiments)} experiments in the queue.")
    original_pending_count = len(pending_experiments)
    processed_count = 0

    # Process queue sequentially
    # Check should_continue in the loop condition
    while pending_experiments and should_continue:
        # Check flag *before* popping, in case signal received while idle
        if not should_continue:
             print("Shutdown signal detected before processing next experiment.")
             break

        exp_config = pending_experiments.pop(0) # Get the next experiment
        experiment_id = exp_config.get('experiment_id', 'MISSING_ID')

        # Handle missing ID case
        if experiment_id == 'MISSING_ID':
             print("ERROR: Found experiment config with missing ID in queue. Skipping.")
             # Save the queue state immediately after removing broken item
             rewrite_experiment_file(pending_file, pending_experiments)
             continue # Skip to next iteration

        # Check if already completed
        if experiment_id in completed_keys:
            print(f"Skipping {experiment_id}: Already found in completed experiments.")
            # Update the pending file immediately to reflect removal
            rewrite_experiment_file(pending_file, pending_experiments)
            continue

        # Check flag *again* right before running the (potentially long) experiment
        if not should_continue:
            print(f"Shutdown signal detected before running experiment {experiment_id}.")
            # Put the experiment back at the front of the queue
            pending_experiments.insert(0, exp_config)
            rewrite_experiment_file(pending_file, pending_experiments) # Save queue state
            break # Exit the while loop

        # --- Run the experiment ---
        # Pass the checker function
        completed_exp_data = run_experiment(exp_config, check_continue_flag)
        processed_count += 1

        # --- Process results ---
        # run_experiment now returns *something* even on failure/interrupt
        if completed_exp_data:
            exp_status = completed_exp_data.get('status', 'unknown')
            exp_id = completed_exp_data.get('experiment_id', 'unknown') # Get ID again

            # Append result (minimal or full) to completed log
            append_to_experiment_file(completed_file, completed_exp_data)
            completed_keys.add(exp_id) # Add ID to completed set

            # Display summary and notify based on status
            print("\n--- Experiment Results ---")
            details = get_experiment_details(completed_exp_data)
            print(details)

            notification_prefix = "❓" # Default
            if exp_status in ['completed', 'stopped_early']:
                notification_prefix = "✅"
            elif exp_status in ['failed_or_interrupted', 'crashed', 'completed_save_failed']:
                notification_prefix = "❌"
            elif exp_status == 'failed_unknown':
                 notification_prefix = "❓" # Truly unknown failure

            try:
                 # Use prefix and status in notification
                 notification_message = f"{notification_prefix} Experiment {exp_id} finished with status: {exp_status}\n" + details
                 notifications.send_discord_notification(notification_message)
            except Exception as e:
                 print(f"Warning: Failed to send Discord notification: {e}")

        else:
            # This case should ideally not be reached if run_experiment always returns a dict
            print(f"❌ Experiment {experiment_id} run returned None unexpectedly.")
            # Record a failure manually
            failure_data = {'experiment_id': experiment_id, 'status': 'runner_error'}
            append_to_experiment_file(completed_file, failure_data)
            completed_keys.add(experiment_id)
            try:
                notifications.send_discord_notification(f"❓ Experiment Runner Error: {experiment_id}")
            except Exception as e:
                 print(f"Warning: Failed to send Discord notification: {e}")


        # Update the pending experiments file *after* processing each item
        # This ensures removal even if interrupted later
        rewrite_experiment_file(pending_file, pending_experiments)
        remaining_count = len(pending_experiments)
        print(f"Queue status: {remaining_count} remaining.")

        # Optional: Add time estimation logic here if desired

    # --- Post-Queue Processing ---
    if not should_continue:
        print("\nQueue processing interrupted by signal.")
        # The current queue state (with potentially un-run items) should already be saved.
    elif not pending_experiments:
        print("\n🎉 Experiment queue processing complete.")
        try:
            notifications.send_discord_notification("🎉 All pending experiments have been processed!")
        except Exception as e:
            print(f"Warning: Failed to send final Discord notification: {e}")
    else:
        # This case might happen if the loop condition fails mid-iteration after should_continue becomes False
        print("\nQueue processing finished with items remaining (likely due to signal timing).")

    print("--- Researcher finished ---")


if __name__ == "__main__":
    main()
