# researcher.py
import datetime
import itertools
import json
import notifications
import os
import torch
import torch.nn as nn
import numpy as np
import signal
import sys
import time
import argparse
import math # For isnan/isinf checks if needed
import multiprocessing as mp # Added for Event

# Import from project structure
# Assuming models.__init__ exposes train_model and get_data
try:
    # Make sure your models/__init__.py has:
    # from .transformer.train import train_model
    # from .common.data import get_data
    from models import train_model, get_data
    from models.common.utils import safe_json_load, convert_ndarray_to_list
    from models.transformer.train import clean_old_checkpoints # Import cleanup for potential error handling
    from ciphers import _get_cipher_names # Assuming this is used later
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Ensure researcher.py is run from the project root and models are structured correctly.")
    sys.exit(1)

# --- Constants ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EMA_ALPHA = 0.3

# --- Global Flags/Events for Signal Handling ---
should_continue = True          # General flag checked by loops (e.g., training, main queue)
shutdown_event = mp.Event()     # Multiprocessing event specifically for data generation workers

# Connect the shutdown_event to prep_samples
try:
    import prep_samples
    # Share our shutdown_event with prep_samples module
    prep_samples.shutdown_event = shutdown_event
except ImportError:
    print("Note: Could not import prep_samples - will connect event when needed")

# --- Signal Handling ---
def signal_handler(sig, frame):
    """Handle Ctrl+C or kill signals gracefully by setting global flags/events."""
    global should_continue
    if not should_continue: # If already trying to exit, ignore subsequent signals
        print("\nShutdown already in progress. Please wait or force quit (Ctrl+\\).")
        return

    print('\nSignal received. Setting flags for graceful shutdown...')
    print('Latest checkpoint will be saved by train.py if applicable.')
    print('Data generation (if active) will be requested to terminate.')
    print('Run `python researcher.py` again later to resume queue processing.')

    should_continue = False    # Signal main loop and training loop to stop
    shutdown_event.set()       # Signal multiprocessing workers (in prep_samples) to stop

def setup_signal_handlers():
    """Registers the signal handlers."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# --- Core Functions ---

def format_duration(seconds):
    """Formats seconds into a human-readable string (Xh Ym Zs)."""
    if seconds is None or not isinstance(seconds, (int, float)) or math.isnan(seconds) or math.isinf(seconds):
        return "N/A"
    if seconds < 0:
        return "N/A"

    seconds = int(seconds) # Convert to integer seconds

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or (hours > 0 and secs == 0): # Include minutes if hours are shown or if minutes > 0
        parts.append(f"{minutes}m")
    # Always include seconds if duration < 1 minute or if there are remaining seconds
    if hours == 0 and minutes == 0 or secs > 0:
        # Only show seconds if total duration is less than a minute OR if there's a non-zero second component
        if seconds < 60 or secs > 0:
            parts.append(f"{secs}s")

    if not parts: # Handle case of 0 seconds
        return "0s"

    return " ".join(parts)


def get_experiment_details(exp):
    """
    Returns a concise, multi-line summary of experiment details and results.
    Gracefully handles partial data from interrupted/failed runs by using metadata.
    """
    details = []
    exp_id = exp.get('experiment_id', 'N/A')

    # --- Extract Status and Metrics (might be partial) ---
    metrics = exp.get('metrics', {})
    status = exp.get('status', 'unknown') # Use top-level status

    stopped_early = metrics.get('stopped_early', False) # Safely get metrics
    epochs_completed = metrics.get('epochs_completed', '?')
    best_epoch = metrics.get('best_epoch', None)
    best_acc = metrics.get('best_val_accuracy', None)
    best_loss = metrics.get('best_val_loss', None) # Not currently in summary string
    duration_secs = metrics.get('training_duration', None) # Duration should often exist
    duration_str = format_duration(duration_secs)

    # --- Extract Config from Metadata (should always exist) ---
    metadata = exp.get('model_metadata', {})
    # Use metadata as the primary source for config details
    hyperparams_from_meta = metadata.get('hyperparams', {}) # Original hyperparams
    data_params_from_meta = hyperparams_from_meta.get('data_params', {})

    # Data Details
    num_samples = data_params_from_meta.get('num_samples', '?')
    sample_length = data_params_from_meta.get('sample_length', '?')
    ciphers_list = data_params_from_meta.get('ciphers', []) # This might be nested [[...]]
    actual_ciphers = []
    if ciphers_list and isinstance(ciphers_list, list) and len(ciphers_list) > 0:
        # Handle potential double nesting [[..]] vs [..]
        inner_list = ciphers_list[0] if isinstance(ciphers_list[0], list) else ciphers_list
        if isinstance(inner_list, list):
             actual_ciphers = [str(c) for c in inner_list] # Ensure strings
    num_ciphers = len(actual_ciphers) if actual_ciphers else '?'

    # Hyperparameters (prefer metadata fields if they exist, fallback to hyperparams_from_meta)
    d_model = metadata.get('d_model', hyperparams_from_meta.get('d_model', '?'))
    layers = metadata.get('num_encoder_layers', hyperparams_from_meta.get('num_encoder_layers', '?'))
    dropout = metadata.get('dropout_rate', hyperparams_from_meta.get('dropout_rate', '?'))
    batch_size = hyperparams_from_meta.get('batch_size', '?') # Batch size only in original hyperparams

    # Training Setup (from metadata)
    initial_lr = metadata.get('initial_learning_rate', None)
    lr_str = f"{initial_lr:.1e}" if initial_lr is not None else "?"
    es_patience = metadata.get('early_stopping_patience', '?')
    lr_patience = metadata.get('lr_scheduler_patience', '?')
    pat_str = f"ES={es_patience}/LR={lr_patience}"

    sched_type = metadata.get('lr_scheduler_type')
    sched_info_str = "?"
    if sched_type == 'ReduceLROnPlateau':
        factor = metadata.get('lr_scheduler_factor', '?')
        min_lr = metadata.get('lr_scheduler_min_lr', '?')
        factor_str = f"{factor:.1g}" if isinstance(factor, float) else str(factor)
        min_lr_str = f"{min_lr:.1e}" if isinstance(min_lr, float) else str(min_lr)
        sched_info_str = f"RLROP(f={factor_str}, min={min_lr_str})"
    elif sched_type:
        sched_info_str = sched_type

    # --- Build Summary Lines ---
    # Line 1: Core Outcome
    acc_str = f"{best_acc:.4f}" if best_acc is not None else "N/A"
    ep_str = str(best_epoch) if best_epoch is not None and best_epoch > 0 else "?"
    comp_ep_str = str(epochs_completed) if epochs_completed != '?' else "?"
    line1 = f"ID: {exp_id} | Best Acc: {acc_str} @ Ep {ep_str} | Duration: {duration_str} (Total Ep: {comp_ep_str})"
    details.append(line1)

    # Line 2: Optional Status
    # Refined list of non-success statuses
    if status not in ['completed', 'stopped_early', 'completed_safety_limit']:
        status_str = status.replace('_', ' ').upper()
        details.append(f"Status: {status_str}")

    # Line 3: Data
    line3 = f"Data: N={num_samples}, L={sample_length}, C={num_ciphers}"
    details.append(line3)

    # Line 4: Hyperparams & Training
    drop_str = f"{dropout:.2f}" if isinstance(dropout, float) else str(dropout)
    line4 = f"Params: d={d_model}, l={layers}, dr={drop_str}, bs={batch_size} | LR: {lr_str} (Auto) | Sch: {sched_info_str} | Pat: {pat_str}"
    details.append(line4)

    return '\n'.join(details)


# In researcher.py

def run_experiment(exp_config, check_continue_func):
    """
    Loads/generates data, runs one experiment using train_model, saves results.
    Checks check_continue_func for interruption signals during training.
    Data generation (via get_data -> prep_samples) uses the global shutdown_event.

    Args:
        exp_config (Dict): Configuration for the experiment.
        check_continue_func (callable): Function to check for external stop signals
                                       (primarily for the training loop).

    Returns:
        Dict: A dictionary containing the experiment results and metadata,
              including a specific 'status' field indicating outcome
              (e.g., completed, failed, interrupted, setup_failed_*, crashed).
    """
    experiment_id = exp_config.get('experiment_id', 'unknown_id')
    data_params = exp_config.get('data_params', {})
    hyperparams = exp_config.get('hyperparams', {}) # Base hyperparams
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Prepare Base Failure/Metadata Structure ---
    # Used for early returns on setup failures
    def create_failure_result(status, error_message="N/A"):
        return {
            'experiment_id': experiment_id,
            'status': status,
            'error_message': error_message,
            'run_timestamp': run_timestamp,
            # Include original config for context on failure
            'original_config': exp_config,
            # Provide minimal metadata structure even on failure
            'model_metadata': {
                'experiment_id': experiment_id,
                'hyperparams': hyperparams, # Base hyperparams
                'data_params': data_params, # Data params used
                'status': status # Embed status in metadata too
            },
            'metrics': {'status': status}, # Minimal metrics
            'model_filename': None,
            'metadata_filename': None,
            'cm_history_filename': None
        }

    # --- Validate essential params before data loading ---
    if 'num_samples' not in data_params or 'sample_length' not in data_params:
        err_msg = "Missing 'num_samples' or 'sample_length' in data_params"
        print(f"❌ ERROR [{experiment_id}]: {err_msg}")
        return create_failure_result('setup_failed_missing_params', err_msg)
    # Add any other crucial parameter validations here

    print(f"\n>>> Processing Experiment (ID: {experiment_id}) <<<")

    # --- Prepare Data ---
    data = get_data(data_params) # This now returns None on failure/interruption

    if data is None:
        # get_data returned None, indicating failure or interruption during data prep/gen.
        # The specific reason (error message / interrupted log) should have been printed by get_data/prep_samples.
        # We need to check the global shutdown event to differentiate interruption from other errors.
        if not check_continue_func() or shutdown_event.is_set():
            # If global signal flags are set, assume interruption caused the failure.
            err_msg = "Data preparation/generation was interrupted by signal."
            print(f"❌ INFO [{experiment_id}]: {err_msg}")
            return create_failure_result('setup_interrupted_data_generation', err_msg)
        else:
            # If signal flags aren't set, assume a data generation/loading error.
            err_msg = "Failed to load or generate data (get_data returned None). Check logs from get_data/prep_samples."
            print(f"❌ ERROR [{experiment_id}]: {err_msg}")
            return create_failure_result('setup_failed_data_generation', err_msg)

    # If data is not None, check if it's empty (get_data might return empty DF in some edge cases)
    if data.empty:
        err_msg = "Data loaded successfully, but the resulting DataFrame is empty."
        print(f"❌ ERROR [{experiment_id}]: {err_msg}")
        # Treat this as a data failure scenario
        return create_failure_result('setup_failed_data_empty', err_msg)


    # --- Prepare parameters for train_model ---
    # Create a combined dictionary to pass down, ensuring data details are saved in metadata later
    hyperparams_for_train = hyperparams.copy()
    hyperparams_for_train['experiment_id'] = experiment_id # Ensure ID is there
    hyperparams_for_train['data_params'] = data_params # Nest data_params inside
    # print(f"DEBUG [{experiment_id}]: Combined Hyperparams for train_model: {hyperparams_for_train}") # Optional Debug

    # --- Run Training ---
    model = None
    final_metrics = None
    model_metadata = None
    cm_history_path = None
    run_outcome_status = "unknown" # Track outcome more specifically

    try:
        # Pass the loaded data, combined hyperparams, and the researcher's continue check function
        model, final_metrics, model_metadata, cm_history_path = train_model(
            data, hyperparams_for_train, check_continue_func
        )

        # --- Determine Outcome based on returned metrics from train_model ---
        # train_model is expected to return metrics and metadata even on failure/interruption
        if final_metrics and isinstance(final_metrics, dict) and 'status' in final_metrics:
            run_outcome_status = final_metrics['status'] # Get specific status (e.g., completed, failed, interrupted)
        elif model_metadata and isinstance(model_metadata, dict) and 'status' in model_metadata:
            # Fallback if metrics somehow missing status but metadata has it
            run_outcome_status = model_metadata.get('status', 'unknown_train_error')
            print(f"WARNING [{experiment_id}]: train_model metrics missing status. Inferred '{run_outcome_status}' from metadata.")
        else:
             # Safety check if train_model returned unexpected/invalid values
             run_outcome_status = 'runner_train_call_error'
             print(f"ERROR [{experiment_id}]: train_model returned unexpected None/invalid values. Status set to '{run_outcome_status}'.")
             # Create minimal structures for reporting
             final_metrics = {'status': run_outcome_status}
             model_metadata = {'experiment_id': experiment_id, 'hyperparams': hyperparams_for_train, 'status': run_outcome_status}

    except Exception as e:
         # Catch unexpected errors during the train_model call itself
         run_outcome_status = 'crashed'
         error_message = f"Exception during train_model call: {e}"
         print(f"❌ CRITICAL ERROR [{experiment_id}]: {error_message}")
         traceback.print_exc()
         # Attempt checkpoint cleanup if possible after a crash
         try:
             print(f"INFO [{experiment_id}]: Attempting checkpoint cleanup after crash...")
             clean_old_checkpoints(experiment_id, completed=True) # Treat crash as completion for cleanup
         except Exception as cleanup_e:
             print(f"WARNING [{experiment_id}]: Could not perform checkpoint cleanup after crash: {cleanup_e}")

         # Create minimal structures for return after crash
         final_metrics = {'status': run_outcome_status, 'error_message': str(e)}
         model_metadata = {'experiment_id': experiment_id, 'hyperparams': hyperparams_for_train, 'status': run_outcome_status}

    # --- Process and Store Results ---
    # Start with the original config, add/overwrite with results
    completed_exp_data = exp_config.copy()
    completed_exp_data['run_timestamp'] = run_timestamp
    completed_exp_data['status'] = run_outcome_status # Set the specific status from training/crash

    # --- Ensure metrics and metadata are serializable and add them ---
    # Use the metrics dict returned by train_model (or created on crash)
    serializable_metrics = convert_ndarray_to_list(final_metrics) if final_metrics else {'status': run_outcome_status}
    completed_exp_data['metrics'] = serializable_metrics

    # Use the metadata dict returned by train_model (or created on crash)
    # train_model should have included hyperparams/data_params correctly
    serializable_metadata = convert_ndarray_to_list(model_metadata) if model_metadata else {
        'experiment_id': experiment_id, 'hyperparams': hyperparams_for_train, 'status': run_outcome_status
    }
    # Ensure status is present in final metadata saved
    if 'status' not in serializable_metadata:
        serializable_metadata['status'] = run_outcome_status
    completed_exp_data['model_metadata'] = serializable_metadata

    # Add cm_history path if generated
    completed_exp_data['cm_history_filename'] = cm_history_path

    # --- Save Model (only if successfully completed or stopped early) ---
    model_filename = None
    # Check for statuses that indicate a usable model might exist
    can_save_model = run_outcome_status in ['completed', 'stopped_early', 'completed_safety_limit']
    if can_save_model and model is not None:
        model_filename_relative = f'data/models/{experiment_id}.pt'
        model_filename_abs = os.path.join(PROJECT_ROOT, model_filename_relative)
        os.makedirs(os.path.dirname(model_filename_abs), exist_ok=True)
        try:
            # Handle potential DataParallel wrapping
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(model_to_save.state_dict(), model_filename_abs)
            model_filename = model_filename_relative # Store relative path for record
            print(f"Model saved to: {model_filename}")
        except Exception as e:
             print(f"❌ ERROR [{experiment_id}]: Failed to save model: {e}")
             completed_exp_data['status'] = 'completed_save_failed' # Update status if saving failed
             model_filename = None # Ensure filename is None if save failed
    elif can_save_model and model is None:
         # If status indicates success but model object is None (shouldn't normally happen)
         print(f"WARNING [{experiment_id}]: Model saving skipped for status '{run_outcome_status}' because model object is None.")
         completed_exp_data['status'] = 'completed_model_missing' # Update status

    completed_exp_data['model_filename'] = model_filename # Add filename (or None)

    # --- Save Metadata (Always save if possible, contains config/status info) ---
    metadata_filename = None
    metadata_filename_relative = f'data/models/{experiment_id}_metadata.json'
    metadata_filename_abs = os.path.join(PROJECT_ROOT, metadata_filename_relative)
    os.makedirs(os.path.dirname(metadata_filename_abs), exist_ok=True)
    try:
        # Ensure hyperparams (including data_params) are in metadata if somehow missing
        if 'hyperparams' not in serializable_metadata:
             serializable_metadata['hyperparams'] = hyperparams_for_train

        with open(metadata_filename_abs, 'w') as f:
            json.dump(serializable_metadata, f, indent=2, cls=NpEncoder) # Use encoder if needed
        metadata_filename = metadata_filename_relative # Store relative path
        completed_exp_data['metadata_filename'] = metadata_filename
        # Metadata saved silently
    except Exception as e:
         print(f"❌ ERROR [{experiment_id}]: Failed to save metadata: {e}")
         completed_exp_data['metadata_filename'] = None # Mark as failed
         # Update status only if it was previously considered successful
         if completed_exp_data['status'] not in ['failed', 'interrupted', 'crashed', 'setup_failed_data_generation', 'setup_interrupted_data_generation']:
            completed_exp_data['status'] = 'completed_metadata_save_failed'

    # Return the enriched experiment data dictionary
    return completed_exp_data

# Helper Encoder if numpy arrays sneak into metadata somehow
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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
        'cipher_names': ciphers_list, # Use cipher_names as expected by get_data
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


def initialize_environment_and_config(args):
    """Sets up paths, checks GPU/numpy, registers signals, returns config."""
    config = {}

    # Set up paths
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    config['project_root'] = PROJECT_ROOT
    config['data_dir'] = data_dir
    config['pending_file'] = os.path.join(data_dir, 'pending_experiments.json')
    config['completed_file'] = os.path.join(data_dir, 'completed_experiments.json')

    # Environment info
    gpu_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if gpu_available else "CPU"
    cuda_devices = torch.cuda.device_count() if gpu_available else 0
    
    # Configure numpy
    try:
        import numpy as np
        numpy_version = np.__version__
        config['np'] = np
        config['numpy_available'] = True
    except ImportError:
        numpy_version = "Not Available"
        config['np'] = None
        config['numpy_available'] = False
        
    # Print environment info in two compact lines
    print(f"Versions: PyTorch {torch.__version__} | NumPy {numpy_version}")
    if gpu_available and cuda_devices > 1:
        print(f"Devices: {cuda_devices}x {device_name}")
    else:
        print(f"Device: {device_name}")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Define the checker function
    def check_continue_flag():
        return should_continue
    config['check_continue_flag'] = check_continue_flag

    return config


def run_immediate_mode(args, config):
    """Handles the logic for immediate mode execution."""
    if not config['check_continue_flag']():
         print("Shutdown signal detected before starting immediate run.")
         return

    print("\n--- Running in Immediate Mode ---")
    experiment_config = create_immediate_experiment(args)
    completed_exp = run_experiment(experiment_config, config['check_continue_flag'])

    if completed_exp:
        print("\n--- Immediate Experiment Summary ---")
        details = get_experiment_details(completed_exp)
        print(details)
        append_to_experiment_file(config['completed_file'], completed_exp)
        print("Immediate experiment result saved to completed log.")
    else:
         status = completed_exp.get('status', 'failed_unknown') if completed_exp else 'failed_unknown'
         print(f"Immediate experiment did not complete successfully (Status: {status}). Check logs.")
         if completed_exp:
             append_to_experiment_file(config['completed_file'], completed_exp)
    print("--- Immediate Mode Finished ---")


def initialize_queue_processing(config):
    """Loads queue, history, calculates initial EMA, returns queue state."""
    queue_state = {
        'pending_experiments': [],
        'completed_keys': set(),
        'current_ema_duration': None,
        'initial_run_count': 0
    }

    pending_file = config['pending_file']
    completed_file = config['completed_file']
    np = config['np']
    numpy_available = config['numpy_available']

    # Load Pending Experiments
    queue_state['pending_experiments'] = get_pending_experiments(pending_file)

    # Initialize EMA Duration from History
    try:
        all_completed = safe_json_load(completed_file)
        successful_runs = [
            exp for exp in all_completed
            if exp.get('status') in ['completed', 'stopped_early']
               and isinstance(exp.get('metrics', {}).get('training_duration'), (int, float))
               and not math.isnan(exp['metrics']['training_duration'])
               and not math.isinf(exp['metrics']['training_duration'])
        ]
        historical_durations = [exp['metrics']['training_duration'] for exp in successful_runs]
        queue_state['initial_run_count'] = len(historical_durations)

        if historical_durations:
            if numpy_available:
                queue_state['current_ema_duration'] = np.mean(historical_durations)
            else:
                queue_state['current_ema_duration'] = sum(historical_durations) / len(historical_durations)

        # Get completed keys after loading
        queue_state['completed_keys'] = {exp.get('experiment_id') for exp in all_completed if exp.get('experiment_id')}
        del all_completed # Free memory

    except FileNotFoundError:
        print("Warning: completed_experiments.json not found. Starting with no history.")
    except Exception as e:
        print(f"Warning: Error loading or processing historical data: {e}. Starting with no history.")
        # Ensure completed_keys is initialized even on error
        queue_state['completed_keys'] = set()

    return queue_state


def process_single_experiment(exp_config, config, queue_state):
    """
    Runs one experiment via run_experiment, logs results, summarizes status,
    updates EMA duration, and determines if the experiment needs requeuing due
    to interruption (either during setup/data-gen or training).

    Args:
        exp_config (Dict): Config for the experiment to run.
        config (Dict): General researcher config (paths, directories, etc.).
        queue_state (Dict): Current queue state (contains EMA).

    Returns:
        Tuple[Optional[Dict], float, bool, Optional[str]]:
            - completed_exp_data (Dict | None): Processed data dict for the experiment.
                                                Contains status, metadata, metrics.
                                                Should always be a dict, even on failure.
            - updated_ema_duration (float): The EMA duration after potentially updating.
            - was_interrupted (bool): True if the experiment status indicates an interruption
                                      requiring requeue.
            - summary_details (str | None): Formatted string summary of the experiment details,
                                           or a basic status message if details cannot be generated.
    """
    # Extract config and state details
    check_continue_flag = config['check_continue_flag'] # Function from researcher.py main scope
    completed_file = config['completed_file']
    current_ema_duration = queue_state['current_ema_duration'] # Get current EMA state
    # Use experiment_id from config as the canonical ID unless run_experiment returns a different one
    experiment_id = exp_config.get('experiment_id', 'Error_No_ID_In_Config')

    # --- Run Experiment ---
    # run_experiment now returns a dictionary even on setup failures/crashes
    completed_exp_data = run_experiment(exp_config, check_continue_flag)

    # --- Process Results ---
    new_duration = None         # Duration of the completed run for EMA calculation
    was_interrupted = False     # Flag to indicate if requeue is needed
    exp_status = 'runner_error' # Default status if data is invalid
    exp_id_reported = experiment_id # Default to config ID for reporting
    summary_details = None      # Initialize details string

    if completed_exp_data and isinstance(completed_exp_data, dict):
        # Successfully got a result dictionary from run_experiment
        exp_status = completed_exp_data.get('status', 'unknown_status_in_data')
        exp_id_reported = completed_exp_data.get('experiment_id', experiment_id) # Use ID from result if available

        # --- Check for Interruption Statuses ---
        # Interruption can happen during data generation (new) or training (old)
        interruption_statuses = {'interrupted', 'setup_interrupted_data_generation'}
        if exp_status in interruption_statuses:
            was_interrupted = True
            print(f"   INFO [{exp_id_reported}]: Experiment was interrupted (status: {exp_status}). Will be requeued.")
            # Do NOT log to completed_file for interrupted runs that need requeuing.
        else:
            # Experiment finished (completed, failed, crashed, setup_failed etc.) but wasn't interrupted.
            # Append the result to the completed experiments log file.
            append_to_experiment_file(completed_file, completed_exp_data)
            # Note: The main loop updates the 'completed_keys' set based on 'was_interrupted' flag.

        # --- Generate Summary Details String ---
        # Try to generate details from the returned data, even for partial/failed runs
        try:
            summary_details = get_experiment_details(completed_exp_data) # Assumes get_experiment_details handles various statuses
        except Exception as e:
            print(f"   WARNING [{exp_id_reported}]: Could not generate full experiment details: {e}")
            # Provide a basic summary fallback
            summary_details = (f"ID: {exp_id_reported} | Status: {exp_status} | "
                               f"Note: Error generating full summary. Check logs/results file.")

        # --- Extract duration for EMA (only for runs considered 'successful enough') ---
        # Define statuses that represent a completed run whose duration is meaningful for ETA
        successful_statuses_for_ema = {'completed', 'stopped_early', 'completed_safety_limit'}
        if exp_status in successful_statuses_for_ema:
            # Extract duration safely, checking type and validity
            duration_val = completed_exp_data.get('metrics', {}).get('training_duration')
            if isinstance(duration_val, (int, float)) and not math.isnan(duration_val) and not math.isinf(duration_val) and duration_val >= 0:
                new_duration = duration_val
            else:
                print(f"   WARNING [{exp_id_reported}]: Invalid or missing 'training_duration' in metrics for EMA calculation (value: {duration_val}).")

    else:
        # Handle unexpected case where run_experiment itself failed catastrophically and returned None or invalid data
        # This case should ideally be rare now as run_experiment aims to always return a dict.
        print(f"❌ ERROR [{experiment_id}]: run_experiment call returned invalid data (not a dict).")
        exp_status = 'runner_error' # Assign specific status for this rare case
        # Create a minimal failure dictionary to log and return
        completed_exp_data = {
            'experiment_id': experiment_id,
            'status': exp_status,
            'original_config': exp_config, # Include original config for debugging
            'run_timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'error_message': 'run_experiment returned None or invalid data'
        }
        append_to_experiment_file(completed_file, completed_exp_data)
        summary_details = f"ID: {experiment_id} | Status: {exp_status} | CRITICAL: run_experiment call failed unexpectedly."
        # completed_exp_data is now the minimal dict, EMA won't update, was_interrupted is False

    # --- Update EMA Duration ---
    updated_ema_duration = current_ema_duration # Start with current EMA
    if new_duration is not None:
        # A valid duration was extracted from a successful run
        if current_ema_duration is None:
            # Initialize EMA if this is the first valid run duration
            updated_ema_duration = new_duration
            print(f"   INFO [EMA]: Initialized with first run duration: {format_duration(updated_ema_duration)}")
        else:
            # Update existing EMA using the exponential moving average formula
            updated_ema_duration = (new_duration * EMA_ALPHA) + (current_ema_duration * (1 - EMA_ALPHA))
            print(f"   INFO [EMA]: Updated: Previous={format_duration(current_ema_duration)}, New={format_duration(new_duration)}, Next={format_duration(updated_ema_duration)}")
    elif updated_ema_duration is None and not was_interrupted and exp_status != 'runner_error':
        # If EMA is still None after a non-interrupted, non-runner-error run, log a warning.
        print(f"   WARNING [EMA]: EMA remains uninitialized after experiment {exp_id_reported} (status: {exp_status}).")


    # Return the result dictionary, the potentially updated EMA, the interruption flag, and the summary string
    return completed_exp_data, updated_ema_duration, was_interrupted, summary_details


def display_progress_and_eta(processed_count, original_pending_count, remaining_count, current_ema_duration, initial_run_count):
    """Calculates and returns the Progress and ETA strings."""
    progress_str = f"Progress: {processed_count}/{original_pending_count} processed this session | {remaining_count} remaining in queue"

    eta_str = "ETA: Calculating..."
    if current_ema_duration is not None:
        if remaining_count > 0:
            eta_seconds = current_ema_duration * remaining_count
            formatted_eta = format_duration(eta_seconds)
            # Use f-string interpolation directly for clarity
            basis_str = f"based on {initial_run_count} historical runs" if initial_run_count > 0 else "from runs this session"
            eta_str = f"ETA: {formatted_eta} (EMA {basis_str})"
        else:
             eta_str = "ETA: Queue complete!"
    elif remaining_count == 0:
         eta_str = "ETA: Queue complete!"
         # EMA is None and queue is empty -> no successful runs occurred.

    return progress_str, eta_str


def finalize_queue_processing(status_should_continue, pending_experiments):
    """Prints the final status message after the queue loop."""
    if not status_should_continue:
        print("\nQueue processing interrupted by signal.")
    elif not pending_experiments:
        print("\n🎉 Experiment queue processing complete.")
        try:
            notifications.send_discord_notification("🎉 All pending experiments have been processed!")
        except Exception as e:
            print(f"Warning: Failed to send final Discord notification: {e}")
    else:
        print("\nQueue processing finished unexpectedly with items remaining.")


# --- Main Execution ---

def main():
    global should_continue # Allow main to modify the global flag if needed (though signal handler does)
    args = argument_parser()

    # --- Phase 1: Initialization ---
    config = initialize_environment_and_config(args)
    check_continue_flag = config['check_continue_flag'] # Local alias for convenience

    # --- Phase 2: Mode Selection ---
    if args.immediate:
        run_immediate_mode(args, config)
        return # Exit after immediate mode

    # --- Phase 3: Queue Processing Setup ---
    queue_state = initialize_queue_processing(config)
    pending_experiments = queue_state['pending_experiments'] # Local alias
    completed_keys = queue_state['completed_keys']       # Local alias
    current_ema_duration = queue_state['current_ema_duration'] # Local state

    if not pending_experiments:
        print("\n📋 Experiment queue is empty.")
        print("Use manage_queue.py or suggest_experiments.py to add experiments.")
        return

    original_pending_count = len(pending_experiments)
    processed_count = 0
    print(f"Found {original_pending_count} experiments in the queue.")

    # --- Phase 4: Queue Processing Loop ---
    while pending_experiments and check_continue_flag():
        # Check signal before pop
        if not check_continue_flag(): break

        # Get next experiment config (store original in case we need to requeue)
        exp_config = pending_experiments.pop(0)
        # Use experiment_id from config as the fallback ID
        experiment_id_from_config = exp_config.get('experiment_id', 'MISSING_ID_IN_CONFIG')

        # --- Pre-run Checks ---
        if experiment_id_from_config == 'MISSING_ID_IN_CONFIG':
             print("ERROR: Found experiment config with missing ID. Skipping.")
             rewrite_experiment_file(config['pending_file'], pending_experiments)
             continue # Skip to next iteration
        if experiment_id_from_config in completed_keys:
            print(f"Skipping {experiment_id_from_config}: Found in completed experiments log.")
            rewrite_experiment_file(config['pending_file'], pending_experiments)
            continue # Skip to next iteration
        if not check_continue_flag(): # Check signal *right* before run
            print(f"Shutdown signal detected before running {experiment_id_from_config}.")
            pending_experiments.insert(0, exp_config) # Put back original config
            break # Exit loop, state will be saved outside

        # --- Process the Single Experiment ---
        # Calls run_experiment -> train_model internally
        # Returns: processed data dict, updated EMA, interrupted flag, summary string
        completed_data, updated_ema, was_interrupted, summary_details = process_single_experiment(
            exp_config, config, queue_state
        )

        # --- Update State Based on Outcome ---
        processed_count += 1
        current_ema_duration = updated_ema # Update EMA state for next iteration/ETA calc

        # Determine the definitive experiment ID and status for logging/notification
        exp_id_for_reporting = experiment_id_from_config # Default to config ID
        exp_status = 'unknown' # Default status

        if was_interrupted:
            # --- Handle Interruption: Requeue ---
            print(f"   Requeuing interrupted experiment {exp_id_for_reporting}...")
            # Use the ID from the original config for requeuing
            pending_experiments.insert(0, exp_config) # Put original config back at the front
            # Set status explicitly for interrupted case
            exp_status = 'interrupted'
            # If completed_data exists (it should for interrupts), update reporting ID if different
            if completed_data and completed_data.get('experiment_id'):
                exp_id_for_reporting = completed_data['experiment_id']

        elif completed_data and completed_data.get('experiment_id'):
            # --- Handle Completion/Failure: Mark as completed ---
            # Use ID and status from the returned data (most reliable source)
            exp_id_for_reporting = completed_data['experiment_id']
            exp_status = completed_data.get('status', 'unknown_status_in_data')
            # Add ID to completed set only if it wasn't interrupted
            completed_keys.add(exp_id_for_reporting)
            # Note: process_single_experiment already printed status and appended to file

        else:
             # --- Handle Runner Error (completed_data is None or invalid) ---
             print(f"   Marking experiment {exp_id_for_reporting} as completed (runner_error).")
             completed_keys.add(exp_id_for_reporting) # Add config ID to avoid re-running
             exp_status = 'runner_error' # Set status explicitly for runner error
             # summary_details should have been set inside process_single_experiment for this case

        # --- Save Queue State ---
        # Save the pending experiments list (potentially with the interrupted one re-inserted)
        rewrite_experiment_file(config['pending_file'], pending_experiments)

        # --- Get Progress and ETA Strings ---
        remaining_count = len(pending_experiments)
        progress_str, eta_str = display_progress_and_eta(
            processed_count,
            original_pending_count,
            remaining_count,
            current_ema_duration, # Use the updated EMA
            queue_state['initial_run_count']
        )

        # --- Send Discord Notification ---
        # Determine prefix based on the final status
        notification_prefix = "❓" # Default
        if exp_status in ['completed', 'stopped_early', 'completed_safety_limit']: notification_prefix = "✅"
        elif exp_status == 'interrupted': notification_prefix = "🛑"
        elif exp_status in ['failed', 'crashed', 'setup_failed_missing_params', 'setup_failed_data_load', 'setup_failed_data_exception', 'runner_train_call_error', 'completed_save_failed', 'completed_model_missing', 'completed_metadata_save_failed', 'runner_error', 'unknown_status_in_data']: notification_prefix = "❌" # Added runner_error and unknowns
        # Add other potential failure statuses if needed

        # Compose the notification message
        # Use the reporting ID and status determined above
        # Format status message - don't uppercase 'stopped_early' since it's not an error
        if exp_status == 'stopped_early':
            status_display = "Training complete (early stopping)"
        else:
            status_display = exp_status.replace('_', ' ').upper()
            
        notification_title = f"{notification_prefix} Exp {exp_id_for_reporting}: {status_display}"
        # Use the retrieved summary_details string, progress, and eta strings
        # Add fallback for summary_details if it's None
        details_for_notification = summary_details if summary_details else f"Status: {exp_status}"
        # Combine all parts for the Discord message
        discord_message = f"{notification_title}\n{details_for_notification}\n{progress_str}\n{eta_str}"

        try:
            notifications.send_discord_notification(discord_message)
        except Exception as e:
            print(f"Warning: Failed to send Discord notification: {e}")

        # --- Display Console Output ---
        # Print the fully composed message which is also sent to Discord
        # This ensures console and notification are identical.
        # discord_message includes: Prefix, Title, Details, Progress, ETA
        print("\n--- Experiment Summary ---") # Keep the header for clarity
        print(discord_message)              # Print the unified message
        print("----------------------------------------") # Keep the separator

    # --- Phase 5: Finalization ---
    # Pass the *current* value of the global flag
    finalize_queue_processing(should_continue, pending_experiments)
    # Ensure queue state is saved if loop broken by signal
    if not should_continue:
         rewrite_experiment_file(config['pending_file'], pending_experiments)

    print("--- Researcher finished ---")


if __name__ == "__main__":
    setup_signal_handlers()
    main()
