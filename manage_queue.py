#!/usr/bin/env python3
"""
Experiment Queue Management Tool

Manages the experiment queue using standardized experiment IDs and robust
parameter-based duplicate checking.

Usage:
    # Add experiments to queue (default)
    python manage_queue.py --d_model 128,256 --nhead 4,8

    # Replace entire queue with new experiments
    python manage_queue.py --replace --d_model 128,256 --nhead 4,8

    # Clear all experiments from queue
    python manage_queue.py --clear

    # List current experiments in queue
    python manage_queue.py --list
"""

import os
import sys
import json
import argparse
import itertools
import datetime
from ciphers import _get_cipher_names
from typing import List, Dict, Any, Set, Tuple, Optional

# --- Import Utilities ---
try:
    from models.common.utils import (
        safe_json_load,
        _find_max_daily_counter,
        PENDING_EXPERIMENTS_FILE,
        COMPLETED_EXPERIMENTS_FILE
    )
except ImportError as e:
    print(f"Error importing utilities: {e}")
    print("Ensure you are running from the project root and models/common/utils.py is accessible.")
    sys.exit(1)

# Set file location as working directory (if run from elsewhere)
# dir_path = os.path.dirname(os.path.realpath(__file__))
# os.chdir(dir_path)

# Default parameter settings for experiments
# Learning rate removed, it's now determined automatically in train.py
DEFAULT_PARAMS = {
    'ciphers': [_get_cipher_names()],
    'num_samples': [100000],
    'sample_length': [500],
    # Transformer hyperparameters
    'd_model': [128],
    'nhead': [4],
    'num_encoder_layers': [2],
    'dim_feedforward': [512],
    'batch_size': [32],
    'dropout_rate': [0.1],
    'base_patience': [5] # Default base patience
    # Optional scheduler params can still be added if needed
}

# File paths are imported from utils


def save_json(file_path: str, data: List[Dict]):
    """Save data as JSON to a file."""
    abs_file_path = os.path.abspath(file_path)
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(abs_file_path), exist_ok=True)
        with open(abs_file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"ERROR: Could not save JSON to {abs_file_path}: {e}")


def parse_comma_separated_values(value: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated values into a list of strings."""
    if value is None:
        return None
    return [item.strip() for item in value.split(',') if item.strip()]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Manage experiment queue')

    # Queue management flags
    parser.add_argument('--clear', action='store_true', help='Clear the experiment queue')
    parser.add_argument('--replace', action='store_true', help='Replace the current queue with new experiments')
    parser.add_argument('--list', action='store_true', help='List current pending experiments')

    # Experiment parameters - allow multiple values via comma separation
    # Use lowercase 'all' for consistency in help text
    parser.add_argument('--ciphers', type=parse_comma_separated_values, help='Comma-separated list of ciphers (use lowercase), or "all"')
    parser.add_argument('--num_samples', type=parse_comma_separated_values, help='Comma-separated list of sample counts (e.g., "100000,200000")')
    parser.add_argument('--sample_length', type=parse_comma_separated_values, help='Comma-separated list of sample lengths (e.g., "500,1000")')
    parser.add_argument('--d_model', type=parse_comma_separated_values, help='Comma-separated list of embedding dimensions')
    parser.add_argument('--nhead', type=parse_comma_separated_values, help='Comma-separated list of attention head counts')
    parser.add_argument('--num_encoder_layers', type=parse_comma_separated_values, help='Comma-separated list of encoder layer counts')
    parser.add_argument('--dim_feedforward', type=parse_comma_separated_values, help='Comma-separated list of feedforward dimensions')
    parser.add_argument('--batch_size', type=parse_comma_separated_values, help='Comma-separated list of batch sizes')
    parser.add_argument('--dropout_rate', type=parse_comma_separated_values, help='Comma-separated list of dropout rates')
    parser.add_argument('--base_patience', type=parse_comma_separated_values, help='Comma-separated list of base patience values (used for LR scheduler and early stopping)')
    # Optional: Add arguments for scheduler params if desired
    # parser.add_argument('--lr_factor', type=parse_comma_separated_values, help='Factor for ReduceLROnPlateau')
    # parser.add_argument('--lr_patience', type=parse_comma_separated_values, help='Patience for ReduceLROnPlateau')
    # parser.add_argument('--min_lr', type=parse_comma_separated_values, help='Min LR for ReduceLROnPlateau')

    return parser.parse_args()


def process_args(args: argparse.Namespace) -> Dict[str, List[Any]]:
    """Process command line arguments into typed parameter lists for experiment generation."""
    params = {} # Start fresh

    # Helper to process arg with type conversion and default fallback
    def _process_param(arg_name, cli_values, default_value, type_converter):
        if cli_values is not None:
            try:
                # Ensure positive integers for patience
                processed = [type_converter(x) for x in cli_values]
                if type_converter == int and arg_name == 'base_patience':
                    if any(p <= 0 for p in processed):
                        print(f"Error: --base_patience values must be positive integers. Using default.")
                        return default_value # Return the default list
                return processed
            except ValueError as e:
                print(f"Error converting value for --{arg_name}: {e}. Using default.")
                return default_value # Return the default list
        else:
            return default_value # Return the default list

    # --- Process Ciphers (Convert to Lowercase) ---
    if args.ciphers is not None:
        # Convert provided cipher names to lowercase immediately
        lowercase_ciphers_arg = [c.lower() for c in args.ciphers]

        if lowercase_ciphers_arg == ['all']:
            # Get default list (ensure it's lowercase)
            all_ciphers_lower = _get_cipher_names()
            if not all(c == c.lower() for c in all_ciphers_lower):
                 print("Warning: _get_cipher_names() in ciphers.py did not return all lowercase names. Converting.")
                 all_ciphers_lower = [c.lower() for c in all_ciphers_lower]
            params['ciphers'] = [all_ciphers_lower] # Wrap in outer list for itertools.product
        else:
            # Use the lowercased list provided by user
            params['ciphers'] = [lowercase_ciphers_arg] # Wrap in outer list
    else:
        # Use default (ensure it's lowercase)
        all_ciphers_lower = _get_cipher_names()
        if not all(c == c.lower() for c in all_ciphers_lower):
             print("Warning: _get_cipher_names() in ciphers.py did not return all lowercase names. Converting default.")
             all_ciphers_lower = [c.lower() for c in all_ciphers_lower]
        params['ciphers'] = [all_ciphers_lower]

    # Data parameters
    params['num_samples'] = _process_param('num_samples', args.num_samples, DEFAULT_PARAMS['num_samples'], int)
    params['sample_length'] = _process_param('sample_length', args.sample_length, DEFAULT_PARAMS['sample_length'], int)

    # Transformer hyperparameters
    params['d_model'] = _process_param('d_model', args.d_model, DEFAULT_PARAMS['d_model'], int)
    params['nhead'] = _process_param('nhead', args.nhead, DEFAULT_PARAMS['nhead'], int)
    params['num_encoder_layers'] = _process_param('num_encoder_layers', args.num_encoder_layers, DEFAULT_PARAMS['num_encoder_layers'], int)
    params['dim_feedforward'] = _process_param('dim_feedforward', args.dim_feedforward, DEFAULT_PARAMS['dim_feedforward'], int)
    params['batch_size'] = _process_param('batch_size', args.batch_size, DEFAULT_PARAMS['batch_size'], int)
    params['dropout_rate'] = _process_param('dropout_rate', args.dropout_rate, DEFAULT_PARAMS['dropout_rate'], float)
    params['base_patience'] = _process_param('base_patience', args.base_patience, DEFAULT_PARAMS['base_patience'], int)
    # Optional: Process scheduler params if args were added and exist in DEFAULT_PARAMS
    # for sched_param in ['lr_scheduler_factor', 'lr_scheduler_patience', 'lr_scheduler_min_lr']:
    #     if sched_param in DEFAULT_PARAMS: # Check if defined in defaults
    #         # Determine the corresponding arg name (e.g., lr_scheduler_factor -> lr_factor)
    #         arg_name = sched_param.replace('scheduler_', '') # Simple replacement logic
    #         cli_arg_value = getattr(args, arg_name, None)
    #         # Determine the appropriate type converter
    #         type_conv = float if 'factor' in sched_param or 'min_lr' in sched_param else int
    #         params[sched_param] = _process_param(arg_name, cli_arg_value, DEFAULT_PARAMS[sched_param], type_conv)

    return params


def generate_experiments(params: Dict[str, List[Any]]) -> List[Dict]:
    """
    Generate valid experiment configurations from parameter combinations,
    assigning a unique sequential ID to each within the batch run.
    Exits with an error if any combination is invalid.
    """
    keys = list(params.keys())
    values = list(params.values())

    combinations = list(itertools.product(*values))
    if not combinations:
        print("No parameter combinations generated based on input.")
        return []

    print(f"Generated {len(combinations)} parameter combinations to validate.")

    valid_experiment_configs_parts = [] # Store parts {data_params, hyperparams}
    invalid_combinations_details = []

    # Identify which keys belong to data_params vs hyperparams based on current structure
    data_param_keys = {'ciphers', 'num_samples', 'sample_length'}

    for i, combination in enumerate(combinations):
        param_dict = dict(zip(keys, combination))
        # Separate into data and hyperparams based on keys
        data_params = {k: param_dict[k] for k in data_param_keys if k in param_dict}
        hyperparams = {k: param_dict[k] for k in keys if k not in data_param_keys}

        is_valid = True
        validation_errors = []
        d_model = hyperparams.get('d_model')
        nhead = hyperparams.get('nhead')
        if d_model is not None and nhead is not None:
            if not isinstance(d_model, int) or not isinstance(nhead, int) or nhead <= 0 or d_model <= 0:
                 is_valid = False
                 validation_errors.append(f"d_model ({d_model}) and nhead ({nhead}) must be positive integers.")
            elif d_model % nhead != 0:
                is_valid = False
                validation_errors.append(f"d_model ({d_model}) is not divisible by nhead ({nhead}).")

        dropout = hyperparams.get('dropout_rate')
        if dropout is not None and not (0.0 <= dropout <= 1.0):
            is_valid = False
            validation_errors.append(f"dropout_rate ({dropout}) must be between 0.0 and 1.0.")

        # Add checks for any other constraints (e.g., scheduler params if added)

        if not is_valid:
            invalid_combinations_details.append({
                "combination_index": i + 1, "params": param_dict, "errors": validation_errors
            })
        else:
            # Store valid parts, ID will be assigned later if all pass
            valid_experiment_configs_parts.append({
                'data_params': data_params, 'hyperparams': hyperparams
            })

    if invalid_combinations_details:
        print("\nERROR: Invalid parameter combinations found. Aborting experiment generation.")
        print("Please correct the following combinations provided via command-line arguments:")
        for invalid in invalid_combinations_details:
            print(f"\n  Combination #{invalid['combination_index']}:")
            for key, value in invalid['params'].items():
                 print(f"    --{key} {value}")
            print(f"  Errors:")
            for error in invalid['errors']:
                print(f"    - {error}")
        sys.exit(1)
    else:
        # All combinations valid, now generate final list with unique sequential IDs
        final_experiments = []
        print(f"All {len(combinations)} generated parameter combinations are valid.")

        today_prefix = datetime.datetime.now().strftime("%Y%m%d")
        current_max_counter = _find_max_daily_counter(today_prefix)
        next_counter = current_max_counter + 1 # Start from the next available number

        for i, config_parts in enumerate(valid_experiment_configs_parts):
            experiment_id = f"{today_prefix}-{next_counter + i}"
            experiment = {
                'experiment_id': experiment_id,
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_params': config_parts['data_params'],
                'hyperparams': config_parts['hyperparams']
            }
            final_experiments.append(experiment)

        print(f"Created {len(final_experiments)} valid experiment configurations with unique IDs starting from {today_prefix}-{next_counter}.")
        return final_experiments


def _get_comparable_params(exp: Dict) -> Optional[tuple]:
    """
    Internal helper to create a hashable, order-independent representation
    of an experiment's parameters (data + hyper) for duplicate checking.
    Sorts lists within parameters (like 'ciphers') for consistency.
    """
    data_params = exp.get('data_params')
    hyperparams = exp.get('hyperparams')

    if not isinstance(data_params, dict) or not isinstance(hyperparams, dict):
        return None # Invalid format

    # Combine parameters (Learning Rate is no longer here)
    combined_params = {**data_params, **hyperparams}

    # Create sorted list of items, handling list values specifically
    sorted_items = []
    for key, value in sorted(combined_params.items()):
        if isinstance(value, list):
            try:
                sorted_value = sorted(value)
                sorted_items.append((key, tuple(sorted_value)))
            except TypeError:
                sorted_items.append((key, tuple(value)))
        else:
            try:
                 hash(value)
                 sorted_items.append((key, value))
            except TypeError:
                 print(f"Warning: Non-hashable value type {type(value)} for key '{key}' in experiment {exp.get('experiment_id')}. Cannot check duplicates based on this.")
                 return None # Skip this experiment for duplicate check

    return tuple(sorted_items)


def check_for_duplicates(new_experiments: List[Dict],
                           existing_experiments: List[Dict]) -> List[Dict]:
    """
    Check for new experiments whose parameter combinations already exist in
    the existing experiments list. Uses _get_comparable_params.
    """
    existing_param_tuples: Set[tuple] = set()
    for exp in existing_experiments:
        param_tuple = _get_comparable_params(exp)
        if param_tuple is not None:
            existing_param_tuples.add(param_tuple)

    unique_experiments = []
    processed_new_tuples = set() # Track tuples within the new batch itself
    for new_exp in new_experiments:
        param_tuple = _get_comparable_params(new_exp)
        if param_tuple is not None:
            if param_tuple not in existing_param_tuples and param_tuple not in processed_new_tuples:
                unique_experiments.append(new_exp)
                processed_new_tuples.add(param_tuple) # Add to avoid intra-batch duplicates

    return unique_experiments


def list_queue():
    """List all pending experiments in the queue."""
    pending_experiments = safe_json_load(PENDING_EXPERIMENTS_FILE)

    if not pending_experiments:
        print("Experiment queue is empty.")
        return

    print(f"Queue contains {len(pending_experiments)} experiments:")
    for i, exp in enumerate(pending_experiments):
        exp_id = exp.get('experiment_id', f'Unknown_{i}')
        params = exp.get('hyperparams', {})
        d_params = exp.get('data_params', {})

        # Use base_patience now
        base_patience = params.get('base_patience', '?')
        d_model = params.get('d_model', '?')
        nhead = params.get('nhead', '?')
        layers = params.get('num_encoder_layers', '?')
        ff_dim = params.get('dim_feedforward', '?')
        batch = params.get('batch_size', '?')
        num_samples = d_params.get('num_samples', '?')
        sample_len = d_params.get('sample_length', '?')

        print(f"{i+1}. ID: {exp_id} | d={d_model}, h={nhead}, l={layers}, ff={ff_dim}, bs={batch}, base_pat={base_patience} | Samples={num_samples}x{sample_len}")

def clear_queue():
    """Clear all pending experiments from the queue."""
    save_json(PENDING_EXPERIMENTS_FILE, [])
    print(f"Experiment queue cleared ({PENDING_EXPERIMENTS_FILE}).")


def add_to_queue(experiments_to_add: List[Dict]):
    """Add new, non-duplicate experiments to the queue."""
    if not experiments_to_add:
        print("No valid experiments generated to add.")
        return

    pending_experiments = safe_json_load(PENDING_EXPERIMENTS_FILE)
    completed_experiments = safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    all_existing = pending_experiments + completed_experiments

    unique_experiments = check_for_duplicates(experiments_to_add, all_existing)

    num_generated = len(experiments_to_add)
    num_duplicates = num_generated - len(unique_experiments)
    num_to_add = len(unique_experiments)

    print(f"Generated {num_generated} experiment configurations.")
    print(f"Detected {num_duplicates} duplicates (already pending or completed).")

    if num_to_add == 0:
        print("No new, unique experiments to add to the queue.")
        return

    current_pending_count = len(pending_experiments)
    pending_experiments.extend(unique_experiments)
    save_json(PENDING_EXPERIMENTS_FILE, pending_experiments)

    print(f"Added {num_to_add} unique experiments to the queue.")
    print(f"Total experiments now in queue: {len(pending_experiments)} (was {current_pending_count}).")


def replace_queue(experiments_to_replace_with: List[Dict]):
    """Replace the current queue with new, non-duplicate experiments."""
    if not experiments_to_replace_with:
        print("No valid experiments generated to replace the queue with.")
        return

    completed_experiments = safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    unique_experiments = check_for_duplicates(experiments_to_replace_with, completed_experiments)

    num_generated = len(experiments_to_replace_with)
    num_duplicates = num_generated - len(unique_experiments)
    num_in_new_queue = len(unique_experiments)

    print(f"Generated {num_generated} experiment configurations.")
    print(f"Detected {num_duplicates} duplicates (already completed).")

    save_json(PENDING_EXPERIMENTS_FILE, unique_experiments)

    print(f"Queue replaced with {num_in_new_queue} unique experiments.")


def main():
    """Main function to parse args and manage the experiment queue."""
    args = parse_arguments()

    if args.clear:
        clear_queue()
        return
    if args.list:
        list_queue()
        return

    params = process_args(args)
    new_experiments = generate_experiments(params)

    if args.replace:
        replace_queue(new_experiments)
    else:
        add_to_queue(new_experiments)


if __name__ == "__main__":
    main()
