#!/usr/bin/env python3
"""
Experiment Queue Management Tool

This script manages the experiment queue for the cipher classifier project.
It can generate permutations of experiment parameters and add them to the queue,
replace the current queue with new experiments, or clear the queue entirely.

Usage:
    # Add experiments to queue (default)
    python manage_queue.py --d_model 128,256 --nhead 4,8

    # Replace entire queue with new experiments
    python manage_queue.py --replace --d_model 128,256 --nhead 4,8

    # Clear all experiments from queue
    python manage_queue.py --clear
"""

import os
import sys
import json
import argparse
import itertools
import datetime
from ciphers import _get_cipher_names

# Set file location as working directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

# Default parameter settings for experiments
DEFAULT_PARAMS = {
    'ciphers': [_get_cipher_names()],
    'num_samples': [100000],
    'sample_length': [500],
    'epochs': [30],
    # Transformer hyperparameters
    'd_model': [128, 256],  # Embedding dimension
    'nhead': [4, 8],  # Number of attention heads
    'num_encoder_layers': [2, 4],  # Number of transformer layers
    'dim_feedforward': [512, 1024],  # Hidden dimension in feed forward network
    'batch_size': [32, 64],
    'dropout_rate': [0.1, 0.2],
    'learning_rate': [1e-4, 3e-4]  # Lower learning rates for transformers
}

# File paths
PENDING_EXPERIMENTS_FILE = 'data/pending_experiments.json'
COMPLETED_EXPERIMENTS_FILE = 'data/completed_experiments.json'


def safe_json_load(file_path):
    """Load JSON data safely from a file."""
    try:
        if not os.path.exists(file_path):
            return []
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}. Returning empty list.")
        return []
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Returning empty list.")
        return []


def save_json(file_path, data):
    """Save data as JSON to a file."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_comma_separated_values(value):
    """Parse comma-separated values into a list."""
    if value is None:
        return None
    return [item.strip() for item in value.split(',')]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Manage experiment queue')
    
    # Queue management flags
    parser.add_argument('--clear', action='store_true', help='Clear the experiment queue')
    parser.add_argument('--replace', action='store_true', help='Replace the current queue with new experiments')
    parser.add_argument('--list', action='store_true', help='List current pending experiments')
    
    # Experiment parameters
    parser.add_argument('--ciphers', type=parse_comma_separated_values, help='Comma-separated list of ciphers')
    parser.add_argument('--num_samples', type=parse_comma_separated_values, help='Comma-separated list of sample counts')
    parser.add_argument('--sample_length', type=parse_comma_separated_values, help='Comma-separated list of sample lengths')
    parser.add_argument('--epochs', type=parse_comma_separated_values, help='Comma-separated list of epoch counts')
    
    # Transformer parameters
    parser.add_argument('--d_model', type=parse_comma_separated_values, help='Comma-separated list of embedding dimensions')
    parser.add_argument('--nhead', type=parse_comma_separated_values, help='Comma-separated list of attention head counts')
    parser.add_argument('--num_encoder_layers', type=parse_comma_separated_values, help='Comma-separated list of encoder layer counts')
    parser.add_argument('--dim_feedforward', type=parse_comma_separated_values, help='Comma-separated list of feedforward dimensions')
    parser.add_argument('--batch_size', type=parse_comma_separated_values, help='Comma-separated list of batch sizes')
    parser.add_argument('--dropout_rate', type=parse_comma_separated_values, help='Comma-separated list of dropout rates')
    parser.add_argument('--learning_rate', type=parse_comma_separated_values, help='Comma-separated list of learning rates')
    
    return parser.parse_args()


def process_args(args):
    """Process command line arguments into experiment parameters."""
    params = DEFAULT_PARAMS.copy()
    
    # Override defaults with any provided arguments
    if args.ciphers is not None:
        if args.ciphers == ['all']:
            params['ciphers'] = [_get_cipher_names()]
        else:
            params['ciphers'] = [args.ciphers]
    
    if args.num_samples is not None:
        params['num_samples'] = [int(x) for x in args.num_samples]
    
    if args.sample_length is not None:
        params['sample_length'] = [int(x) for x in args.sample_length]
    
    if args.epochs is not None:
        params['epochs'] = [int(x) for x in args.epochs]
    
    if args.d_model is not None:
        params['d_model'] = [int(x) for x in args.d_model]
    
    if args.nhead is not None:
        params['nhead'] = [int(x) for x in args.nhead]
    
    if args.num_encoder_layers is not None:
        params['num_encoder_layers'] = [int(x) for x in args.num_encoder_layers]
    
    if args.dim_feedforward is not None:
        params['dim_feedforward'] = [int(x) for x in args.dim_feedforward]
    
    if args.batch_size is not None:
        params['batch_size'] = [int(x) for x in args.batch_size]
    
    if args.dropout_rate is not None:
        params['dropout_rate'] = [float(x) for x in args.dropout_rate]
    
    if args.learning_rate is not None:
        params['learning_rate'] = [float(x) for x in args.learning_rate]
    
    return params


def get_experiment_key(exp):
    """Generate a unique key for an experiment based on its parameters."""
    data_params = exp.get('data_params', {})
    hyperparams = exp.get('hyperparams', {})
    
    key_parts = []
    
    # Add data parameters to key
    for cipher in sorted(data_params.get('ciphers', [])[0]):
        key_parts.append(cipher)
    
    key_parts.append(str(data_params.get('num_samples')))
    key_parts.append(str(data_params.get('sample_length')))
    
    # Add hyperparameters to key
    key_parts.append(str(hyperparams.get('epochs')))
    key_parts.append(str(hyperparams.get('d_model')))
    key_parts.append(str(hyperparams.get('nhead')))
    key_parts.append(str(hyperparams.get('num_encoder_layers')))
    key_parts.append(str(hyperparams.get('dim_feedforward')))
    key_parts.append(str(hyperparams.get('batch_size')))
    key_parts.append(str(hyperparams.get('dropout_rate')))
    key_parts.append(str(hyperparams.get('learning_rate')))
    
    return '_'.join(key_parts)


def generate_experiments(params):
    """
    Generate experiment configurations from parameter combinations.
    
    Args:
        params: Dictionary of parameter lists to combine
        
    Returns:
        List of experiment configurations
    """
    # Extract the keys and values from the params dictionary
    keys = list(params.keys())
    values = list(params.values())
    
    # Generate all combinations of parameters
    combinations = list(itertools.product(*values))
    
    # Create experiment configurations for each combination
    experiments = []
    for i, combination in enumerate(combinations):
        # Create a dictionary from the keys and the current combination
        param_dict = dict(zip(keys, combination))
        
        # Split parameters into data parameters and hyperparameters
        data_params = {
            'ciphers': param_dict['ciphers'],
            'num_samples': param_dict['num_samples'],
            'sample_length': param_dict['sample_length']
        }
        
        hyperparams = {
            'epochs': param_dict['epochs'],
            'd_model': param_dict['d_model'],
            'nhead': param_dict['nhead'],
            'num_encoder_layers': param_dict['num_encoder_layers'],
            'dim_feedforward': param_dict['dim_feedforward'],
            'batch_size': param_dict['batch_size'],
            'dropout_rate': param_dict['dropout_rate'],
            'learning_rate': param_dict['learning_rate']
        }
        
        # Create the experiment configuration
        experiment = {
            'experiment_id': f'exp_{i+1}',
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'data_params': data_params,
            'hyperparams': hyperparams
        }
        
        experiments.append(experiment)
    
    return experiments


def check_for_duplicates(new_experiments, existing_experiments):
    """
    Check for duplicate experiments based on parameter combinations.
    
    Args:
        new_experiments: List of new experiment configurations
        existing_experiments: List of existing experiment configurations
        
    Returns:
        List of non-duplicate experiments
    """
    # Extract existing experiment keys
    existing_keys = set()
    for exp in existing_experiments:
        key = get_experiment_key(exp)
        existing_keys.add(key)
    
    # Filter out duplicates
    unique_experiments = []
    for exp in new_experiments:
        key = get_experiment_key(exp)
        if key not in existing_keys:
            unique_experiments.append(exp)
    
    return unique_experiments


def list_queue():
    """List all pending experiments in the queue."""
    pending_experiments = safe_json_load(PENDING_EXPERIMENTS_FILE)
    
    if not pending_experiments:
        print("Experiment queue is empty.")
        return
    
    print(f"Queue contains {len(pending_experiments)} experiments:")
    for i, exp in enumerate(pending_experiments):
        # Extract key parameters for display
        exp_id = exp.get('experiment_id', f'Unknown_{i}')
        ciphers = ','.join(exp.get('data_params', {}).get('ciphers', ['unknown'])[0][:3])
        if len(exp.get('data_params', {}).get('ciphers', ['unknown'])[0]) > 3:
            ciphers += "..."
        
        params = exp.get('hyperparams', {})
        d_model = params.get('d_model', 'unknown')
        nhead = params.get('nhead', 'unknown')
        layers = params.get('num_encoder_layers', 'unknown')
        
        print(f"{i+1}. {exp_id}: d_model={d_model}, nhead={nhead}, layers={layers}, ciphers={ciphers}")


def clear_queue():
    """Clear all pending experiments from the queue."""
    save_json(PENDING_EXPERIMENTS_FILE, [])
    print("Experiment queue cleared.")


def add_to_queue(experiments):
    """Add experiments to the queue."""
    # Load existing experiments
    pending_experiments = safe_json_load(PENDING_EXPERIMENTS_FILE)
    completed_experiments = safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    
    # Check for duplicates against both pending and completed experiments
    unique_experiments = check_for_duplicates(experiments, pending_experiments + completed_experiments)
    
    # Print duplicate stats
    num_duplicates = len(experiments) - len(unique_experiments)
    print(f"Considering {len(experiments)} experiments for duplicates.")
    print(f"{num_duplicates} duplicate experiments detected.")
    print(f"{len(unique_experiments)} new experiments to add.")
    
    if not unique_experiments:
        print("No new experiments to add.")
        return
    
    # Add unique experiments to the queue
    pending_experiments.extend(unique_experiments)
    save_json(PENDING_EXPERIMENTS_FILE, pending_experiments)
    
    print(f"Added {len(unique_experiments)} experiments to the queue.")
    print(f"{len(pending_experiments)} total experiments in queue.")


def replace_queue(experiments):
    """Replace the current queue with new experiments."""
    # Load completed experiments for duplicate checking
    completed_experiments = safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    
    # Check for duplicates against completed experiments
    unique_experiments = check_for_duplicates(experiments, completed_experiments)
    
    # Print duplicate stats
    num_duplicates = len(experiments) - len(unique_experiments)
    print(f"Considering {len(experiments)} experiments for duplicates.")
    print(f"{num_duplicates} duplicate experiments with completed experiments detected.")
    
    # Replace the queue
    save_json(PENDING_EXPERIMENTS_FILE, unique_experiments)
    
    print(f"Queue replaced with {len(unique_experiments)} experiments.")


def main():
    """Main function to manage the experiment queue."""
    args = parse_arguments()
    
    # Handle clear flag
    if args.clear:
        clear_queue()
        return
    
    # Handle list flag
    if args.list:
        list_queue()
        return
    
    # Generate new experiments
    params = process_args(args)
    experiments = generate_experiments(params)
    
    # Handle replace flag
    if args.replace:
        replace_queue(experiments)
    else:
        add_to_queue(experiments)


if __name__ == "__main__":
    main()