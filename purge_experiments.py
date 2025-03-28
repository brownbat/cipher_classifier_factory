#!/usr/bin/env python3
"""
Utility to remove experiments from completed_experiments.json based on filter criteria,
and delete their associated files.
"""
import json
import os
import sys
import argparse
from models.common.utils import safe_json_load, convert_ndarray_to_list

def parse_filter_string(filter_str):
    """
    Parse a filter string into a structured filter dictionary.
    Format: "param1=val1,val2;param2=val3,val4"
    
    Example: "epochs=1,2;d_model=32;dropout_rate=0.1"
    Returns: {
        'hyperparams': {
            'epochs': [1, 2],
            'd_model': [32],
            'dropout_rate': [0.1]
        }
    }
    """
    filter_params = {'hyperparams': {}}
    
    # Split by semicolons to get parameter groups
    param_groups = filter_str.split(';')
    
    for group in param_groups:
        if not group.strip():
            continue
            
        # Split by equals sign to get parameter name and values
        if '=' in group:
            param, values = group.split('=', 1)
            param = param.strip()
            
            # Split values by comma
            value_strings = [v.strip() for v in values.split(',')]
            parsed_values = []
            
            for val in value_strings:
                if not val:
                    continue
                    
                try:
                    # Try to convert to appropriate type
                    if '.' in val:
                        parsed_values.append(float(val))
                    else:
                        parsed_values.append(int(val))
                except ValueError:
                    parsed_values.append(val)
            
            filter_params['hyperparams'][param] = parsed_values
    
    return filter_params

def load_filter_config(filter_arg):
    """
    Load filter configuration from a string or file.
    
    Args:
        filter_arg: Either a filter string or path to a JSON file
        
    Returns:
        Dictionary containing filter parameters
    """
    # Check if the argument is a file path
    if os.path.exists(filter_arg) and filter_arg.endswith('.json'):
        try:
            with open(filter_arg, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: Filter file '{filter_arg}' is not valid JSON")
            return {}
        except Exception as e:
            print(f"ERROR: Could not load filter file: {e}")
            return {}
    else:
        # Treat as a filter string
        try:
            return parse_filter_string(filter_arg)
        except Exception as e:
            print(f"ERROR: Could not parse filter string: {e}")
            return {}

def match_experiment(exp, filter_params):
    """
    Check if an experiment matches any of the filter criteria.
    Returns True if the experiment should be removed.
    """
    # Check hyperparameters
    if 'hyperparams' in filter_params:
        for param, filter_values in filter_params['hyperparams'].items():
            if param in exp.get('hyperparams', {}):
                if exp['hyperparams'][param] in filter_values:
                    return True
    
    # Check data parameters
    if 'data_params' in filter_params:
        for param, filter_values in filter_params['data_params'].items():
            if param in exp.get('data_params', {}):
                if exp['data_params'][param] in filter_values:
                    return True
    
    return False

def delete_experiment_files(experiment, thorough=False):
    """
    Delete files associated with an experiment.
    
    Args:
        experiment: The experiment data dictionary
        thorough: If True, delete all associated files including visualizations
                 If False, only delete model and checkpoint files
    """
    deleted_files = []
    exp_id = experiment.get('experiment_id')
    
    # Always delete model file
    model_file = experiment.get('model_filename')
    if model_file and os.path.exists(model_file):
        os.remove(model_file)
        deleted_files.append(model_file)
        print(f"Deleted: {model_file}")
    
    # Always delete metadata file
    metadata_file = experiment.get('metadata_filename')
    if metadata_file and os.path.exists(metadata_file):
        os.remove(metadata_file)
        deleted_files.append(metadata_file)
        print(f"Deleted: {metadata_file}")
    
    # Delete visualization files if thorough cleanup requested
    if thorough:
        # Delete confusion matrix file if it exists
        if 'cm_gif_filename' in experiment:
            cm_file = experiment['cm_gif_filename']
            if os.path.exists(cm_file):
                os.remove(cm_file)
                deleted_files.append(cm_file)
                print(f"Deleted: {cm_file}")
        
        # Check for loss graph files
        if exp_id:
            loss_dir = os.path.join("data", "loss_graphs")
            if os.path.exists(loss_dir):
                for filename in os.listdir(loss_dir):
                    if filename.startswith(exp_id):
                        filepath = os.path.join(loss_dir, filename)
                        os.remove(filepath)
                        deleted_files.append(filepath)
                        print(f"Deleted: {filepath}")
    
    # Always check for checkpoint files
    if exp_id:
        checkpoint_dir = os.path.join("data", "checkpoints")
        if os.path.exists(checkpoint_dir):
            for filename in os.listdir(checkpoint_dir):
                if filename.startswith(exp_id) and filename.endswith('.pt'):
                    filepath = os.path.join(checkpoint_dir, filename)
                    os.remove(filepath)
                    deleted_files.append(filepath)
                    print(f"Deleted: {filepath}")
    
    return deleted_files

def find_partial_experiments():
    """
    Find experiments with checkpoint files but not in completed_experiments.json.
    These are in-progress experiments that were interrupted.
    
    Returns a list of experiment IDs for partial experiments.
    """
    # Check for partial experiments (with checkpoint files but not completed)
    checkpoint_dir = 'data/checkpoints'
    pending_file = 'data/pending_experiments.json'
    partial_exp_ids = []
    
    # Exit if checkpoints directory doesn't exist
    if not os.path.exists(checkpoint_dir):
        return partial_exp_ids
        
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    # Get pending experiment IDs
    pending_ids = []
    if os.path.exists(pending_file):
        try:
            pending_exps = safe_json_load(pending_file)
            pending_ids = [exp.get('experiment_id') for exp in pending_exps]
        except Exception as e:
            print(f"Error reading pending experiments: {e}")
    
    # Check each checkpoint file
    for cf in checkpoint_files:
        parts = cf.split('_')
        exp_id = parts[0]
        
        # Only include if in pending queue
        if exp_id in pending_ids:
            partial_exp_ids.append(exp_id)
    
    return partial_exp_ids


def cleanup_partial_experiment(exp_id, dry_run=True):
    """
    Clean up a partial experiment by:
    1. Removing its checkpoint files
    2. Removing it from the pending_experiments.json

    Returns list of deleted files.
    """
    deleted_files = []
    
    # Remove checkpoint files
    checkpoint_dir = 'data/checkpoints'
    if os.path.exists(checkpoint_dir):
        for filename in os.listdir(checkpoint_dir):
            if filename.startswith(exp_id) and filename.endswith('.pt'):
                filepath = os.path.join(checkpoint_dir, filename)
                if not dry_run:
                    try:
                        os.remove(filepath)
                        print(f"Deleted: {filepath}")
                    except Exception as e:
                        print(f"Error deleting {filepath}: {e}")
                deleted_files.append(filepath)
    
    # Remove from pending_experiments.json
    pending_file = 'data/pending_experiments.json'
    if os.path.exists(pending_file):
        try:
            pending_exps = safe_json_load(pending_file)
            original_count = len(pending_exps)
            
            # Filter out the experiment
            pending_exps = [exp for exp in pending_exps if exp.get('experiment_id') != exp_id]
            
            # Save the updated list if it changed
            if len(pending_exps) != original_count and not dry_run:
                with open(pending_file, 'w') as f:
                    json.dump(pending_exps, f, indent=4)
                print(f"Removed experiment {exp_id} from pending queue")
        except Exception as e:
            print(f"Error updating pending experiments: {e}")
    
    return deleted_files


def main():
    parser = argparse.ArgumentParser(description='Remove experiments based on filter criteria')
    
    # Create a mode group for exclusive operation modes
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--filter', 
                      help='Filter criteria: either a filter string (e.g., "epochs=1,2;d_model=32") or path to a JSON config file')
    mode_group.add_argument('--partial', action='store_true',
                      help='Remove partial/incomplete experiments with checkpoints that have not been completed')
    
    # Options
    parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be done without making changes')
    parser.add_argument('--no-confirm', action='store_true',
                      help='Skip confirmation prompt')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show detailed information about experiments being removed')
    parser.add_argument('--inverse', action='store_true',
                      help='Invert the filter - remove experiments that do NOT match the criteria')
    parser.add_argument('--thorough', action='store_true',
                      help='Thoroughly clean up all associated visualization files')
    parser.add_argument('--count-only', action='store_true',
                      help='Show only count of affected experiments, not names')
    
    args = parser.parse_args()
    
    # Handle partial experiments mode
    if args.partial:
        print("Checking for partial/incomplete experiments...")
        partial_exps = find_partial_experiments()
        
        if not partial_exps:
            print("No partial experiments found.")
            return
            
        print(f"Found {len(partial_exps)} partial experiment(s):")
        for i, exp_id in enumerate(partial_exps, 1):
            print(f"  {i}. {exp_id}")
        
        if args.dry_run:
            print("\nDRY RUN - No changes will be made")
            return
            
        if not args.no_confirm:
            response = input("\nProceed with removing these partial experiments? (y/n): ")
            if response.lower() != 'y':
                print("Cancelled. No changes made.")
                return
                
        # Clean up partial experiments
        print("\nCleaning up partial experiments...")
        all_deleted_files = []
        for exp_id in partial_exps:
            deleted_files = cleanup_partial_experiment(exp_id, dry_run=False)
            all_deleted_files.extend(deleted_files)
            
        print(f"\nSuccess! Removed {len(partial_exps)} partial experiments and deleted {len(all_deleted_files)} files.")
        return
    
    # Handle filter mode (when --partial is not specified)
    print("Loading experiments...")
    completed_file = 'data/completed_experiments.json'
    experiments = safe_json_load(completed_file)
    print(f"Loaded {len(experiments)} experiments")
    
    # Parse filter criteria
    filter_params = load_filter_config(args.filter)
    if not filter_params:
        print("No valid filter criteria provided. Exiting.")
        sys.exit(1)
        
    print(f"Using filter criteria: {filter_params}")
    
    # Filter experiments
    to_keep = []
    to_remove = []
    
    for exp in experiments:
        matches_filter = match_experiment(exp, filter_params)
        # In normal mode, remove those that match; in inverse mode, remove those that don't match
        if (matches_filter and not args.inverse) or (not matches_filter and args.inverse):
            to_remove.append(exp)
        else:
            to_keep.append(exp)
    
    # Show results
    matching_count = len(to_remove) if not args.inverse else len(to_keep)
    non_matching_count = len(to_keep) if not args.inverse else len(to_remove)
    
    print(f"{matching_count} experiments match the filter, {non_matching_count} do not.")
    print(f"{'INVERSE' if args.inverse else 'NORMAL'} mode: Remove those that fall {'outside' if args.inverse else 'within'} the filter.")
    
    # Show a summary of the experiments to be removed
    should_show_names = args.verbose or (len(to_remove) <= 10 and not args.count_only)
    if to_remove and should_show_names:
        print("\nExperiments to remove:")
        for i, exp in enumerate(to_remove, 1):
            exp_id = exp.get('experiment_id', 'unknown')
            
            if args.verbose:
                # Show more details in verbose mode
                epochs = exp.get('hyperparams', {}).get('epochs', 'unknown')
                actual_epochs = len(exp.get('metrics', {}).get('val_accuracy', [])) if 'metrics' in exp else 0
                d_model = exp.get('hyperparams', {}).get('d_model', 'unknown')
                nhead = exp.get('hyperparams', {}).get('nhead', 'unknown')
                num_encoder_layers = exp.get('hyperparams', {}).get('num_encoder_layers', 'unknown')
                print(f"  {i}. {exp_id} (epochs: {epochs}/{actual_epochs}, d_model: {d_model}, "
                      f"nhead: {nhead}, layers: {num_encoder_layers})")
            else:
                # Just show experiment ID in normal mode
                print(f"  {i}. {exp_id}")
    
    if args.dry_run:
        print("\nDRY RUN - No changes will be made")
        return
    
    # Confirm action
    if not to_remove:
        print("No matching experiments found. Nothing to do.")
        return
    
    if not args.no_confirm:
        response = input("\nProceed with removing these experiments? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled. No changes made.")
            return
    
    # Delete files and update experiments
    print("\nDeleting associated files...")
    deleted_files = []
    for exp in to_remove:
        deleted_files.extend(delete_experiment_files(exp, thorough=args.thorough))
    
    # Save updated experiments list
    print("Saving updated experiments...")
    with open(completed_file, 'w') as f:
        to_keep = convert_ndarray_to_list(to_keep)
        json.dump(to_keep, f, indent=4)
    
    print(f"\nSuccess! Removed {len(to_remove)} experiments and deleted {len(deleted_files)} files.")

if __name__ == "__main__":
    main()