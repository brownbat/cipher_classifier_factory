#!/usr/bin/env python3
"""
Simple experiment suggestion utility that recommends new hyperparameters
based on the best performing experiment. Uses a boundary exploration approach.
"""

import argparse
import json
from datetime import datetime
import os


def safe_json_load(file_path):
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"WARNING: {file_path} is empty or invalid. An empty dict will be used.")
        return {}
    except FileNotFoundError:
        print(f"WARNING: {file_path} not found. Creating a new file.")
        with open(file_path, 'w') as file:
            json.dump({}, file)
        return {}


def find_best_experiment(experiments, metric='val_accuracy'):
    """Find the best performing experiment based on the specified metric."""
    best_exp_id = None
    best_performance = float('-inf') if metric != 'val_loss' else float('inf')
    best_experiment = None
    
    # Handle different experiment data formats
    if isinstance(experiments, list):
        for exp in experiments:
            if 'metrics' not in exp or metric not in exp['metrics']:
                continue
                
            # Get metric value (handle list metrics)
            value = exp['metrics'][metric]
            if isinstance(value, list) and value:
                value = value[-1]
                
            # For loss metrics, lower is better
            if metric == 'val_loss':
                if value < best_performance:
                    best_performance = value
                    best_exp_id = exp.get('experiment_id', exp.get('id', 'unknown'))
                    best_experiment = exp
            # For other metrics like accuracy, higher is better
            else:
                if value > best_performance:
                    best_performance = value
                    best_exp_id = exp.get('experiment_id', exp.get('id', 'unknown'))
                    best_experiment = exp
    else:
        # Process as dictionary
        for exp_id, exp in experiments.items():
            if 'metrics' not in exp or metric not in exp['metrics']:
                continue
                
            # Get metric value (handle list metrics)
            value = exp['metrics'][metric]
            if isinstance(value, list) and value:
                value = value[-1]
                
            # For loss metrics, lower is better
            if metric == 'val_loss':
                if value < best_performance:
                    best_performance = value
                    best_exp_id = exp_id
                    best_experiment = exp
            # For other metrics like accuracy, higher is better
            else:
                if value > best_performance:
                    best_performance = value
                    best_exp_id = exp_id
                    best_experiment = exp
    
    return best_exp_id, best_performance, best_experiment


def suggest_next_experiments(best_experiment, max_suggestions=3):
    """Generate suggestions based on the best experiment parameters."""
    if not best_experiment or 'hyperparams' not in best_experiment:
        return []
        
    best_params = best_experiment['hyperparams']
    suggestions = []
    
    # Define parameter boundaries
    param_ranges = {
        'd_model': {'min': 32, 'max': 1024, 'type': 'int'},
        'nhead': {'min': 1, 'max': 16, 'type': 'int'},
        'num_encoder_layers': {'min': 1, 'max': 12, 'type': 'int'},
        'dim_feedforward': {'min': 64, 'max': 4096, 'type': 'int'},
        'learning_rate': {'min': 1e-6, 'max': 1e-2, 'type': 'float'},
        'batch_size': {'min': 8, 'max': 512, 'type': 'int'},
        'dropout_rate': {'min': 0.0, 'max': 0.5, 'type': 'float'}
    }
    
    # For each parameter, check if it's at a boundary and suggest going further
    for param, value in best_params.items():
        if param not in param_ranges:
            continue
            
        param_range = param_ranges[param]
        new_suggestion = best_params.copy()
        
        # Check if parameter is at upper bound
        if param in ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 'batch_size']:
            # If we're at max value of parameter range, don't suggest higher
            if value >= param_range['max']:
                continue
                
            # For parameters where higher tends to be better, suggest larger value
            # if the current value is the highest we've tried
            if value == max(value, param_range['min']):
                if param_range['type'] == 'int':
                    # For integers like d_model, nhead, multiply by 2 (but don't exceed max)
                    new_value = min(value * 2, param_range['max'])
                    # Make sure we're increasing by at least 1
                    new_value = max(new_value, value + 1)
                    new_suggestion[param] = int(new_value)
                else:
                    # For floats, increase by 50%
                    new_value = min(value * 1.5, param_range['max'])
                    new_suggestion[param] = new_value
                    
                suggestions.append({
                    'params': new_suggestion,
                    'description': f"Increase {param} from {value} to {new_suggestion[param]}"
                })
                
                # Only one suggestion per adjustment
                if len(suggestions) >= max_suggestions:
                    break
        
        # Check if parameter is at lower bound
        if param in ['learning_rate', 'dropout_rate']:
            # If we're at min value of parameter range, don't suggest lower
            if value <= param_range['min']:
                continue
                
            # For parameters where lower might be better, suggest smaller value
            # if the current value is the lowest we've tried
            if value == min(value, param_range['max']):
                if param_range['type'] == 'int':
                    # For integers, divide by 2 (but don't go below min)
                    new_value = max(value // 2, param_range['min'])
                    # Make sure we're decreasing by at least 1
                    new_value = min(new_value, value - 1)
                    new_suggestion[param] = int(new_value)
                else:
                    # For learning_rate, reduce by factor of 3
                    if param == 'learning_rate':
                        new_value = max(value / 3, param_range['min'])
                    else:
                        # For other floats, decrease by 50%
                        new_value = max(value / 2, param_range['min'])
                    new_suggestion[param] = new_value
                    
                suggestions.append({
                    'params': new_suggestion,
                    'description': f"Decrease {param} from {value} to {new_suggestion[param]}"
                })
                
                # Only one suggestion per adjustment
                if len(suggestions) >= max_suggestions:
                    break
    
    return suggestions


def save_suggestions_to_queue(suggestions, queue_file='data/pending_experiments.json', clear=False):
    """Save suggested experiments to the pending queue."""
    # Load existing queue
    queue = {} if clear else safe_json_load(queue_file)
    
    # Add suggestions to queue
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    added_count = 0
    
    for i, suggestion in enumerate(suggestions):
        # Generate experiment ID
        experiment_id = f"suggestion_{i+1}_{timestamp}"
        
        # Create experiment config
        experiment = {
            "id": experiment_id,
            "hyperparams": suggestion['params'],
            "description": suggestion['description']
        }
        
        queue[experiment_id] = experiment
        added_count += 1
    
    # Save queue
    with open(queue_file, 'w') as f:
        json.dump(queue, f, indent=2)
    
    return added_count


def main():
    parser = argparse.ArgumentParser(description="Suggest new experiments based on the best performing one")
    
    parser.add_argument('--metric', type=str, default='val_accuracy',
                      choices=['val_accuracy', 'val_loss'],
                      help="Target metric to optimize")
    parser.add_argument('--count', type=int, default=3,
                      help="Maximum number of suggestions to generate")
    parser.add_argument('--queue', action='store_true',
                      help="Add suggestions to experiment queue")
    parser.add_argument('--clear_queue', action='store_true',
                      help="Clear experiment queue before adding suggestions")
    
    args = parser.parse_args()
    
    # Load experiments
    experiments = safe_json_load('data/completed_experiments.json')
    
    if not experiments:
        print("No experiments found. Run some experiments first.")
        return
    
    # Find best experiment
    best_exp_id, best_performance, best_experiment = find_best_experiment(experiments, args.metric)
    
    if not best_experiment:
        print(f"No valid experiments found with {args.metric} metric.")
        return
    
    # Display best experiment
    print(f"\nBest experiment so far:")
    print(f"ID: {best_exp_id}")
    print(f"{args.metric}: {best_performance:.4f}")
    print("Parameters:")
    for param, value in best_experiment.get('hyperparams', {}).items():
        print(f"  {param}: {value}")
    
    # Generate suggestions
    suggestions = suggest_next_experiments(best_experiment, args.count)
    
    if not suggestions:
        print("\nNo clear suggestions available. The best experiment may be at parameter bounds.")
        return
    
    # Print suggestions
    print("\nSuggested experiments:")
    for i, suggestion in enumerate(suggestions):
        print(f"\n{i+1}. {suggestion['description']}")
        print("   Parameters:")
        for param, value in suggestion['params'].items():
            print(f"   {param}: {value}")
    
    # Add to queue if requested
    if args.queue:
        added = save_suggestions_to_queue(suggestions, clear=args.clear_queue)
        
        if args.clear_queue:
            print(f"\nCleared queue and added {added} suggested experiments")
        else:
            print(f"\nAdded {added} suggested experiments to queue")
            
        print("Run experiments with: python researcher.py")


if __name__ == "__main__":
    main()