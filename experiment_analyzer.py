import json
import os
import numpy as np
from visualization import safe_json_load, filter_experiments, calculate_avg_accuracy_per_value

def analyze_experiments():
    """
    Comprehensive analysis of transformer model experiments.
    Analyzes parameter effectiveness, top-performing models, and provides
    recommendations for future experiments.
    """
    # Load data from completed experiments
    data = []
    
    # Load the main completed experiments file
    if os.path.exists('data/completed_experiments.json'):
        data.extend(safe_json_load('data/completed_experiments.json'))
    
    # No filtering - we want to see all parameters
    filtered_data = filter_experiments(data, {})
    
    # Parameter averages section
    print_parameter_averages(filtered_data)
    
    # Top models section
    print_top_models(filtered_data)
    
    # Learning curve analysis
    print_learning_curve_analysis(filtered_data)
    
    # Parameter combination analysis
    print_parameter_combinations(filtered_data)
    
    # Print recommendations
    print_recommendations(filtered_data)


def print_parameter_averages(filtered_data):
    """Calculate and print accuracy averages for each parameter value"""
    params = [
        'learning_rate', 'd_model', 'nhead', 'num_encoder_layers', 
        'dim_feedforward', 'batch_size', 'dropout_rate', 'epochs'
    ]
    
    print("Parameter Averages:")
    print("===================")
    
    param_averages = {}
    for param in params:
        avg_accuracy = calculate_avg_accuracy_per_value(filtered_data, param)
        if avg_accuracy:
            param_averages[param] = avg_accuracy
            print(f"\n{param}:")
            # Sort by parameter value for better readability
            for value, accuracy in sorted(avg_accuracy.items()):
                print(f"  {value}: {accuracy:.3f}")
    
    return param_averages


def print_top_models(filtered_data, top_n=5):
    """Print the top N models by validation accuracy"""
    # Sort by validation accuracy (using the last value in the list)
    sorted_data = sorted(
        filtered_data, 
        key=lambda x: x['metrics']['val_accuracy'][-1] if x['metrics']['val_accuracy'] else 0, 
        reverse=True
    )
    
    # Take the top N models
    top_models = sorted_data[:top_n]
    
    print("\n\nTop Models by Accuracy:")
    print("======================")
    
    # Print header
    print("\nExp ID | Accuracy | LR     | d_model | nhead | layers | FF dim  | batch | dropout")
    print("----------------------------------------------------------------------")
    
    for model in top_models:
        exp_id = model.get('experiment_id', 'unknown')
        accuracy = model['metrics']['val_accuracy'][-1] if model['metrics']['val_accuracy'] else 0
        
        # Extract key hyperparameters
        hyperparams = model.get('hyperparams', {})
        lr = hyperparams.get('learning_rate', 'N/A')
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        layers = hyperparams.get('num_encoder_layers', 'N/A')
        ff_dim = hyperparams.get('dim_feedforward', 'N/A')
        batch = hyperparams.get('batch_size', 'N/A')
        dropout = hyperparams.get('dropout_rate', 'N/A')
        
        print(f"{exp_id:6} | {accuracy:.4f} | {lr:6} | {d_model:6} | {nhead:5} | {layers:6} | {ff_dim:6} | {batch:5} | {dropout}")


def calc_improvement(model, num_epochs=5):
    """Calculate improvement in validation accuracy over the last N epochs"""
    if 'val_accuracy' not in model['metrics'] or len(model['metrics']['val_accuracy']) < num_epochs + 1:
        return 0
    
    values = model['metrics']['val_accuracy']
    # Compare current value with value num_epochs ago
    return values[-1] - values[-num_epochs-1]


def print_learning_curve_analysis(filtered_data):
    """Identify models that were still improving at the end of training"""
    print("\n\nLearning Curve Analysis:")
    print("=======================")
    
    # Sort by improvement in last 5 epochs
    improving_models = sorted(
        filtered_data,
        key=lambda x: calc_improvement(x, 5),
        reverse=True
    )[:5]
    
    print("\nModels with most improvement in final 5 epochs (still learning):")
    print("\nExp ID | Improvement | Final Acc | LR     | d_model | nhead | layers | FF dim")
    print("------------------------------------------------------------------------")
    
    for model in improving_models:
        exp_id = model.get('experiment_id', 'unknown')
        improvement = calc_improvement(model, 5)
        final_acc = model['metrics']['val_accuracy'][-1] if model['metrics']['val_accuracy'] else 0
        
        # Extract key hyperparameters
        hyperparams = model.get('hyperparams', {})
        lr = hyperparams.get('learning_rate', 'N/A')
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        layers = hyperparams.get('num_encoder_layers', 'N/A')
        ff_dim = hyperparams.get('dim_feedforward', 'N/A')
        
        print(f"{exp_id:6} | {improvement:+.4f}    | {final_acc:.4f}  | {lr:6} | {d_model:6} | {nhead:5} | {layers:6} | {ff_dim:6}")
    
    return improving_models


def print_parameter_combinations(filtered_data):
    """Analyze which parameter combinations perform best"""
    print("\n\nPromising Parameter Combinations:")
    print("===============================")
    
    # Analyze combinations of key parameters
    key_params = ['learning_rate', 'd_model', 'nhead', 'num_encoder_layers']
    
    # Get combinations with at least 2 experiments
    combinations = {}
    for exp in filtered_data:
        hyperparams = exp.get('hyperparams', {})
        combo = tuple(hyperparams.get(param, 'N/A') for param in key_params)
        if combo not in combinations:
            combinations[combo] = []
        combinations[combo].append(exp)
    
    # Filter to combinations with at least 2 experiments
    valid_combinations = {k: v for k, v in combinations.items() if len(v) >= 2}
    
    # Calculate average accuracy for each combination
    combo_accuracies = {}
    for combo, exps in valid_combinations.items():
        accs = [exp['metrics']['val_accuracy'][-1] for exp in exps if exp['metrics']['val_accuracy']]
        if accs:
            combo_accuracies[combo] = sum(accs) / len(accs)
    
    # Get top combinations
    top_combinations = sorted(combo_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
    
    print("\nTop parameter combinations:")
    print("\nLR     | d_model | nhead | layers | Avg Accuracy | # Exps")
    print("------------------------------------------------------------")
    
    for combo, acc in top_combinations:
        lr, d_model, nhead, layers = combo
        num_exps = len(valid_combinations[combo])
        print(f"{lr:6} | {d_model:6} | {nhead:5} | {layers:6} | {acc:.4f}      | {num_exps}")
    
    return top_combinations


def print_recommendations(filtered_data):
    """Generate recommendations for future experiments based on analysis"""
    # Get parameter averages
    param_averages = {}
    params = ['learning_rate', 'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward']
    for param in params:
        param_averages[param] = calculate_avg_accuracy_per_value(filtered_data, param)
    
    # Get improving models
    improving_models = sorted(
        filtered_data,
        key=lambda x: calc_improvement(x, 5),
        reverse=True
    )[:3]
    
    print("\n\nRecommendations for New Experiments:")
    print("==================================")
    
    # Best learning rate
    best_lr = max(param_averages['learning_rate'].items(), key=lambda x: x[1])[0]
    print(f"• Learning rate: {best_lr} (significantly outperforms other values)")
    
    # Best d_model values
    d_model_items = sorted(param_averages['d_model'].items(), key=lambda x: x[1], reverse=True)
    best_d_models = [val for val, acc in d_model_items if acc > 0.5]
    print(f"• d_model: {', '.join(map(str, best_d_models))}")
    
    # Best nhead values
    nhead_items = sorted(param_averages['nhead'].items(), key=lambda x: x[1], reverse=True)
    best_nheads = [val for val, acc in nhead_items if acc > 0.5]
    print(f"• nhead: {', '.join(map(str, best_nheads))}")
    
    # Best num_encoder_layers
    layers_items = sorted(param_averages['num_encoder_layers'].items(), key=lambda x: x[1], reverse=True)
    best_layers = [val for val, acc in layers_items if acc > 0.5]
    print(f"• num_encoder_layers: {', '.join(map(str, best_layers))}")
    
    # Prioritize experiments still improving
    print("\nFor extended epochs, focus on these configurations that were still improving:")
    for i, model in enumerate(improving_models, 1):
        hyperparams = model.get('hyperparams', {})
        config = {
            'lr': hyperparams.get('learning_rate'),
            'd_model': hyperparams.get('d_model'),
            'nhead': hyperparams.get('nhead'),
            'layers': hyperparams.get('num_encoder_layers'),
            'dim_ff': hyperparams.get('dim_feedforward')
        }
        print(f"{i}. LR={config['lr']}, d_model={config['d_model']}, nhead={config['nhead']}, layers={config['layers']}, dim_ff={config['dim_ff']}")


if __name__ == "__main__":
    analyze_experiments()