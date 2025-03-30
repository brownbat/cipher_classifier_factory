'''
some floating point errors with learning rate, maybe store as two ints, number and exponent

exploration is key
to do it, we need to characterize which experiments we've run
and to understand the argument scheme


'''


# experiment_analyzer.py
import json
import os
import numpy as np
import itertools
from collections import defaultdict

# Assuming visualization.py exists with these functions
# If not, define dummy versions or implement them
try:
    from visualization import safe_json_load, filter_experiments, calculate_avg_accuracy_per_value
except ImportError:
    print("Warning: visualization.py not found. Using dummy functions.")
    def safe_json_load(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def filter_experiments(data, filters):
        # Simple dummy filter
        return data # In reality, implement filtering logic

    def calculate_avg_accuracy_per_value(data, param_key):
        param_values = defaultdict(list)
        for exp in data:
            if 'hyperparams' in exp and param_key in exp['hyperparams'] and \
               'metrics' in exp and 'val_accuracy' in exp['metrics'] and exp['metrics']['val_accuracy']:
                val = exp['hyperparams'][param_key]
                # Handle potential non-hashable types like lists if necessary, though unlikely for these params
                try:
                    param_values[val].append(exp['metrics']['val_accuracy'][-1])
                except TypeError:
                     print(f"Warning: Could not use value {val} for parameter {param_key} as dict key.") # Should not happen for defined params
                     pass


        avg_accuracies = {
            val: sum(accs) / len(accs)
            for val, accs in param_values.items() if accs
        }
        return avg_accuracies

# --- Analysis Functions (largely unchanged, but ensure they return data) ---

def get_parameter_averages(filtered_data):
    """Calculate accuracy averages for each parameter value"""
    params = [
        'learning_rate', 'd_model', 'nhead', 'num_encoder_layers',
        'dim_feedforward', 'batch_size', 'dropout_rate', 'epochs'
    ]
    param_averages = {}
    print("Parameter Averages:")
    print("===================")
    for param in params:
        avg_accuracy = calculate_avg_accuracy_per_value(filtered_data, param)
        if avg_accuracy:
            param_averages[param] = avg_accuracy
            print(f"\n{param}:")
            sorted_items = sorted(avg_accuracy.items(), key=lambda item: item[0]) # Sort by value
            # Optional: Sort by accuracy instead: sorted(avg_accuracy.items(), key=lambda item: item[1], reverse=True)
            for value, accuracy in sorted_items:
                print(f"  {value}: {accuracy:.4f}")

    return param_averages


def get_top_models(filtered_data, top_n=5):
    """Return the top N models by validation accuracy"""
    # Sort by validation accuracy (using the last value in the list)
    sorted_data = sorted(
        filtered_data,
        key=lambda x: x.get('metrics', {}).get('val_accuracy', [0])[-1], # Safer access
        reverse=True
    )
    top_models = sorted_data[:top_n]

    print("\n\nTop Models by Accuracy:")
    print("======================")
    print("\nExp ID | Accuracy | LR       | d_model | nhead | layers | FF dim  | batch | dropout")
    print("------------------------------------------------------------------------------") # Adjusted length
    for model in top_models:
        exp_id = model.get('experiment_id', 'unknown')
        accuracy = model.get('metrics', {}).get('val_accuracy', [0])[-1]
        hyperparams = model.get('hyperparams', {})
        lr = hyperparams.get('learning_rate', 'N/A')
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        layers = hyperparams.get('num_encoder_layers', 'N/A')
        ff_dim = hyperparams.get('dim_feedforward', 'N/A')
        batch = hyperparams.get('batch_size', 'N/A')
        dropout = hyperparams.get('dropout_rate', 'N/A')

        # Format for alignment
        print(f"{str(exp_id)[:6]:<6} | {accuracy:.4f}   | {str(lr):<8} | {str(d_model):<7} | {str(nhead):<5} | {str(layers):<6} | {str(ff_dim):<7} | {str(batch):<5} | {str(dropout):<7}")

    return top_models


def calc_improvement(model, num_epochs=5):
    """Calculate improvement in validation accuracy over the last N epochs"""
    val_acc = model.get('metrics', {}).get('val_accuracy', [])
    if len(val_acc) < num_epochs + 1:
        return 0 # Not enough data

    # Ensure values are numeric
    try:
        # Use max over last few epochs vs max over previous few for robustness? Or just end vs start_of_window
        # Current: End vs N epochs ago
        return float(val_acc[-1]) - float(val_acc[-num_epochs-1])
    except (TypeError, ValueError):
        return 0 # Handle non-numeric data if necessary


def get_learning_curve_analysis(filtered_data, top_n=5):
    """Identify models that were still improving significantly at the end of training"""
    # Calculate improvement for all models
    for model in filtered_data:
         model['improvement_last_5'] = calc_improvement(model, 5)

    # Sort by improvement
    improving_models = sorted(
        filtered_data,
        key=lambda x: x.get('improvement_last_5', 0),
        reverse=True
    )
    # Filter out models with no improvement or very little
    improving_models = [m for m in improving_models if m.get('improvement_last_5', 0) > 0.001][:top_n] # Threshold avoids near-zero noise

    print("\n\nLearning Curve Analysis:")
    print("=======================")
    print("\nModels with most improvement in final 5 epochs (candidates for longer training):")
    print("\nExp ID | Improvement | Final Acc | LR       | d_model | nhead | layers | FF dim")
    print("-------------------------------------------------------------------------------") # Adjusted length
    for model in improving_models:
        exp_id = model.get('experiment_id', 'unknown')
        improvement = model.get('improvement_last_5', 0)
        final_acc = model.get('metrics', {}).get('val_accuracy', [0])[-1]
        hyperparams = model.get('hyperparams', {})
        lr = hyperparams.get('learning_rate', 'N/A')
        d_model = hyperparams.get('d_model', 'N/A')
        nhead = hyperparams.get('nhead', 'N/A')
        layers = hyperparams.get('num_encoder_layers', 'N/A')
        ff_dim = hyperparams.get('dim_feedforward', 'N/A')

        print(f"{str(exp_id)[:6]:<6} | {improvement:+.4f}    | {final_acc:.4f}    | {str(lr):<8} | {str(d_model):<7} | {str(nhead):<5} | {str(layers):<6} | {str(ff_dim):<7}")

    return improving_models


def get_parameter_combinations(filtered_data, min_experiments=2, top_n=5):
    """Analyze which parameter combinations perform best"""
    print("\n\nPromising Parameter Combinations:")
    print("===============================")

    # Define core parameters for combination analysis
    key_params = ['learning_rate', 'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward'] # Added dim_feedforward

    combinations = defaultdict(list)
    for exp in filtered_data:
        hyperparams = exp.get('hyperparams', {})
        # Create a tuple of values for the key parameters, using 'N/A' if a param is missing
        combo = tuple(hyperparams.get(param, 'N/A') for param in key_params)
        if 'metrics' in exp and 'val_accuracy' in exp['metrics'] and exp['metrics']['val_accuracy']:
            combinations[combo].append(exp['metrics']['val_accuracy'][-1]) # Store final accuracy

    # Filter for combinations with enough experiments
    valid_combinations = {
        combo: accs for combo, accs in combinations.items()
        if len(accs) >= min_experiments and 'N/A' not in combo # Ensure all key params were present
    }

    # Calculate average accuracy and standard deviation for each valid combination
    combo_stats = {}
    for combo, accs in valid_combinations.items():
        if accs:
            avg_acc = np.mean(accs)
            std_dev = np.std(accs)
            combo_stats[combo] = {'avg_acc': avg_acc, 'std_dev': std_dev, 'num_exps': len(accs)}

    # Get top combinations by average accuracy
    # Sort items (combo, stats_dict) by avg_acc in the stats_dict
    sorted_combos = sorted(combo_stats.items(), key=lambda item: item[1]['avg_acc'], reverse=True)
    top_combinations = sorted_combos[:top_n]

    print(f"\nTop combinations (avg acc over >= {min_experiments} experiments):")
    # Dynamically generate header based on key_params
    header = " | ".join([p.replace('_', ' ').replace('num ', '')[:7].strip() for p in key_params]) # Shorten names
    print(f"\n{header:<45} | Avg Acc  | Std Dev  | # Exps")
    print("-" * (45 + 1 + 10 + 1 + 10 + 1 + 7)) # Adjust line length dynamically

    results_for_recommendations = []
    for combo, stats in top_combinations:
        # Create a dict mapping param names to values for easier use later
        combo_dict = dict(zip(key_params, combo))
        results_for_recommendations.append(combo_dict)

        # Format combo values for printing
        formatted_combo = " | ".join([f"{str(v):<7}" for v in combo])
        print(f"{formatted_combo:<45} | {stats['avg_acc']:.4f}   | {stats['std_dev']:.4f}   | {stats['num_exps']}")

    # Also identify combinations with low std dev (stable performance) - might be interesting
    stable_combinations = sorted(combo_stats.items(), key=lambda item: item[1]['std_dev'])[:top_n]
    print(f"\nMost stable combinations (lowest std dev over >= {min_experiments} experiments):")
    print(f"\n{header:<45} | Avg Acc  | Std Dev  | # Exps")
    print("-" * (45 + 1 + 10 + 1 + 10 + 1 + 7))
    for combo, stats in stable_combinations:
        formatted_combo = " | ".join([f"{str(v):<7}" for v in combo])
        print(f"{formatted_combo:<45} | {stats['avg_acc']:.4f}   | {stats['std_dev']:.4f}   | {stats['num_exps']}")


    return results_for_recommendations # Return the list of top combo dicts

# --- Enhanced Recommendations ---

def generate_suggestions(param_averages, top_models, improving_models, top_combinations, existing_experiments, num_suggestions=3):
    """
    Generates concrete hyperparameter suggestions for next experiments.

    Args:
        param_averages (dict): Dict mapping param_name -> {value: avg_accuracy}.
        top_models (list): List of top N experiment dicts.
        improving_models (list): List of experiment dicts still improving.
        top_combinations (list): List of dicts representing top hyperparam combinations.
        existing_experiments (list): List of all experiment dicts (to avoid suggesting duplicates).
        num_suggestions (int): Approx number of suggestions per category.

    Returns:
        dict: A dictionary containing lists of suggested hyperparameter dicts, categorized by strategy.
    """
    suggestions = defaultdict(list)
    suggested_configs = set() # Keep track of configs already suggested or run

    # Add existing hyperparameter sets (as tuples) to avoid duplicates
    core_params = ['learning_rate', 'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 'batch_size', 'dropout_rate'] # Define core set
    for exp in existing_experiments:
        if 'hyperparams' in exp:
             # Create a tuple of core hyperparams, using None if missing
            config_tuple = tuple(exp['hyperparams'].get(p) for p in core_params)
            suggested_configs.add(config_tuple)

    # --- Strategy 1: Combine Best Average Values (Exploitation) ---
    print("\nGenerating suggestions based on best average parameter values...")
    best_values = {}
    # Select top 1-2 values for each parameter based on average accuracy
    for param, averages in param_averages.items():
        if param in core_params and averages: # Only consider core tunable params
             # Sort values by accuracy, descending
            sorted_vals = sorted(averages.items(), key=lambda item: item[1], reverse=True)
            # Take top 1 or 2, maybe filter by accuracy threshold?
            best_values[param] = [val for val, acc in sorted_vals[:2]] # Take top 2

    # Generate combinations of these best values
    param_names = list(best_values.keys())
    value_lists = [best_values[p] for p in param_names]

    # Use itertools.product to get all combinations
    count = 0
    for combo_values in itertools.product(*value_lists):
        if count >= num_suggestions * 2: break # Limit combinations explored

        suggestion = dict(zip(param_names, combo_values))

        # Basic validation/constraints
        if 'd_model' in suggestion and 'nhead' in suggestion:
            if suggestion['d_model'] % suggestion['nhead'] != 0:
                continue # Skip invalid combination

        # Check if this configuration (or similar) has already been run or suggested
        config_tuple = tuple(suggestion.get(p) for p in core_params)
        if config_tuple not in suggested_configs:
            suggestions['best_avg_combination'].append(suggestion)
            suggested_configs.add(config_tuple)
            count += 1
            if len(suggestions['best_avg_combination']) >= num_suggestions:
                break # Stop once we have enough for this category


    # --- Strategy 2: Refine Top Models (Exploitation) ---
    print("Generating suggestions based on refining top models...")
    params_to_tweak = ['learning_rate', 'dropout_rate', 'dim_feedforward'] # Params suitable for small adjustments
    lr_factors = [0.5, 0.7, 1.5, 2.0] # Multipliers for LR
    dropout_deltas = [-0.05, +0.05] # Add/subtract from dropout
    ff_factors = [0.75, 1.5] # Scale dim_feedforward

    count = 0
    for model in top_models:
        if count >= num_suggestions: break
        base_config = model.get('hyperparams', {}).copy()
        if not base_config: continue

        # Try tweaking one parameter at a time
        for param in params_to_tweak:
             if param not in base_config: continue
             original_value = base_config[param]

             if param == 'learning_rate':
                 factors = lr_factors
                 new_values = [original_value * f for f in factors]
             elif param == 'dropout_rate':
                 deltas = dropout_deltas
                 new_values = [max(0.0, min(0.9, original_value + d)) for d in deltas] # Clamp dropout [0, 0.9]
             elif param == 'dim_feedforward':
                 factors = ff_factors
                 # Ensure feedforward dim is integer, maybe multiple of 4 or 8?
                 new_values = [int(original_value * f / 4) * 4 for f in factors] # Keep it aligned
                 new_values = [v for v in new_values if v > 0] # Must be positive

             else: # Should not happen with current params_to_tweak
                  continue

             for new_val in new_values:
                 if abs(new_val - original_value) < 1e-9: continue # Skip if value didn't change significantly

                 tweaked_config = base_config.copy()
                 tweaked_config[param] = new_val

                 # Validate nhead constraint if d_model was part of base_config
                 if 'd_model' in tweaked_config and 'nhead' in tweaked_config:
                     if tweaked_config['d_model'] % tweaked_config['nhead'] != 0:
                         continue

                 config_tuple = tuple(tweaked_config.get(p) for p in core_params)
                 if config_tuple not in suggested_configs:
                     suggestions['refine_top_models'].append(tweaked_config)
                     suggested_configs.add(config_tuple)
                     count += 1
                     if count >= num_suggestions: break # Limit suggestions per top model
             if count >= num_suggestions: break
        if count >= num_suggestions: break


    # --- Strategy 3: Extend Training for Improving Models ---
    # This is less about *new* hyperparameters and more about *continuing* runs
    print("Identifying configurations suitable for extended training...")
    for model in improving_models:
         config = model.get('hyperparams', {})
         if config:
             # Add epochs suggestion contextually
             config_tuple = tuple(config.get(p) for p in core_params)
             if config_tuple not in suggested_configs: # Avoid duplicate listing if suggested elsewhere
                 # We don't add to suggested_configs here as it's about extending, not a new run
                 suggestions['extend_training'].append(config)


    # --- Strategy 4: Explore Around Top Combinations (Exploration) ---
    print("Generating suggestions based on exploring near top combinations...")
    count = 0
    for combo_dict in top_combinations: # Use the dicts returned by get_parameter_combinations
        if count >= num_suggestions: break
        base_config = combo_dict.copy()

        # Combine with other 'best average' params not in the combo
        for param, best_vals in best_values.items():
            if param not in base_config and best_vals:
                base_config[param] = best_vals[0] # Add the single best average value

        # Try tweaking one parameter from the original combo
        params_in_combo = list(combo_dict.keys()) # Params that defined this combination
        for param in params_in_combo:
            if param not in best_values or len(best_values[param]) < 2 : continue # Need alternatives

            original_value = base_config[param]
            # Try the second-best average value for this param if available
            alternative_value = best_values[param][1] # Assumes best_values took top 2

            if abs(alternative_value - original_value) < 1e-9: continue

            tweaked_config = base_config.copy()
            tweaked_config[param] = alternative_value

            # Validation
            if 'd_model' in tweaked_config and 'nhead' in tweaked_config:
                 if tweaked_config['d_model'] % tweaked_config['nhead'] != 0:
                     continue

            config_tuple = tuple(tweaked_config.get(p) for p in core_params)
            if config_tuple not in suggested_configs:
                suggestions['explore_near_combinations'].append(tweaked_config)
                suggested_configs.add(config_tuple)
                count += 1
                if count >= num_suggestions: break
        if count >= num_suggestions: break


    return suggestions


def print_recommendations(suggestions):
    """Prints the generated suggestions clearly."""
    print("\n\nRecommendations for Next Experiments:")
    print("===================================")
    print("Based on analysis of completed runs. These are suggestions, performance not guaranteed.")
    print("Core Params considered:", ['learning_rate', 'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward', 'batch_size', 'dropout_rate']) # Show context

    total_suggestions = 0

    if suggestions['best_avg_combination']:
        print("\n--- Strategy 1: Combine Best Average Parameter Values (Exploitation) ---")
        print("   Goal: Leverage parameter values that performed well on average.")
        for i, config in enumerate(suggestions['best_avg_combination'], 1):
            print(f"   Suggestion {i}:")
            for param, value in config.items():
                print(f"     {param}: {value}")
            total_suggestions += 1
    else:
         print("\n--- Strategy 1: Combine Best Average Parameter Values ---")
         print("   No new combinations generated (maybe limited data or ran out of options).")


    if suggestions['refine_top_models']:
        print("\n--- Strategy 2: Refine Top Performing Models (Exploitation) ---")
        print("   Goal: Make small adjustments to the best configurations found so far.")
        for i, config in enumerate(suggestions['refine_top_models'], 1):
            print(f"   Suggestion {i}:")
            # Highlight the change? Difficult without knowing the base model here.
            for param, value in config.items():
                 print(f"     {param}: {value}")
            total_suggestions += 1
    else:
         print("\n--- Strategy 2: Refine Top Performing Models ---")
         print("   No new refinement suggestions generated.")


    if suggestions['extend_training']:
        print("\n--- Strategy 3: Extend Training for Improving Models ---")
        print("   Goal: Continue training models that hadn't plateaued yet.")
        for i, config in enumerate(suggestions['extend_training'], 1):
            print(f"   Suggestion {i} (Consider more epochs):")
            for param, value in config.items():
                 print(f"     {param}: {value}")
            # Note: total_suggestions not incremented as it's not strictly *new* params
    else:
         print("\n--- Strategy 3: Extend Training for Improving Models ---")
         print("   No models identified as significantly improving in the final epochs.")


    if suggestions['explore_near_combinations']:
        print("\n--- Strategy 4: Explore Around Top Parameter Combinations (Exploration) ---")
        print("   Goal: Explore the hyperparameter space near known high-performing regions.")
        for i, config in enumerate(suggestions['explore_near_combinations'], 1):
            print(f"   Suggestion {i}:")
            for param, value in config.items():
                 print(f"     {param}: {value}")
            total_suggestions += 1
    else:
         print("\n--- Strategy 4: Explore Around Top Parameter Combinations ---")
         print("   No new exploration suggestions generated near top combinations.")


    print(f"\nTotal *new* hyperparameter configurations suggested: {total_suggestions}")
    if total_suggestions == 0:
        print("\nConsider running more diverse experiments or adjusting analysis parameters (e.g., num_suggestions, thresholds).")


# --- Main Execution ---

def analyze_experiments():
    """
    Comprehensive analysis of transformer model experiments.
    Analyzes parameter effectiveness, top-performing models, and provides
    recommendations for future experiments.
    """
    # Load data
    data = []
    completed_path = 'data/completed_experiments.json'
    if os.path.exists(completed_path):
        print(f"Loading data from {completed_path}...")
        data.extend(safe_json_load(completed_path))
    else:
        print(f"Warning: {completed_path} not found. No data to analyze.")
        return

    if not data:
        print("No experiment data loaded. Exiting.")
        return

    print(f"Loaded {len(data)} experiment records.")

    # No filtering - analyze all available data for recommendations
    filtered_data = filter_experiments(data, {})
    if not filtered_data:
         print("No data remains after filtering (or filtering failed). Exiting.")
         return


    # --- Run Analysis Sections ---
    param_averages = get_parameter_averages(filtered_data)
    top_models = get_top_models(filtered_data, top_n=5)
    improving_models = get_learning_curve_analysis(filtered_data, top_n=5)
    top_combinations = get_parameter_combinations(filtered_data, min_experiments=2, top_n=5) # Returns list of dicts

    # --- Generate and Print Recommendations ---
    # Pass the results of the analyses to the suggestion generator
    suggestions = generate_suggestions(
        param_averages,
        top_models,
        improving_models,
        top_combinations,
        existing_experiments=filtered_data, # Pass all data to check against existing runs
        num_suggestions=3 # Target number of suggestions per category
    )
    print_recommendations(suggestions)


if __name__ == "__main__":
    analyze_experiments()
