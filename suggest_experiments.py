#!/usr/bin/env python3
"""
Analyzes completed experiments and suggests new hyperparameter configurations
to explore, based on performance trends and refining top results.
Generates commands compatible with manage_queue.py.
Assumes experiments were run with early stopping and 'best' metrics are recorded.
"""

import json
import os
import numpy as np
import pandas as pd
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import argparse
import datetime
import math

# --- Configuration ---
COMPLETED_FILE = 'data/completed_experiments.json'
PENDING_FILE = 'data/pending_experiments.json'
MIN_EXPERIMENTS_FOR_ANALYSIS = 20
PRIMARY_METRIC = 'best_val_accuracy'
METRIC_MODE = 'max'
NUM_SUGGESTIONS_PER_STRATEGY = 4

# --- Helper Functions ---

def safe_json_load(file_path):
    """Load JSON file safely."""
    try:
        if not os.path.exists(file_path): return []
        with open(file_path, 'r') as f:
            content = f.read()
            if not content: return []
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {file_path}. Returning empty list.")
        return []
    except FileNotFoundError:
        print(f"Warning: File not found {file_path}. Returning empty list.")
        return []

# --- Space Definition (Learning Rate Removed!) ---
PARAM_SPACE_DEFINITION = {
    'd_model': {'type': 'int', 'range': [32, 1024], 'scale': 'linear', 'step': 32},
    'nhead': {'type': 'int', 'values': [2, 4, 8, 16], 'scale': 'linear'},
    'num_encoder_layers': {'type': 'int', 'range': [1, 10], 'scale': 'linear', 'step': 1},
    'dim_feedforward': {'type': 'int', 'range': [64, 2048], 'scale': 'linear', 'step': 64},
    'batch_size': {'type': 'int', 'values': [8, 16, 32, 64, 128, 256], 'scale': 'log'},
    'dropout_rate': {'type': 'float', 'range': [0.0, 0.5], 'scale': 'linear', 'step': 0.025},
    'base_patience': {'type': 'int', 'range': [3, 15], 'scale': 'linear', 'step': 1}, # ADDED
}

# Define constraints
def check_constraints(hyperparams: dict) -> bool:
    """Checks if a hyperparameter dictionary satisfies known constraints."""
    d_model = hyperparams.get('d_model')
    nhead = hyperparams.get('nhead')
    if d_model is not None and nhead is not None:
        if not isinstance(d_model, int) or not isinstance(nhead, int) or nhead <= 0 or d_model <= 0 or d_model % nhead != 0:
            return False
    return True

class HyperparameterSpace:
    """Class to hold and manage the hyperparameter space definition."""
    def __init__(self, definition: dict, constraints_funcs: list[callable]):
        self.parameters = definition
        self.constraints = constraints_funcs
        self.param_names = list(definition.keys()) # Canonical order

    def get_param_names(self) -> list[str]:
        """Return the names of the parameters in a fixed order."""
        return self.param_names

    def validate(self, hyperparams: dict) -> bool:
        """Validate a single hyperparameter configuration against the defined space."""
        # Validate only params present in the input config against the space definition
        for name, value in hyperparams.items():
            if name not in self.parameters:
                 # Allow hyperparameters not explicitly defined in the space (e.g., experiment_id)
                 continue

            spec = self.parameters[name]
            if value is None: # Decide if None is valid for this parameter
                 continue # Allow None for now

            # Type checking
            if spec['type'] == 'int' and not isinstance(value, int): return False
            if spec['type'] == 'float' and not isinstance(value, (float, int)): return False

            # Value/Range checking
            if 'values' in spec and value not in spec['values']: return False
            if 'range' in spec and 'values' not in spec:
                min_val, max_val = spec['range']
                try:
                    if not (min_val <= float(value) <= max_val): return False
                except (ValueError, TypeError): return False

        # Check functional constraints using all provided hyperparams
        for func in self.constraints:
            if not func(hyperparams):
                return False
        return True

    def get_nearby_values(self, param_name: str, value: Any, n: int = 2) -> List[Any]:
        """Suggest n nearby values for a parameter, respecting type, scale, bounds."""
        if param_name not in self.parameters: return []
        spec = self.parameters[param_name]
        nearby = set()

        if 'values' in spec:
            values_list = sorted(spec['values'])
            try:
                current_index = values_list.index(value)
                indices_to_check = []
                for i in range(1, n + 1):
                    if current_index - i >= 0: indices_to_check.append(current_index - i)
                    if current_index + i < len(values_list): indices_to_check.append(current_index + i)
                for idx in indices_to_check: nearby.add(values_list[idx])
            except ValueError: pass
            return sorted(list(nearby))[:n]

        if 'range' in spec:
            step = spec.get('step', None)
            min_val, max_val = spec['range']

            if spec['type'] == 'int':
                step = step or 1
                for i in range(1, n + 1):
                    lower = int(round(value - i * step))
                    upper = int(round(value + i * step))
                    if lower >= min_val: nearby.add(lower)
                    if upper <= max_val: nearby.add(upper)
            elif spec['type'] == 'float':
                if spec['scale'] == 'log':
                    factor = 1.5
                    current_lower, current_upper = value, value
                    for _ in range(n):
                        current_lower /= factor
                        current_upper *= factor
                        if current_lower >= min_val: nearby.add(current_lower)
                        if current_upper <= max_val: nearby.add(current_upper)
                else: # Linear float
                    step = step or abs(value * 0.1) or 0.1
                    for i in range(1, n + 1):
                        lower = value - i * step
                        upper = value + i * step
                        if lower >= min_val: nearby.add(lower)
                        if upper <= max_val: nearby.add(upper)

        # Filter out original value and sort
        filtered_nearby = sorted([v for v in nearby if not math.isclose(v, value, rel_tol=1e-9)])[:n]
        # Ensure correct type (mainly for int)
        if spec['type'] == 'int':
            return [int(v) for v in filtered_nearby]
        else:
            return filtered_nearby


    def get_trend_values(self, param_name: str, value: Any, direction: str, n: int = 2) -> List[Any]:
        """Suggest n values in a specific direction (increase/decrease)."""
        if param_name not in self.parameters or direction not in ['increase', 'decrease']: return []
        spec = self.parameters[param_name]
        trend_values = []

        if 'values' in spec:
            values_list = sorted(spec['values'])
            try:
                current_index = values_list.index(value)
                idx_step = 1 if direction == 'increase' else -1
                for _ in range(n):
                    next_index = current_index + idx_step
                    if 0 <= next_index < len(values_list):
                        trend_values.append(values_list[next_index])
                        current_index = next_index
                    else: break
            except ValueError: pass
            return trend_values

        if 'range' in spec:
            factor = 1.5 if spec['scale'] == 'log' else 1.0
            step = spec.get('step', 1 if spec['type'] == 'int' else abs(value * 0.1) or 0.1)
            min_val, max_val = spec['range']

            current_val = value
            added_count = 0
            while added_count < n:
                new_val = None
                if spec['type'] == 'int':
                    new_val = int(round(current_val + step if direction == 'increase' else current_val - step))
                elif spec['type'] == 'float':
                    if spec['scale'] == 'log':
                        new_val = current_val * factor if direction == 'increase' else current_val / factor
                    else:
                        new_val = current_val + step if direction == 'increase' else current_val - step

                if new_val is None: break
                if not (min_val <= new_val <= max_val): break # Out of bounds

                if not math.isclose(new_val, current_val, rel_tol=1e-9):
                    trend_values.append(new_val)
                    current_val = new_val
                    added_count += 1
                else: break # Value stagnated

        # Ensure correct type
        if spec['type'] == 'int':
            return [int(v) for v in trend_values]
        else:
            return trend_values


# --- Data Loading and Parsing ---

@dataclass
class ExperimentResult:
    """Structured representation of a completed experiment."""
    experiment_id: str
    hyperparams: Dict[str, Any]
    data_params: Dict[str, Any]
    metric_value: Optional[float]
    best_epoch: Optional[int]
    epochs_completed: Optional[int]
    training_duration: Optional[float]

def parse_experiment_data(raw_data: list[dict], space: HyperparameterSpace, metric_key: str) -> list[ExperimentResult]:
    """Parses raw experiment data, validates, and extracts the specified metric."""
    parsed_results = []
    print(f"Parsing {len(raw_data)} raw experiment records for metric '{metric_key}'...")
    # No longer need required_hyperparams check here, validation handles defined space

    for i, raw_exp in enumerate(raw_data):
        if not isinstance(raw_exp, dict): continue
        experiment_id = raw_exp.get('experiment_id', f'unknown_{i+1}')
        hyperparams = raw_exp.get('hyperparams', {})
        data_params = raw_exp.get('data_params', {})
        metrics = raw_exp.get('metrics', {})

        if not isinstance(hyperparams, dict) or not isinstance(metrics, dict): continue

        # Validate hyperparameters against the defined space
        # This now implicitly ignores params not in the space (like learning_rate)
        if not space.validate(hyperparams):
            # print(f"{experiment_id}: Hyperparameters failed validation. Params: {hyperparams}")
            continue

        # Extract specified primary metric
        metric_value = metrics.get(metric_key)
        if metric_value is None or not isinstance(metric_value, (int, float)) or not math.isfinite(metric_value):
            # print(f"{experiment_id}: Invalid or missing primary metric '{metric_key}' value '{metric_value}'.")
            continue

        best_epoch = metrics.get('best_epoch')
        epochs_completed = metrics.get('epochs_completed')
        training_duration = metrics.get('training_duration')

        result = ExperimentResult(
            experiment_id=experiment_id,
            hyperparams=hyperparams,
            data_params=data_params,
            metric_value=float(metric_value),
            best_epoch=best_epoch if isinstance(best_epoch, int) else None,
            epochs_completed=epochs_completed if isinstance(epochs_completed, int) else None,
            training_duration=training_duration if isinstance(training_duration, (int, float)) else None
        )
        parsed_results.append(result)

    print(f"Successfully parsed {len(parsed_results)} valid experiment results.")
    return parsed_results

def get_experiment_config_tuple(exp_hyperparams: dict, space: HyperparameterSpace) -> tuple:
    """Creates a canonical tuple of hyperparameters defined in the space for duplicate checking."""
    # Only include params defined in the space, in the defined order
    return tuple(exp_hyperparams.get(p) for p in space.get_param_names())

def load_and_prepare_data(space: HyperparameterSpace, metric_key: str) -> Tuple[List[ExperimentResult], Set[tuple]]:
    """Loads data, parses using the specified metric, returns results and existing configs set."""
    print(f"Loading completed experiments from: {COMPLETED_FILE}")
    raw_completed = safe_json_load(COMPLETED_FILE)
    print(f"Loading pending experiments from: {PENDING_FILE}")
    raw_pending = safe_json_load(PENDING_FILE)

    completed_results = parse_experiment_data(raw_completed, space, metric_key)

    existing_configs = set()
    # Add completed configs (using only params defined in space)
    for res in completed_results:
        if space.validate(res.hyperparams): # Ensure validation passed
             existing_configs.add(get_experiment_config_tuple(res.hyperparams, space))

    # Add pending configs (using only params defined in space)
    pending_count = 0
    for p_exp in raw_pending:
         if isinstance(p_exp, dict) and 'hyperparams' in p_exp:
            if space.validate(p_exp['hyperparams']):
                existing_configs.add(get_experiment_config_tuple(p_exp['hyperparams'], space))
                pending_count += 1

    print(f"Found {len(existing_configs)} unique valid configurations (completed + {pending_count} pending).")
    return completed_results, existing_configs

# --- Analysis Engine ---

def analyze_parameter_trends(results: List[ExperimentResult], space: HyperparameterSpace) -> Dict[str, pd.DataFrame]:
    """Analyzes the impact of each hyperparameter (in the defined space) on the primary metric."""
    if not results: return {}
    print("\n--- Parameter Trend Analysis (Based on Best Metric per Run) ---")

    data_for_df = []
    param_names_in_space = space.get_param_names() # Get params we care about
    for res in results:
        # Include only hyperparameters defined in the space for analysis consistency
        record = {p: res.hyperparams.get(p) for p in param_names_in_space}
        record['metric'] = res.metric_value
        record['experiment_id'] = res.experiment_id
        data_for_df.append(record)

    if not data_for_df: return {}
    df = pd.DataFrame(data_for_df)

    trend_analysis = {}
    for param_name in param_names_in_space: # Iterate only over params in defined space
        if param_name not in df.columns or df[param_name].isnull().all() or df[param_name].nunique() < 2:
            continue

        try:
            stats = df.groupby(param_name)['metric'].agg(['mean', 'median', 'std', 'count']).sort_index()
            reliable_stats = stats[stats['count'] > 1]
            stats_to_use = reliable_stats if not reliable_stats.empty else stats

            if not stats_to_use.empty:
                trend_analysis[param_name] = stats_to_use
                print(f"\nParameter: {param_name} (Stats based on groups with >1 sample if available)")
                print(stats_to_use.to_string(float_format="%.4f"))
        except Exception as e:
             print(f"Could not analyze {param_name}: {e}")

    return trend_analysis

def find_best_experiment(results: List[ExperimentResult], metric_mode: str) -> Optional[ExperimentResult]:
    """Finds the single best experiment based on the primary metric and mode."""
    valid_results = [r for r in results if r.metric_value is not None]
    if not valid_results: return None
    try:
        if metric_mode == 'max':
            best_exp = max(valid_results, key=lambda exp: exp.metric_value)
        else: # min mode
            best_exp = min(valid_results, key=lambda exp: exp.metric_value)
        return best_exp
    except ValueError: return None

# --- Recommendation Engine ---

def check_config(config: Dict[str, Any], space: HyperparameterSpace, existing_configs: Set[tuple]) -> bool:
    """Helper to validate a config against space/constraints and check for duplicates."""
    # Create tuple based *only* on params defined in the space for checking
    config_tuple = get_experiment_config_tuple(config, space)
    # Validate the full config (including params perhaps not in space def, like experiment_id)
    # The space.validate should handle constraints correctly.
    return config_tuple not in existing_configs and space.validate(config)

def suggest_cold_start_batches(space: HyperparameterSpace, existing_configs: Set[tuple], num_suggestions: int) -> List[Dict]:
    """Suggests diverse batches for initial exploration, respecting space definition."""
    print(f"\n--- Generating Cold Start Suggestions (Target: {num_suggestions}) ---")
    suggestions = []
    core_params = space.get_param_names()
    temp_existing = existing_configs.copy()

    # Helper to snap value (using space methods if available, simplified here)
    def snap_value(value, spec):
        if spec['type'] == 'int':
             step = spec.get('step', 1)
             snapped = round(value / step) * step
             if 'range' in spec: snapped = max(spec['range'][0], min(spec['range'][1], snapped))
             return int(snapped)
        return value # No change for float/categorical here

    # Strategy: Generate points near boundaries and center
    batch_points_per_param = defaultdict(set)
    for pname, spec in space.parameters.items():
        if 'values' in spec:
            vals = sorted(spec['values'])
            batch_points_per_param[pname].add(vals[0]) # Low boundary
            if len(vals) > 1: batch_points_per_param[pname].add(vals[-1]) # High boundary
            if len(vals) > 2: batch_points_per_param[pname].add(vals[len(vals)//2]) # Center
        elif 'range' in spec:
             low, high = spec['range']
             center = low + (high - low) * 0.5
             batch_points_per_param[pname].add(snap_value(low, spec))
             batch_points_per_param[pname].add(snap_value(high, spec))
             batch_points_per_param[pname].add(snap_value(center, spec))

    # Create combinations from these points
    param_names = list(batch_points_per_param.keys())
    value_sets = [sorted(list(batch_points_per_param[p])) for p in param_names]

    generated_count = 0
    for combo_values in itertools.product(*value_sets):
        temp_config = dict(zip(param_names, combo_values))
        # Check validity and uniqueness
        if check_config(temp_config, space, temp_existing):
             # Add as a single-experiment batch
             formatted_batch = {k: [v] for k, v in temp_config.items() if k in core_params}
             suggestions.append(formatted_batch)
             # Add its tuple to temp_existing to prevent duplicates within suggestions
             temp_existing.add(get_experiment_config_tuple(temp_config, space))
             generated_count += 1
             if generated_count >= num_suggestions * 2: # Generate a bit more to choose from later if needed
                 break # Limit number of generated raw suggestions

    # Select diverse suggestions (if more generated than needed) - simplistic selection for now
    final_suggestions = suggestions[:num_suggestions]
    print(f"Suggesting {len(final_suggestions)} diverse configurations for cold start.")
    return final_suggestions


def suggest_trend_following_batches(
    analysis: Dict[str, pd.DataFrame],
    base_config: Dict[str, Any],
    space: HyperparameterSpace,
    existing_configs: Set[tuple],
    metric_mode: str,
    num_suggestions: int
    ) -> List[Dict]:
    """Suggests experiments by moving parameters towards values with better average performance."""
    print(f"\n--- Generating Trend Following Suggestions (Target: {num_suggestions}) ---")
    suggestions = []
    core_params = space.get_param_names()
    added_configs_in_this_run = set()

    # Sort parameters by potential impact (e.g., std dev of metric across values) if possible
    # For now, just iterate through analysed params
    analysed_params = list(analysis.keys())

    for param_name in analysed_params:
        if param_name not in base_config or param_name not in space.parameters: continue
        stats_df = analysis[param_name]
        current_value = base_config[param_name]

        try: stats_df.index = stats_df.index.astype(type(current_value))
        except Exception: continue
        if current_value not in stats_df.index: continue

        sorted_stats = stats_df.sort_values('mean', ascending=(metric_mode == 'min'))
        best_val_in_stats = sorted_stats.index[0]

        direction = None
        if metric_mode == 'max':
            if best_val_in_stats > current_value: direction = 'increase'
            elif best_val_in_stats < current_value: direction = 'decrease'
        else: # min mode
             if best_val_in_stats < current_value: direction = 'decrease'
             elif best_val_in_stats > current_value: direction = 'increase'

        suggested_values = []
        if direction:
            suggested_values = space.get_trend_values(param_name, current_value, direction, n=2)
        else: # Current value is best statistically, refine around it
             suggested_values = space.get_nearby_values(param_name, current_value, n=2)


        if suggested_values:
            valid_suggestions_for_param = []
            for sugg_val in suggested_values:
                 temp_config = base_config.copy()
                 temp_config[param_name] = sugg_val
                 # Create tuple based on space definition for checking
                 config_tuple = get_experiment_config_tuple(temp_config, space)
                 if config_tuple not in added_configs_in_this_run and check_config(temp_config, space, existing_configs):
                     valid_suggestions_for_param.append(sugg_val)
                     added_configs_in_this_run.add(config_tuple)

            if valid_suggestions_for_param:
                 batch = base_config.copy()
                 batch[param_name] = sorted(valid_suggestions_for_param)
                 # Format lists, ensure only core params included
                 formatted_batch = {k: ([v] if not isinstance(v, list) else v) for k, v in batch.items() if k in core_params}
                 suggestions.append(formatted_batch)

        if len(suggestions) >= num_suggestions: break

    print(f"Generated {len(suggestions)} trend-following suggestions.")
    return suggestions


def suggest_refinement_batches(
    best_config: Dict[str, Any],
    space: HyperparameterSpace,
    existing_configs: Set[tuple],
    num_suggestions: int
    ) -> List[Dict]:
    """Suggests small variations around the best known configuration."""
    print(f"\n--- Generating Refinement Suggestions (Near Best Run) (Target: {num_suggestions}) ---")
    suggestions = []
    core_params = space.get_param_names()
    added_configs_in_this_run = set()

    for param_name in core_params:
        if param_name not in best_config or param_name not in space.parameters: continue

        current_value = best_config[param_name]
        nearby_values = space.get_nearby_values(param_name, current_value, n=2)

        if nearby_values:
            valid_suggestions_for_param = []
            for sugg_val in nearby_values:
                temp_config = best_config.copy()
                temp_config[param_name] = sugg_val
                config_tuple = get_experiment_config_tuple(temp_config, space)
                if config_tuple not in added_configs_in_this_run and check_config(temp_config, space, existing_configs):
                    valid_suggestions_for_param.append(sugg_val)
                    added_configs_in_this_run.add(config_tuple)

            if valid_suggestions_for_param:
                 batch = best_config.copy()
                 batch[param_name] = sorted(valid_suggestions_for_param)
                 formatted_batch = {k: ([v] if not isinstance(v, list) else v) for k, v in batch.items() if k in core_params}
                 suggestions.append(formatted_batch)

        if len(suggestions) >= num_suggestions: break

    print(f"Generated {len(suggestions)} refinement suggestions.")
    return suggestions

# --- Output Formatting ---

def format_batch_for_queue(batch_dict: dict, strategy_name: str) -> str:
    """ Formats a batch dictionary into a manage_queue.py command string. """
    command = f"python manage_queue.py"
    # Use sorted keys from the batch dict itself
    sorted_items = sorted(batch_dict.items())
    has_params = False
    for param, values in sorted_items:
        if not values: continue
        has_params = True
        values_str = []
        for v in values:
             if isinstance(v, float):
                 # Use scientific notation only for very small magnitudes
                 if abs(v) > 1e-9 and abs(v) < 1e-4:
                     values_str.append(f"{v:.1e}")
                 else:
                     formatted_v = f"{v:.8f}".rstrip('0').rstrip('.') if '.' in str(v) else str(v)
                     values_str.append(formatted_v if formatted_v != '-0' else '0') # Handle potential "-0"
             else:
                 values_str.append(str(v))
        command += f" --{param} {','.join(values_str)}"

    if not has_params: return ""
    command += f"  # Suggestion based on: {strategy_name}"
    return command

def format_recommendations_for_queue(recommendations: Dict[str, List[Dict]]) -> Tuple[str, int]:
    """ Formats all suggested batches into command-line strings and counts unique commands. """
    output_lines = []
    unique_commands = set()

    for strategy, batches in recommendations.items():
        strategy_lines = []
        for i, batch in enumerate(batches):
             # Ensure only params defined in the space are included in the command
             space_params = PARAM_SPACE_DEFINITION.keys()
             filtered_batch = {k: v for k, v in batch.items() if k in space_params}
             if not filtered_batch: continue # Skip if batch becomes empty after filtering

             command = format_batch_for_queue(filtered_batch, strategy)
             if command and command not in unique_commands:
                 strategy_lines.append(command)
                 unique_commands.add(command)

        if strategy_lines:
            output_lines.append(f"\n--- Strategy: {strategy} ---")
            output_lines.extend(strategy_lines)

    num_unique = len(unique_commands)
    if num_unique > 0:
        output_lines.insert(0, f"--- Recommended Batches for manage_queue.py ({num_unique} unique commands) ---")
    else:
        output_lines.append("--- No unique experiment suggestions generated ---")

    return "\n".join(output_lines), num_unique

# --- Main Orchestration ---

def main():
    parser = argparse.ArgumentParser(
        description="Analyze completed experiments and suggest new hyperparameter configurations."
    )
    parser.add_argument('--metric', type=str, default=PRIMARY_METRIC, help=f"Metric key (default: {PRIMARY_METRIC})")
    parser.add_argument('--mode', choices=['min', 'max'], default=METRIC_MODE, help=f"Optimization mode (default: {METRIC_MODE})")
    parser.add_argument('--min_exp', type=int, default=MIN_EXPERIMENTS_FOR_ANALYSIS, help=f"Min experiments for analysis (default: {MIN_EXPERIMENTS_FOR_ANALYSIS})")
    parser.add_argument('--num_sugg', type=int, default=NUM_SUGGESTIONS_PER_STRATEGY, help=f"Target suggestions per strategy (default: {NUM_SUGGESTIONS_PER_STRATEGY})")
    parser.add_argument('--output', type=str, default=None, help="Optional file path to save suggestions.")
    args = parser.parse_args()

    print("--- Starting Experiment Suggestion ---")
    print(f"Optimizing Metric: '{args.metric}' (Mode: {args.mode})")
    print(f"Min experiments for analysis: {args.min_exp}")
    print(f"Target suggestions per strategy: {args.num_sugg}")

    # Initialize space and load data using specified metric
    space = HyperparameterSpace(PARAM_SPACE_DEFINITION, [check_constraints])
    completed_results, existing_configs = load_and_prepare_data(space, args.metric)

    recommendations = defaultdict(list)

    # Decide strategy based on data availability
    if len(completed_results) < args.min_exp:
        print(f"\nInsufficient data ({len(completed_results)} valid runs) for full analysis.")
        cold_start_batches = suggest_cold_start_batches(space, existing_configs, args.num_sugg)
        recommendations['cold_start'] = cold_start_batches
    else:
        print(f"\nSufficient data ({len(completed_results)} valid runs) for analysis.")
        best_experiment = find_best_experiment(completed_results, args.mode)

        if not best_experiment:
            print("\nError: Could not identify a best-performing experiment.")
            cold_start_batches = suggest_cold_start_batches(space, existing_configs, args.num_sugg)
            recommendations['cold_start'] = cold_start_batches
        else:
            # Display best experiment details (using only params in the defined space for clarity)
            print(f"\nBest experiment found: {best_experiment.experiment_id} ({args.metric}: {best_experiment.metric_value:.4f})")
            best_params_in_space = {k: v for k, v in best_experiment.hyperparams.items() if k in space.get_param_names()}
            best_params_str = ", ".join(f"{k}={v}" for k, v in sorted(best_params_in_space.items()))
            print(f"  Best Params (in space): {best_params_str}")
            if best_experiment.best_epoch: print(f"  Achieved at epoch: {best_experiment.best_epoch}")

            analysis_results = analyze_parameter_trends(completed_results, space)
            core_params_in_space = space.get_param_names()

            # Generate suggestions, updating existing_configs after each strategy
            if analysis_results:
                trend_batches = suggest_trend_following_batches(
                    analysis_results, best_experiment.hyperparams, space, existing_configs, args.mode, args.num_sugg
                )
                recommendations['trend_following'] = trend_batches
                for batch in trend_batches:
                     param_names = list(batch.keys())
                     value_lists = [batch[p] for p in param_names]
                     for combo_values in itertools.product(*value_lists):
                         temp_config = dict(zip(param_names, combo_values))
                         # Ensure all core params are present for tuple generation
                         full_config = {p: temp_config.get(p) for p in core_params_in_space}
                         existing_configs.add(get_experiment_config_tuple(full_config, space))

            refinement_batches = suggest_refinement_batches(
                best_experiment.hyperparams, space, existing_configs, args.num_sugg
            )
            recommendations['refinement'] = refinement_batches
            for batch in refinement_batches:
                 param_names = list(batch.keys())
                 value_lists = [batch[p] for p in param_names]
                 for combo_values in itertools.product(*value_lists):
                     temp_config = dict(zip(param_names, combo_values))
                     full_config = {p: temp_config.get(p) for p in core_params_in_space}
                     existing_configs.add(get_experiment_config_tuple(full_config, space))

    # Format and Output Results
    queue_commands, num_commands = format_recommendations_for_queue(recommendations)
    print("\n" + queue_commands)

    if num_commands > 0:
        if args.output:
             output_filename = args.output
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "data/param_analysis"
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"suggestions_{timestamp}.txt")

        try:
            with open(output_filename, "w") as f:
                f.write(f"# Experiment suggestions generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Optimizing: {args.metric} ({args.mode})\n")
                f.write(queue_commands)
            print(f"\nSuggestions also saved to: {output_filename}")
        except Exception as e:
            print(f"\nError saving suggestions to file {output_filename}: {e}")
    else:
        print("\nNo new unique experiment suggestions were generated.")

    print("\n--- Suggestion process complete ---")

if __name__ == "__main__":
    main()
