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
# <<< CHANGE: Import utils >>>
import models.common.utils as utils

# --- Configuration ---
# <<< CHANGE: Use utils paths >>>
COMPLETED_FILE = utils.COMPLETED_EXPERIMENTS_FILE
PENDING_FILE = utils.PENDING_EXPERIMENTS_FILE

MIN_EXPERIMENTS_FOR_ANALYSIS = 10 # Reduced threshold slightly

# <<< CHANGE: Ensure PRIMARY_METRIC matches key saved in completed_experiments.json >>>
# Key 'best_val_accuracy' seems correct based on train.py/researcher.py
PRIMARY_METRIC = 'best_val_accuracy'
METRIC_MODE = 'max' # 'max' for accuracy, 'min' for loss

# <<< USER VERIFICATION NEEDED: Ensure PARAM_SPACE_DEFINITION matches actual hyperparams used/saved >>>
PARAM_SPACE_DEFINITION = {
    # Transformer parameters (align with train.py)
    'd_model': {'type': 'int', 'values': [64, 128, 256, 512], 'scale': 'linear'}, # Often powers of 2
    'nhead': {'type': 'int', 'values': [2, 4, 8, 16], 'scale': 'linear'}, # Must divide d_model
    'num_encoder_layers': {'type': 'int', 'range': [1, 8], 'scale': 'linear', 'step': 1},
    'dim_feedforward': {'type': 'int', 'values': [128, 256, 512, 1024, 2048], 'scale': 'linear'}, # Often multiple of d_model
    # Training parameters (align with train.py / researcher.py defaults/queue)
    'batch_size': {'type': 'int', 'values': [16, 32, 64, 128], 'scale': 'log'},
    'dropout_rate': {'type': 'float', 'range': [0.0, 0.5], 'scale': 'linear', 'step': 0.05}, # Wider steps
    'learning_rate': {'type': 'float', 'range': [1e-5, 1e-3], 'scale': 'log'}, # Adjusted range slightly
    'patience': {'type': 'int', 'values': [3, 5, 10, 15, 20, 25]}, # Optional to add patience here
}
NUM_SUGGESTIONS_PER_STRATEGY = 3 # Slightly reduced suggestions per strategy

# --- Helper Functions ---
# <<< REMOVE: Redundant safe_json_load, use utils.safe_json_load >>>
# def safe_json_load(file_path): ...

# Define constraints (remains the same)
def check_constraints(hyperparams: dict) -> bool:
    """Checks if a hyperparameter dictionary satisfies known constraints."""
    d_model = hyperparams.get('d_model')
    nhead = hyperparams.get('nhead')
    if d_model is not None and nhead is not None:
        if not isinstance(d_model, int) or not isinstance(nhead, int) or nhead <= 0 or d_model <= 0 or d_model % nhead != 0:
            # print(f"Constraint fail: d_model={d_model}, nhead={nhead}") # Debug print
            return False
    # Add more constraints if needed (e.g., dim_feedforward relation to d_model)
    # dim_ff = hyperparams.get('dim_feedforward')
    # if d_model is not None and dim_ff is not None and dim_ff < d_model:
    #     return False # Example: Feedforward dim usually >= d_model
    return True

# HyperparameterSpace class (remains mostly the same, maybe minor tweaks in validate/nearby)
class HyperparameterSpace:
    """Class to hold and manage the hyperparameter space definition."""
    def __init__(self, definition: dict, constraints_funcs: list[callable]):
        self.parameters = definition
        self.constraints = constraints_funcs
        self.param_names = sorted(list(definition.keys())) # Keep canonical order sorted

    def get_param_names(self) -> list[str]:
        """Return the names of the parameters in a fixed order."""
        return self.param_names

    def validate(self, hyperparams: dict) -> bool:
        """Validate a single hyperparameter configuration against the defined space."""
        # Ensure all required parameters defined in the space are present
        # for name in self.param_names:
        #     if name not in hyperparams:
        #         print(f"Validation fail: Missing required param {name} in {hyperparams}")
        #         return False # Make missing params an error

        for name, spec in self.parameters.items():
            if name not in hyperparams:
                 # If not requiring all params, skip validation for missing ones
                 continue

            value = hyperparams[name]
            if value is None:
                 # Decide if None is valid for this parameter (usually not for model dims, LR, etc.)
                 # print(f"Validation fail: Param {name} is None in hyperparams {hyperparams}")
                 return False # Assume None is generally invalid unless specified

            # Type checking
            if spec['type'] == 'int' and not isinstance(value, int):
                # print(f"Type fail: {name}={value} ({type(value)}) not int")
                return False
            if spec['type'] == 'float' and not isinstance(value, (float, int)):
                # print(f"Type fail: {name}={value} ({type(value)}) not float/int")
                return False

            # Use 'values' for categorical implicitly
            if 'values' in spec:
                # Allow approximate match for floats within 'values'
                if spec['type'] == 'float':
                    if not any(math.isclose(value, v_opt, rel_tol=1e-9) for v_opt in spec['values']):
                        # print(f"Value fail (float): {name}={value} not close to any in {spec['values']}")
                        return False
                elif value not in spec['values']: # Exact match for non-floats
                    # print(f"Value fail: {name}={value} not in {spec['values']}")
                    return False

            # Range checking (only if 'values' is not used)
            if 'range' in spec and 'values' not in spec:
                min_val, max_val = spec['range']
                try:
                    # Check bounds strictly (or with small tolerance for floats)
                    tolerance = 1e-9 if spec['type'] == 'float' else 0
                    if not (min_val - tolerance <= float(value) <= max_val + tolerance):
                        # print(f"Range fail: {name}={value} not in [{min_val}, {max_val}]")
                        return False
                except (ValueError, TypeError):
                     # print(f"Range check type error: {name}={value}")
                     return False

        # Check functional constraints
        for func in self.constraints:
            if not func(hyperparams):
                # print(f"Functional constraint fail for {hyperparams}") # Debug
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
                # Handle float comparison carefully
                current_index = -1
                if spec['type'] == 'float':
                     for idx, v_opt in enumerate(values_list):
                         if math.isclose(value, v_opt, rel_tol=1e-9):
                             current_index = idx
                             break
                else:
                    current_index = values_list.index(value)

                if current_index != -1:
                    indices_to_check = []
                    for i in range(1, n + 1):
                        if current_index - i >= 0: indices_to_check.append(current_index - i)
                        if current_index + i < len(values_list): indices_to_check.append(current_index + i)
                    for idx in indices_to_check:
                        nearby.add(values_list[idx])
            except ValueError:
                pass # Value not in list
            # <<< CHANGE: Return sorted list, limit to n >>>
            return sorted(list(nearby))[:n]


        if spec['type'] == 'int':
            step = spec.get('step', 1)
            min_val, max_val = spec.get('range', (None, None))
            min_defined, max_defined = min_val is not None, max_val is not None
            for i in range(1, n + 1):
                lower = value - i * step
                upper = value + i * step
                if not min_defined or lower >= min_val: nearby.add(lower)
                if not max_defined or upper <= max_val: nearby.add(upper)

        elif spec['type'] == 'float':
            step = spec.get('step') # Step might be defined for linear floats
            min_val, max_val = spec.get('range', (None, None))
            min_defined, max_defined = min_val is not None, max_val is not None

            if spec['scale'] == 'log':
                factor = 1.5 # Multiplicative factor
                current_lower, current_upper = value, value
                for _ in range(n):
                    current_lower /= factor
                    current_upper *= factor
                    if not min_defined or current_lower >= min_val: nearby.add(current_lower)
                    if not max_defined or current_upper <= max_val: nearby.add(current_upper)
            else: # Linear scale float
                 # Ensure step is reasonable
                step = step or abs(value * 0.1) or 1e-5 # Default step: 10% or small fixed value
                for i in range(1, n + 1):
                    lower = value - i * step
                    upper = value + i * step
                    if not min_defined or lower >= min_val: nearby.add(lower)
                    if not max_defined or upper <= max_val: nearby.add(upper)

        # Filter original value and limit results
        # <<< CHANGE: Use math.isclose for float comparison >>>
        filtered_nearby = [v for v in nearby if not math.isclose(v, value, rel_tol=1e-9)]
        return sorted(filtered_nearby)[:n]


    def get_trend_values(self, param_name: str, value: Any, direction: str, n: int = 2) -> List[Any]:
        """Suggest n values in a specific direction (increase/decrease)."""
        if param_name not in self.parameters or direction not in ['increase', 'decrease']: return []
        spec = self.parameters[param_name]
        trend_values = []

        if 'values' in spec:
            values_list = sorted(spec['values'])
            try:
                # Handle float comparison
                current_index = -1
                if spec['type'] == 'float':
                    for idx, v_opt in enumerate(values_list):
                        if math.isclose(value, v_opt, rel_tol=1e-9):
                            current_index = idx
                            break
                else:
                    current_index = values_list.index(value)

                if current_index != -1:
                    idx_step = 1 if direction == 'increase' else -1
                    for _ in range(n):
                        next_index = current_index + idx_step
                        if 0 <= next_index < len(values_list):
                            trend_values.append(values_list[next_index])
                            current_index = next_index
                        else:
                            break
            except ValueError:
                 pass
            return trend_values

        # Handle 'range' based parameters
        factor = 1.5 if spec['scale'] == 'log' else 1.0
        step = spec.get('step', 1 if spec['type'] == 'int' else abs(value * 0.1) or 1e-5)
        min_val, max_val = spec.get('range', (None, None))
        min_defined, max_defined = min_val is not None, max_val is not None

        current_val = value
        added_count = 0
        while added_count < n:
            new_val = None
            op_sign = 1 if direction == 'increase' else -1

            if spec['type'] == 'int':
                 new_val = int(round(current_val + op_sign * step))
            elif spec['type'] == 'float':
                if spec['scale'] == 'log':
                     new_val = current_val * (factor ** op_sign)
                else: # Linear float
                    new_val = current_val + op_sign * step

            if new_val is None: break

            # Check bounds (with tolerance for floats)
            tolerance = 1e-9 if spec['type'] == 'float' else 0
            in_bounds = True
            if min_defined and new_val < min_val - tolerance: in_bounds = False
            if max_defined and new_val > max_val + tolerance: in_bounds = False
            # Clamp to bounds if slightly outside due to float math? Optional.
            # if min_defined and new_val < min_val: new_val = min_val
            # if max_defined and new_val > max_val: new_val = max_val

            if not in_bounds: break

            # Check if significantly different from the last value added or the original
            # <<< CHANGE: Use math.isclose for float comparison >>>
            is_different = not math.isclose(new_val, current_val, rel_tol=1e-9)
            if is_different:
                 trend_values.append(new_val)
                 current_val = new_val
                 added_count += 1
            else:
                # Value stagnated (e.g., step too small, reached boundary precisely)
                break

        return trend_values

# --- Data Loading and Parsing ---

@dataclass
class ExperimentResult:
    """Structured representation of a completed experiment."""
    experiment_id: str
    hyperparams: Dict[str, Any]
    data_params: Dict[str, Any]
    metric_value: Optional[float] # Value of the PRIMARY_METRIC
    # <<< CHANGE: Add other metrics if needed for analysis/reporting >>>
    best_epoch: Optional[int] = None
    epochs_completed: Optional[int] = None
    training_duration: Optional[float] = None


def parse_experiment_data(raw_data: list[dict], space: HyperparameterSpace, metric_key: str) -> list[ExperimentResult]:
    """
    Parses raw experiment data, validates hyperparameters, and extracts the primary metric.
    """
    parsed_results = []
    print(f"Parsing {len(raw_data)} raw experiment records for metric '{metric_key}'...")
    required_hyperparams = set(space.get_param_names())

    for i, raw_exp in enumerate(raw_data):
        if not isinstance(raw_exp, dict): continue

        experiment_id = raw_exp.get('experiment_id', f'unknown_{i+1}')
        hyperparams = raw_exp.get('hyperparams', {})
        data_params = raw_exp.get('data_params', {})
        metrics = raw_exp.get('metrics', {})

        if not isinstance(hyperparams, dict) or not isinstance(metrics, dict): continue

        # --- Validate Hyperparameters against defined space ---
        # Only validate experiments where *all* defined params are present? Or be lenient?
        # Let's be slightly lenient: validate if constraints pass for PRESENT params.
        if not space.validate(hyperparams):
            # print(f"{experiment_id}: Hyperparameters failed validation. Params: {hyperparams}")
            continue

        # --- Extract primary metric ---
        metric_value = metrics.get(metric_key)
        if metric_value is None or not isinstance(metric_value, (int, float)) or not math.isfinite(metric_value):
            # print(f"{experiment_id}: Invalid or missing primary metric value '{metric_value}'.")
            continue

        result = ExperimentResult(
            experiment_id=experiment_id,
            hyperparams=hyperparams,
            data_params=data_params,
            metric_value=float(metric_value),
            # <<< CHANGE: Extract other relevant metrics >>>
            best_epoch=metrics.get('best_epoch'),
            epochs_completed=metrics.get('epochs_completed'),
            training_duration=metrics.get('training_duration')
        )
        parsed_results.append(result)

    print(f"Successfully parsed {len(parsed_results)} valid experiment results for analysis.")
    return parsed_results

def get_experiment_config_tuple(exp_hyperparams: dict, space: HyperparameterSpace) -> tuple:
    """Creates a canonical tuple representation of hyperparameters for duplicate checking."""
    # Use sorted param names from space for consistency
    return tuple(exp_hyperparams.get(p) for p in space.get_param_names())

def load_and_prepare_data(space: HyperparameterSpace, metric_key: str) -> Tuple[List[ExperimentResult], Set[tuple]]:
    """Loads completed and pending data, parses, returns results and existing configs set."""
    print(f"Loading completed experiments from: {COMPLETED_FILE}")
    # <<< CHANGE: Use utils.safe_json_load >>>
    raw_completed = utils.safe_json_load(COMPLETED_FILE)
    print(f"Loading pending experiments from: {PENDING_FILE}")
    raw_pending = utils.safe_json_load(PENDING_FILE)

    # <<< CHANGE: Pass metric_key to parse_experiment_data >>>
    completed_results = parse_experiment_data(raw_completed, space, metric_key)

    existing_configs = set()
    # Add completed configs
    for res in completed_results:
        # Ensure hyperparams are valid before adding tuple (already done in parse, but belts & braces)
        if space.validate(res.hyperparams):
             existing_configs.add(get_experiment_config_tuple(res.hyperparams, space))

    # Add pending configs
    pending_count = 0
    for p_exp in raw_pending:
         if isinstance(p_exp, dict) and 'hyperparams' in p_exp:
            # Validate pending experiment hyperparams before adding to set
            if space.validate(p_exp['hyperparams']):
                existing_configs.add(get_experiment_config_tuple(p_exp['hyperparams'], space))
                pending_count += 1
            # else: # Optional: Log invalid pending configs
            #     print(f"Pending config failed validation: {p_exp.get('hyperparams')}")

    print(f"Found {len(existing_configs)} unique valid configurations ({len(completed_results)} completed + {pending_count} pending).")
    return completed_results, existing_configs

# --- Analysis Engine ---

def analyze_parameter_trends(results: List[ExperimentResult], space: HyperparameterSpace, metric_key: str) -> Dict[str, pd.DataFrame]:
    """
    Analyzes the impact of each hyperparameter on the primary metric using Pandas.
    """
    if not results: return {}
    print(f"\n--- Parameter Trend Analysis (Based on '{metric_key}') ---")

    data_for_df = []
    for res in results:
        # Include only parameters defined in the space for analysis consistency
        record = {p: res.hyperparams.get(p) for p in space.get_param_names()}
        record['metric'] = res.metric_value # Add the primary metric
        record['experiment_id'] = res.experiment_id
        data_for_df.append(record)

    if not data_for_df: return {}
    df = pd.DataFrame(data_for_df)

    trend_analysis = {}
    for param_name in space.get_param_names():
        if param_name not in df.columns or df[param_name].isnull().all() or df[param_name].nunique() < 2:
            continue

        # Handle potential categorical floats that need grouping by closeness
        is_float_categorical = False
        if 'values' in space.parameters[param_name] and space.parameters[param_name]['type'] == 'float':
             is_float_categorical = True

        try:
            if is_float_categorical:
                 # Group floats by mapping them to the closest defined value in the space
                 defined_vals = sorted(space.parameters[param_name]['values'])
                 def map_to_closest(val):
                     if pd.isna(val): return np.nan
                     closest_val = min(defined_vals, key=lambda x: abs(x - val))
                     # Only map if reasonably close to avoid mis-grouping outliers
                     if math.isclose(val, closest_val, rel_tol=1e-6):
                         return closest_val
                     return np.nan # Treat as NaN if not close to a defined value
                 grouping_col = df[param_name].apply(map_to_closest)
                 stats = df.groupby(grouping_col)['metric'].agg(['mean', 'median', 'std', 'count']).sort_index()
            else:
                 # Standard grouping for ints or non-categorical floats/ints
                 stats = df.groupby(param_name)['metric'].agg(['mean', 'median', 'std', 'count']).sort_index()

            # Filter out groups with only 1 data point for reliability? Maybe keep if few groups overall.
            min_group_size = 2 # Require at least 2 samples per group value for stats
            reliable_stats = stats[stats['count'] >= min_group_size]
            stats_to_use = reliable_stats if not reliable_stats.empty else stats # Fallback if filtering leaves nothing

            if not stats_to_use.empty:
                trend_analysis[param_name] = stats_to_use
                print(f"\nParameter: {param_name} (Stats based on groups >= {min_group_size} samples if possible)")
                print(stats_to_use.to_string(float_format="%.4f"))
        except Exception as e:
             print(f"Could not analyze {param_name}: {e}")

    return trend_analysis


def find_best_experiment(results: List[ExperimentResult], metric_mode: str) -> Optional[ExperimentResult]:
    """Finds the single best experiment based on the primary metric and mode."""
    if not results: return None
    valid_results = [r for r in results if r.metric_value is not None and math.isfinite(r.metric_value)]
    if not valid_results: return None

    try:
        if metric_mode == 'max':
            best_exp = max(valid_results, key=lambda exp: exp.metric_value)
        else: # min mode
            best_exp = min(valid_results, key=lambda exp: exp.metric_value)
        return best_exp
    except ValueError:
         return None


# --- Recommendation Engine ---
# (suggest_cold_start_batches, check_config, suggest_trend_following_batches, suggest_refinement_batches
#  remain largely the same conceptually, ensure they use the updated HyperparameterSpace methods correctly)

def suggest_cold_start_batches(space: HyperparameterSpace, existing_configs: Set[tuple], num_suggestions: int) -> List[Dict]:
    """ Suggests diverse batches for initial exploration, checking validity and uniqueness. """
    print("\n--- Generating Cold Start Suggestions ---")
    suggestions = []
    core_params = space.get_param_names()
    temp_existing = existing_configs.copy()

    # Try combinations of boundary and central values
    potential_configs = []
    param_options = {}

    for pname, spec in space.parameters.items():
        options = set()
        if 'values' in spec:
            sorted_vals = sorted(spec['values'])
            options.add(sorted_vals[0]) # Min
            options.add(sorted_vals[-1]) # Max
            if len(sorted_vals) > 2: options.add(sorted_vals[len(sorted_vals)//2]) # Mid
        elif 'range' in spec:
            low, high = spec['range']
            mid = low + (high - low) * 0.5
            options.add(low)
            options.add(high)
            # Snap mid value based on type and step
            if spec['type'] == 'int':
                 step = spec.get('step', 1)
                 snapped_mid = int(round(mid / step) * step)
                 snapped_mid = max(low, min(high, snapped_mid)) # Clamp to range
                 options.add(snapped_mid)
            else: # float
                 options.add(mid) # No snapping for float mid? Or implement if needed.
        if options:
             param_options[pname] = sorted(list(options))

    # Generate product and check validity/uniqueness
    param_names = list(param_options.keys())
    value_lists = [param_options[p] for p in param_names]

    for combo_values in itertools.product(*value_lists):
        temp_config = dict(zip(param_names, combo_values))
        if check_config(temp_config, space, temp_existing):
            potential_configs.append(temp_config)
            # Add to temp_existing to avoid suggesting duplicates within cold start
            temp_existing.add(get_experiment_config_tuple(temp_config, space))

    # Select a diverse subset (e.g., randomly sample or prioritize extremes)
    # For simplicity, just take the first N valid unique ones found
    final_configs = potential_configs[:num_suggestions]

    # Format as batches (each config becomes a single-item batch for cold start)
    for config in final_configs:
         batch = {k: [v] for k, v in config.items()}
         suggestions.append(batch)

    print(f"Suggesting {len(suggestions)} diverse configurations for cold start.")
    return suggestions


def check_config(config: Dict[str, Any], space: HyperparameterSpace, existing_configs: Set[tuple]) -> bool:
    """Helper to validate a config and check for duplicates."""
    config_tuple = get_experiment_config_tuple(config, space)
    # <<< CHANGE: Ensure config validation happens first >>>
    return space.validate(config) and config_tuple not in existing_configs


def suggest_trend_following_batches(
    analysis: Dict[str, pd.DataFrame],
    base_config: Dict[str, Any],
    space: HyperparameterSpace,
    existing_configs: Set[tuple],
    metric_mode: str,
    num_suggestions: int
    ) -> List[Dict]:
    """ Suggests experiments by moving parameters towards values with better average performance. """
    print("\n--- Generating Trend Following Suggestions ---")
    suggestions = []
    core_params = space.get_param_names()
    added_configs_in_this_run = set() # Track suggestions within this function call

    # Prioritize params with clearer trends? (e.g., larger diff in mean metric, lower std)
    # Simple approach: iterate through analyzed params
    params_to_explore = list(analysis.keys())
    np.random.shuffle(params_to_explore) # Randomize order to avoid bias

    for param_name in params_to_explore:
        if len(suggestions) >= num_suggestions: break # Stop if we have enough suggestions
        if param_name not in base_config or param_name not in space.parameters: continue

        stats_df = analysis[param_name]
        current_value = base_config[param_name]

        # Ensure index type matches current value type for comparison
        try:
            stats_df.index = stats_df.index.astype(type(current_value))
        except Exception: continue

        if current_value not in stats_df.index: continue

        # Sort stats by mean performance
        sorted_stats = stats_df.sort_values('mean', ascending=(metric_mode == 'min'))
        best_val_in_stats = sorted_stats.index[0]

        direction = None
        # <<< CHANGE: Use math.isclose for float comparison >>>
        if not math.isclose(best_val_in_stats, current_value, rel_tol=1e-9):
            if (metric_mode == 'max' and best_val_in_stats > current_value) or \
               (metric_mode == 'min' and best_val_in_stats < current_value):
                 direction = 'increase' if best_val_in_stats > current_value else 'decrease'
            elif (metric_mode == 'max' and best_val_in_stats < current_value) or \
                 (metric_mode == 'min' and best_val_in_stats > current_value):
                  direction = 'decrease' if best_val_in_stats < current_value else 'increase'


        suggested_values = []
        if direction:
            # Suggest values in the identified direction (n=1 or 2)
            suggested_values = space.get_trend_values(param_name, current_value, direction, n=1) # Just one step usually better
        else: # Current value is the best among tested points
             # Try nearby values if not at boundary
             # (Boundary check logic can be complex, maybe just always try nearby?)
             suggested_values = space.get_nearby_values(param_name, current_value, n=1) # Try one nearby

        # --- Create and Validate Configs ---
        if suggested_values:
            valid_suggestions_for_param = []
            for sugg_val in suggested_values:
                 temp_config = base_config.copy()
                 temp_config[param_name] = sugg_val
                 config_tuple = get_experiment_config_tuple(temp_config, space)
                 if config_tuple not in added_configs_in_this_run and check_config(temp_config, space, existing_configs):
                     valid_suggestions_for_param.append(sugg_val)
                     added_configs_in_this_run.add(config_tuple)

            if valid_suggestions_for_param:
                 # Create a batch suggestion varying only this parameter
                 batch = base_config.copy()
                 # Format the single parameter variation as a list
                 batch[param_name] = sorted(valid_suggestions_for_param)
                 # Ensure all other params are single-item lists for manage_queue.py
                 formatted_batch = {k: ([v] if k != param_name else v) for k, v in batch.items()}
                 suggestions.append(formatted_batch)


    print(f"Generated {len(suggestions)} trend-following suggestions.")
    return suggestions


def suggest_refinement_batches(
    best_config: Dict[str, Any],
    space: HyperparameterSpace,
    existing_configs: Set[tuple],
    num_suggestions: int
    ) -> List[Dict]:
    """ Suggests small variations around the best known configuration. """
    print("\n--- Generating Refinement Suggestions (Near Best Run) ---")
    suggestions = []
    core_params = space.get_param_names()
    added_configs_in_this_run = set()

    params_to_refine = [p for p in core_params if p in best_config and p in space.parameters]
    np.random.shuffle(params_to_refine) # Randomize parameter order

    for param_name in params_to_refine:
        if len(suggestions) >= num_suggestions: break

        current_value = best_config[param_name]
        # Get 1 or 2 nearby values
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
                 formatted_batch = {k: ([v] if k != param_name else v) for k, v in batch.items()}
                 suggestions.append(formatted_batch)

    print(f"Generated {len(suggestions)} refinement suggestions.")
    return suggestions


# --- Output Formatting ---
def format_batch_for_queue(batch_dict: dict, strategy_name: str) -> str:
    """ Formats a batch dictionary into a manage_queue.py command string. """
    # Include necessary fixed data parameters needed by manage_queue.py
    # Find these values from the first suggestion dict or pass them explicitly
    # Assuming they are constant for now based on last run:
    command_args = {
        "num_samples": 10000, # Get this dynamically if it can vary
        "sample_length": 250, # Get this dynamically if it can vary
    }
    # Update with suggested hyperparams
    command_args.update(batch_dict)

    command = f"python manage_queue.py"
    # Sort items for consistent command string order
    sorted_items = sorted(command_args.items())
    has_params = False

    for param, values in sorted_items:
        if values is None: continue # Skip params with None value
        # Ensure values is a list, even if only one item from suggestions
        values = values if isinstance(values, list) else [values]
        if not values: continue # Skip if list is empty

        has_params = True
        values_str = []
        for v in values:
             if isinstance(v, float):
                 # Use scientific notation for small/large floats, default otherwise
                 if (abs(v) > 0 and abs(v) < 1e-3) or abs(v) >= 1e4:
                      values_str.append(f"{v:.4e}")
                 else:
                      # Format with good precision, avoid trailing .0
                      formatted_v = f"{v:.8g}"
                      values_str.append(formatted_v)
             else:
                 values_str.append(str(v))

        # Handle potential name mismatch ('patience' vs 'early_stopping_patience')
        # Use the name expected by manage_queue.py
        arg_name = 'patience' if param == 'early_stopping_patience' else param
        command += f" --{arg_name} {','.join(values_str)}"

    if not has_params: return "" # Return empty if no params generated command
    command += f"  # Suggestion: {strategy_name}"
    return command

def format_recommendations_for_queue(recommendations: Dict[str, List[Dict]]) -> Tuple[str, int]:
    """ Formats all suggested batches into command-line strings and counts unique commands. """
    output_lines = []
    unique_commands = set()

    for strategy, batches in recommendations.items():
        strategy_output = []
        for i, batch in enumerate(batches):
             command = format_batch_for_queue(batch, f"{strategy}_{i+1}") # Add index to strategy name
             if command and command not in unique_commands:
                 strategy_output.append(command)
                 unique_commands.add(command)

        if strategy_output:
            output_lines.append(f"\n# --- Strategy: {strategy} ---")
            output_lines.extend(strategy_output)

    num_unique = len(unique_commands)
    output_lines.insert(0, f"# --- Recommended Batches for manage_queue.py ({num_unique} unique commands) ---")
    return "\n".join(output_lines), num_unique


# --- Main Orchestration ---

def main():
    parser = argparse.ArgumentParser(
        description="Analyze completed experiments and suggest new hyperparameter configurations."
    )
    parser.add_argument('--metric', type=str, default=PRIMARY_METRIC, help=f"Metric key to optimize (default: {PRIMARY_METRIC})")
    parser.add_argument('--mode', choices=['min', 'max'], default=METRIC_MODE, help=f"Optimization mode (default: {METRIC_MODE})")
    parser.add_argument('--min_exp', type=int, default=MIN_EXPERIMENTS_FOR_ANALYSIS, help=f"Min experiments for analysis (default: {MIN_EXPERIMENTS_FOR_ANALYSIS})")
    parser.add_argument('--num_sugg', type=int, default=NUM_SUGGESTIONS_PER_STRATEGY, help=f"Target suggestions per strategy (default: {NUM_SUGGESTIONS_PER_STRATEGY})")
    parser.add_argument('--output', type=str, default=None, help="Optional file path to save suggestion commands.")

    args = parser.parse_args()

    print("--- Starting Experiment Suggestion ---")
    print(f"Using Project Root: {utils._PROJECT_ROOT}")
    print(f"Optimizing Metric: '{args.metric}' (Mode: {args.mode})")
    print(f"Min experiments for analysis: {args.min_exp}")
    print(f"Target suggestions per strategy: {args.num_sugg}")

    # Initialize space and load data
    space = HyperparameterSpace(PARAM_SPACE_DEFINITION, [check_constraints])
    completed_results, existing_configs = load_and_prepare_data(space, args.metric)

    recommendations = defaultdict(list)
    all_suggestions = [] # Store all suggestion dicts to update existing_configs accurately

    # Decide strategy
    if len(completed_results) < args.min_exp:
        print(f"\nInsufficient data ({len(completed_results)} valid runs). Generating cold start suggestions.")
        cold_start_batches = suggest_cold_start_batches(space, existing_configs, args.num_sugg)
        recommendations['cold_start'] = cold_start_batches
        all_suggestions.extend(cold_start_batches)
    else:
        print(f"\nSufficient data ({len(completed_results)} valid runs) for analysis.")
        best_experiment = find_best_experiment(completed_results, args.mode)

        if not best_experiment:
            print("\nError: Could not identify a best-performing experiment. Falling back to cold start.")
            cold_start_batches = suggest_cold_start_batches(space, existing_configs, args.num_sugg)
            recommendations['cold_start'] = cold_start_batches
            all_suggestions.extend(cold_start_batches)
        else:
            print(f"\nBest experiment found: {best_experiment.experiment_id} ({args.metric}: {best_experiment.metric_value:.4f})")
            # Limit displayed params for brevity
            best_params_subset = {k: v for k, v in best_experiment.hyperparams.items() if k in space.get_param_names()}
            print(f"  Best Params: {best_params_subset}")

            analysis_results = analyze_parameter_trends(completed_results, space, args.metric)

            # Update existing_configs before generating new suggestions
            current_existing_count = len(existing_configs)

            # Generate suggestions - pass args.num_sugg and args.mode
            if analysis_results:
                trend_batches = suggest_trend_following_batches(
                    analysis_results, best_experiment.hyperparams, space, existing_configs, args.mode, args.num_sugg)
                recommendations['trend_following'] = trend_batches
                all_suggestions.extend(trend_batches)
                # Update existing_configs immediately after generation
                for batch in trend_batches:
                     param_names_batch = list(batch.keys())
                     value_lists_batch = [batch[p] for p in param_names_batch]
                     for combo_values in itertools.product(*value_lists_batch):
                         temp_config = dict(zip(param_names_batch, combo_values))
                         existing_configs.add(get_experiment_config_tuple(temp_config, space))
                print(f"Updated existing configs count: {current_existing_count} -> {len(existing_configs)}")
                current_existing_count = len(existing_configs) # Update count

            refinement_batches = suggest_refinement_batches(
                best_experiment.hyperparams, space, existing_configs, args.num_sugg)
            recommendations['refinement'] = refinement_batches
            all_suggestions.extend(refinement_batches)
            # Update existing_configs after refinement
            for batch in refinement_batches:
                 param_names_batch = list(batch.keys())
                 value_lists_batch = [batch[p] for p in param_names_batch]
                 for combo_values in itertools.product(*value_lists_batch):
                     temp_config = dict(zip(param_names_batch, combo_values))
                     existing_configs.add(get_experiment_config_tuple(temp_config, space))
            print(f"Updated existing configs count: {current_existing_count} -> {len(existing_configs)}")


    # --- Format and Output Results ---
    queue_commands, num_commands = format_recommendations_for_queue(recommendations)
    print("\n" + queue_commands)

    # Save commands to a file
    if num_commands > 0:
        if args.output:
             output_filename = args.output
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # <<< CHANGE: Use utils._PROJECT_ROOT for output dir >>>
            output_dir = os.path.join(utils._PROJECT_ROOT, "data", "param_analysis")
            os.makedirs(output_dir, exist_ok=True)
            output_filename = os.path.join(output_dir, f"suggestions_{timestamp}.txt")

        try:
            with open(output_filename, "w") as f:
                f.write(f"# Experiment suggestions generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Optimizing: {args.metric} ({args.mode})\n")
                f.write(f"# Based on {len(completed_results)} completed runs.\n")
                f.write(queue_commands)
            print(f"\nSuggestions also saved to: {output_filename}")
        except Exception as e:
            print(f"\nError saving suggestions to file {output_filename}: {e}")
    else:
        print("\nNo new unique experiment suggestions were generated.")

    print("\n--- Suggestion process complete ---")


if __name__ == "__main__":
    main()
