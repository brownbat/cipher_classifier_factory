import dash
from dash import dcc, html, Input, Output, State, no_update
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np
import argparse
from collections import defaultdict

# Define constants for file paths (assuming utils.py is not easily importable here)
# Ideally, these would be imported from a shared constants module or utils.py
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COMPLETED_EXPERIMENTS_FILE = os.path.join(PROJECT_ROOT, "data", "completed_experiments.json")
ASSETS_FOLDER = os.path.join(PROJECT_ROOT, "assets")
CSS_FILE = os.path.join(ASSETS_FOLDER, "custom_styles.css")

# Define the order and names for key metrics used in visualization
# Index 0: X-axis, Index 1: Y-axis
METRIC_KEYS_ORDER = ['training_duration', 'best_val_accuracy', 'best_val_loss']
METRIC_FALLBACK_KEYS_ORDER = ['training_duration', 'val_accuracy_curve', 'val_loss_curve'] # Used if 'best' metrics aren't available
METRIC_LABELS = ['Training Duration (s)', 'Validation Accuracy', 'Validation Loss']

# --- Utility Functions ---

def safe_json_load(file_path):
    """Loads JSON data from a file, handling errors gracefully."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Ensure it returns a list, even if the file contains a single object (though it shouldn't)
            return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"WARNING: '{os.path.basename(file_path)}' is empty or invalid. Returning empty list.")
        return []
    except FileNotFoundError:
        print(f"WARNING: '{os.path.basename(file_path)}' not found. Returning empty list.")
        # Optionally create an empty file:
        # os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # with open(file_path, 'w') as file:
        #     json.dump([], file)
        return []

def generate_colors(num_colors, saturation=40, lightness=40):
    """Generate 'num_colors' distinct HSL colors."""
    colors = []
    for i in range(num_colors):
        hue = int((360 / num_colors) * i)
        colors.append(f"hsl({hue}, {saturation}%, {lightness}%)")
    return colors

def generate_slider_css(num_sliders, colors):
    """Generate CSS rules for styling sliders with different colors."""
    css_rules = ""
    for i in range(num_sliders):
        color = colors[i % len(colors)]
        css_rules += f"""
        .slider-color-{i+1} .rc-slider-track {{ background-color: {color}; }}
        .slider-color-{i+1} .rc-slider-handle {{ border-color: {color}; }}
        """
    return css_rules

def write_css_to_file(css_content, filename=CSS_FILE):
    """Writes CSS content to the specified file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(css_content)

# --- Data Loading and Filtering ---

def load_data(max_experiments=None):
    """Loads experiment data from the main JSON file."""
    data = safe_json_load(COMPLETED_EXPERIMENTS_FILE)
    print(f"Loaded {len(data)} total experiments from '{os.path.basename(COMPLETED_EXPERIMENTS_FILE)}'")

    # Limit to max_experiments if specified
    if max_experiments is not None and max_experiments > 0:
        data = data[:max_experiments]
        print(f"Limited to {len(data)} experiments based on --max_experiments")

    return data

def parse_filter_string(filter_str):
    """Parses a filter string (e.g., "epochs=40;d_model=128,256") into a dict."""
    filter_params = {'hyperparams': {}, 'data_params': {}}
    param_groups = filter_str.split(';')
    for group in param_groups:
        if '=' in group:
            param, values_str = group.split('=', 1)
            param = param.strip()
            values = []
            for v_str in values_str.split(','):
                v_str = v_str.strip()
                if not v_str: continue
                try:
                    values.append(float(v_str) if '.' in v_str or 'e' in v_str else int(v_str))
                except ValueError:
                    values.append(v_str) # Keep as string if conversion fails

            # Determine if it's likely a hyperparam or data_param (heuristic)
            if param in ['num_samples', 'sample_length', 'ciphers']: # Add known data params
                filter_params['data_params'][param] = values
            else: # Assume hyperparameter otherwise
                filter_params['hyperparams'][param] = values
    return filter_params

def load_filter_config(filter_arg):
    """Loads filter configuration from a string or JSON file path."""
    if os.path.exists(filter_arg) and filter_arg.endswith('.json'):
        try:
            with open(filter_arg, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load or parse filter file '{filter_arg}': {e}")
            return {}
    else:
        try:
            return parse_filter_string(filter_arg)
        except Exception as e:
            print(f"ERROR: Could not parse filter string '{filter_arg}': {e}")
            return {}

def apply_filters(experiments, filter_config):
    """Filters experiments based on the provided filter configuration."""
    if not filter_config or not experiments:
        return experiments

    filtered_experiments = []
    for exp in experiments:
        include = True
        # Check data_params
        for param, allowed_values in filter_config.get('data_params', {}).items():
            if param not in exp.get('data_params', {}) or exp['data_params'][param] not in allowed_values:
                include = False
                break
        if not include: continue

        # Check hyperparams
        for param, allowed_values in filter_config.get('hyperparams', {}).items():
            if param not in exp.get('hyperparams', {}) or exp['hyperparams'][param] not in allowed_values:
                include = False
                break
        if not include: continue

        # Check for essential metrics (ensure experiment likely completed)
        metrics = exp.get('metrics', {})
        if not metrics or METRIC_KEYS_ORDER[0] not in metrics or (METRIC_KEYS_ORDER[1] not in metrics and METRIC_FALLBACK_KEYS_ORDER[1] not in metrics):
             include = False

        if include:
            filtered_experiments.append(exp)

    print(f"Filtered down to {len(filtered_experiments)} experiments")
    return filtered_experiments

# --- Parameter and Data Transformation ---

def discover_parameters_and_order(experiments):
    """
    Discovers all unique parameter keys from data_params and hyperparams
    across all experiments and returns them in a sorted list (canonical order).
    Excludes identifier 'experiment_id'.
    """
    param_keys = set()
    excluded_keys = {'experiment_id'} # Keys not treated as tunable parameters

    for exp in experiments:
        param_keys.update(k for k in exp.get('data_params', {}).keys() if k not in excluded_keys)
        param_keys.update(k for k in exp.get('hyperparams', {}).keys() if k not in excluded_keys)

    return sorted(list(param_keys))

def transform_data(experiments, canonical_param_order):
    """
    Transforms raw experiment data into tuples for visualization, using the
    canonical parameter order.

    Returns:
        List of (parameters_tuple, metrics_tuple, experiment_info_dict)
    """
    transformed_tuples = []
    missing_metrics_count = 0

    for exp in experiments:
        params_list = []
        for param_name in canonical_param_order:
            value = exp.get('data_params', {}).get(param_name, exp.get('hyperparams', {}).get(param_name))
            if isinstance(value, list):
                 value = tuple(value) # Convert lists to hashable tuples
            params_list.append(value)
        parameters_tuple = tuple(params_list)

        # Extract metrics, prioritizing 'best' values
        metrics = exp.get('metrics', {})
        duration = metrics.get(METRIC_KEYS_ORDER[0])
        accuracy = metrics.get(METRIC_KEYS_ORDER[1])
        loss = metrics.get(METRIC_KEYS_ORDER[2])

        # Fallback for accuracy/loss using curves
        if accuracy is None and METRIC_FALLBACK_KEYS_ORDER[1] in metrics:
            curve = metrics[METRIC_FALLBACK_KEYS_ORDER[1]]
            if isinstance(curve, list) and curve: accuracy = curve[-1]
        if loss is None and METRIC_FALLBACK_KEYS_ORDER[2] in metrics:
            curve = metrics[METRIC_FALLBACK_KEYS_ORDER[2]]
            if isinstance(curve, list) and curve: loss = curve[-1]

        # Only include if primary metrics (duration and accuracy) are available
        if duration is not None and accuracy is not None:
            metrics_tuple = (duration, accuracy, loss if loss is not None else float('nan'))
            experiment_info = {'experiment_id': exp.get('experiment_id', 'unknown')}
            transformed_tuples.append((parameters_tuple, metrics_tuple, experiment_info))
        else:
            missing_metrics_count += 1

    if missing_metrics_count > 0:
         print(f"WARNING: Skipped {missing_metrics_count} experiments due to missing essential metrics (duration or accuracy).")

    return transformed_tuples


def build_param_value_map(transformed_data, canonical_param_order):
    """
    Builds a map of parameter names to the set of unique values observed
    for that parameter across all transformed experiments. Values are sorted.
    """
    param_value_map = {name: set() for name in canonical_param_order}
    for params_tuple, _, _ in transformed_data:
        for idx, value in enumerate(params_tuple):
            param_name = canonical_param_order[idx]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                param_value_map[param_name].add(value)

    # Convert sets to sorted lists
    for name, values in param_value_map.items():
        try:
            param_value_map[name] = sorted(list(values))
        except TypeError:
             param_value_map[name] = sorted(list(values), key=str) # Fallback sort for mixed types

    return param_value_map

# --- Slider Mark Generation ---

def calculate_avg_accuracy_per_value(experiments, param_name):
    """Calculates average validation accuracy for each value of a given parameter."""
    value_accuracy_dict = defaultdict(list)
    for exp in experiments:
        value = exp.get('data_params', {}).get(param_name, exp.get('hyperparams', {}).get(param_name))
        if value is not None:
            metrics = exp.get('metrics', {})
            accuracy = metrics.get(METRIC_KEYS_ORDER[1]) # best_val_accuracy
            if accuracy is None and METRIC_FALLBACK_KEYS_ORDER[1] in metrics:
                curve = metrics.get(METRIC_FALLBACK_KEYS_ORDER[1], [])
                if curve: accuracy = curve[-1]

            if accuracy is not None and accuracy > 0:
                 key = tuple(value) if isinstance(value, list) else value
                 value_accuracy_dict[key].append(accuracy)

    avg_accuracy_per_value = { v: sum(accs)/len(accs) for v, accs in value_accuracy_dict.items() if accs }
    return avg_accuracy_per_value


def generate_slider_marks(experiments, param_name, param_values_sorted):
    """Generates marks for a slider, showing average accuracy."""
    avg_accuracy_map = calculate_avg_accuracy_per_value(experiments, param_name)
    marks = {}
    for value in param_values_sorted:
        lookup_key = tuple(value) if isinstance(value, list) else value
        avg_acc = avg_accuracy_map.get(lookup_key)
        # Format label value nicely
        if isinstance(value, tuple): label_value = f"({len(value)} items)"
        elif isinstance(value, float): label_value = f"{value:.1e}" if abs(value) < 1e-3 or abs(value) > 1e4 else f"{value:.4g}"
        else: label_value = str(value)

        marks[value] = f"{label_value} ({avg_acc:.3f})" if avg_acc is not None else label_value
    return marks

# --- Dash App Setup ---

def setup_dash_app(experiments):
    """Sets up the Dash application layout and callbacks."""
    if not experiments:
        app = dash.Dash(__name__, assets_folder=ASSETS_FOLDER)
        app.layout = html.Div([html.H3("No experiment data to visualize.")])
        return app

    # 1. Discover Parameters & Transform Data
    canonical_param_order = discover_parameters_and_order(experiments)
    transformed_data = transform_data(experiments, canonical_param_order)
    if not transformed_data:
        app = dash.Dash(__name__, assets_folder=ASSETS_FOLDER)
        app.layout = html.Div([html.H3("No valid data points to visualize (check metrics).")])
        return app

    # Lookup dict: {parameters_tuple: (metrics_tuple, experiment_info_dict)}
    transformed_data_dict = {params: (metrics, info) for params, metrics, info in transformed_data}

    # 2. Analyze Parameters
    param_value_map = build_param_value_map(transformed_data, canonical_param_order)
    variable_params = {k: v for k, v in param_value_map.items() if len(v) > 1}
    constant_params = {k: v[0] for k, v in param_value_map.items() if len(v) == 1}
    num_variable_params = len(variable_params)
    print(f"Discovered {len(canonical_param_order)} parameters. Variable: {num_variable_params}, Constant: {len(constant_params)}.")

    # 3. Prepare Data for Plotting & Scaling
    metrics_list = [metrics for _, metrics, _ in transformed_data]
    df = pd.DataFrame(metrics_list, columns=METRIC_LABELS[:len(metrics_list[0])])
    x_axis_label, y_axis_label = METRIC_LABELS[0], METRIC_LABELS[1]

    if x_axis_label not in df.columns or y_axis_label not in df.columns:
         app = dash.Dash(__name__, assets_folder=ASSETS_FOLDER)
         app.layout = html.Div([html.H3(f"Error: Metric keys '{x_axis_label}' or '{y_axis_label}' not found.")])
         return app

    min_x, max_x = df[x_axis_label].min(), df[x_axis_label].max()
    min_y, max_y = df[y_axis_label].min(), df[y_axis_label].max()
    buffer_x = (max_x - min_x) * 0.1 if max_x > min_x else 1.0
    buffer_y = (max_y - min_y) * 0.1 if max_y > min_y else 0.1

    graph_layout = {
        'xaxis': {'range': [min_x - buffer_x, max_x + buffer_x], 'title': x_axis_label},
        'yaxis': {'range': [min_y - buffer_y, max_y + buffer_y], 'title': y_axis_label},
        'margin': dict(l=40, r=40, t=40, b=40),
        'hovermode': 'closest',
        'legend': {'tracegroupgap': 5} # Add some space between legend groups
    }

    # 4. Find Best Experiment (based on Y-axis metric: Accuracy)
    best_params_tuple, best_metric_value, best_exp_info = None, -float('inf'), {}
    for params_tuple, metrics_tuple, exp_info in transformed_data:
        if len(metrics_tuple) > 1 and metrics_tuple[1] > best_metric_value:
            best_params_tuple, best_metric_value = params_tuple, metrics_tuple[1]
            best_exp_info = exp_info

    # Prepare info strings
    constant_params_str = ", ".join([f"{k}={v}" for k, v in constant_params.items()])
    constant_params_info = f"Fixed Parameters: {constant_params_str or 'None'}"

    best_variable_params_dict = {}
    if best_params_tuple:
        for idx, param_name in enumerate(canonical_param_order):
            if param_name in variable_params:
                best_variable_params_dict[param_name] = best_params_tuple[idx]

    highest_accuracy_info = "Highest Accuracy: Not Found"
    if best_exp_info:
         best_id = best_exp_info.get('experiment_id', 'unknown')
         best_vars_str = ", ".join([f"{k}={v}" for k,v in best_variable_params_dict.items()])
         highest_accuracy_info = (f"Highest Accuracy ({best_id}): {best_metric_value:.4f} "
                                 f"with: {{{best_vars_str or 'N/A'}}}")


    # 5. Setup Dash App Layout
    app = dash.Dash(__name__, assets_folder=ASSETS_FOLDER)
    colors = generate_colors(num_variable_params)
    slider_css = generate_slider_css(num_variable_params, colors)
    write_css_to_file(slider_css)

    layout_children = [
        html.H3(f"Cipher Classification - Experiment Visualizer ({len(experiments)} Experiments)",
                style={'text-align': 'center', 'margin-bottom': '10px'}),
        html.Div(id='subtitle', style={'text-align': 'center', 'margin-bottom': '20px'}),
        dcc.Graph(id='output-graph', figure={'layout': graph_layout}, config={'displayModeBar': True}),
        html.Div([
            html.P(constant_params_info),
            html.P(highest_accuracy_info),
            html.P(id='current-params-info') # << NEW: Placeholder for current params
        ], style={'padding': '10px', 'border': '1px solid #ddd', 'marginTop': '10px'}),
    ]

    # Create Sliders for Variable Parameters
    slider_components = []
    variable_params_list = sorted(variable_params.keys())

    for idx, param_name in enumerate(variable_params_list):
        values = variable_params[param_name]
        min_val, max_val = values[0], values[-1]
        initial_value = best_variable_params_dict.get(param_name, values[0])
        marks = generate_slider_marks(experiments, param_name, values)

        use_index_slider = not all(isinstance(v, (int, float)) for v in values)
        if not use_index_slider and len(values) > 1 and isinstance(min_val, (int, float)):
            steps = np.diff(values)
            if not np.allclose(steps, steps[0]): use_index_slider = True

        if use_index_slider:
             slider_marks = {i: marks.get(values[i], str(values[i])) for i in range(len(values))}
             slider_min, slider_max = 0, len(values) - 1
             try: slider_value = values.index(initial_value)
             except ValueError: slider_value = 0
             slider_step = 1
        else:
             slider_marks = {val: marks.get(val, str(val)) for val in values}
             slider_min, slider_max = float(min_val), float(max_val)
             try: slider_value = float(initial_value)
             except (ValueError, TypeError): slider_value = float(min_val)
             slider_step = None

        slider_components.append(html.Div([
            html.Label(f'{param_name}'),
            dcc.Slider(
                id={'type': 'param-slider', 'index': param_name},
                min=slider_min, max=slider_max, value=slider_value,
                marks=slider_marks, step=slider_step, included=False,
                className=f'slider-color-{idx+1}'
            ),
            dcc.Store(id={'type': 'slider-mode', 'index': param_name}, data={'use_index': use_index_slider, 'values': values if use_index_slider else None})
        ], style={'padding': '10px 0px'}))

    layout_children.extend(slider_components)
    layout_children.append(dcc.Store(id='param-info-store', data={
         'canonical_order': canonical_param_order,
         'variable_params': variable_params_list,
         'constant_params': constant_params
    }))
    app.layout = html.Div(layout_children)

    # 6. Define Callbacks

    @app.callback(
        Output('output-graph', 'figure'),
        Output('subtitle', 'children'),
        Output('current-params-info', 'children'), # << NEW: Output for current params text
        Input({'type': 'param-slider', 'index': dash.dependencies.ALL}, 'value'),
        State('param-info-store', 'data'),
        State({'type': 'slider-mode', 'index': dash.dependencies.ALL}, 'data')
    )
    def update_graph(slider_values, param_info, slider_modes):
        canonical_order = param_info['canonical_order']
        variable_params_keys = param_info['variable_params']
        constant_params_dict = param_info['constant_params']

        slider_value_map = {key: val for key, val in zip(variable_params_keys, slider_values)}
        slider_mode_map = {key: mode_data for key, mode_data in zip(variable_params_keys, slider_modes)}

        # Reconstruct the selected parameter values dictionary
        current_params_dict = {}
        for param_name in variable_params_keys:
             slider_value = slider_value_map[param_name]
             mode_info = slider_mode_map[param_name]
             if mode_info['use_index']:
                 try: current_params_dict[param_name] = mode_info['values'][slider_value]
                 except IndexError: current_params_dict[param_name] = None
             else:
                 current_params_dict[param_name] = slider_value
        current_params_dict.update(constant_params_dict) # Add constants

        # Build the parameters_tuple key in the canonical order
        try:
            current_key_list = [current_params_dict.get(param_name) for param_name in canonical_order]
            current_key_list = [tuple(v) if isinstance(v, list) else v for v in current_key_list]
            current_key = tuple(current_key_list)
        except TypeError as e:
             fig = px.scatter(title=f"Error forming parameter key: {e}")
             fig.update_layout(**graph_layout)
             return fig, "Error", "Current Variables: Error"

        # Default outputs
        subtitle_text = "Selection Details"
        current_params_text = "Current Variables: N/A"

        # Look up the data for the current parameter combination
        if current_key not in transformed_data_dict:
            fig = px.scatter(x=[0], y=[0], opacity=0)
            fig.add_annotation(text="Untested parameter combination", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(**graph_layout, title="Untested Parameters")
            subtitle_text = "Current selection has not been run."
            current_params_text = "Current Variables: N/A (Untested)"

        else:
            current_metrics, current_exp_info = transformed_data_dict[current_key]
            current_x, current_y = current_metrics[0], current_metrics[1] # Duration, Accuracy
            exp_id = current_exp_info.get('experiment_id', 'unknown')

            # Update Current Params Text
            current_vars_dict = {k: current_params_dict[k] for k in variable_params_keys}
            current_vars_str = ", ".join([f"{k}={v}" for k, v in current_vars_dict.items()])
            current_params_text = f"Current. . . . . . . . . .({exp_id}): {current_y:.4f} with: {{{current_vars_str or 'None'}}}"

            # Create the main scatter plot
            fig_title = f"Experiment: {exp_id}"
            subtitle_text = f"Showing: {x_axis_label}={current_x:.2f}, {y_axis_label}={current_y:.4f}"
            if len(current_metrics) > 2 and not np.isnan(current_metrics[2]):
                subtitle_text += f", {METRIC_LABELS[2]}={current_metrics[2]:.4f}"

            fig = px.scatter(x=[current_x], y=[current_y],
                             labels={'x': x_axis_label, 'y': y_axis_label},
                             title=fig_title)
            fig.update_traces(marker=dict(size=15, color='black', symbol='circle'), name='Current Selection')
            fig.update_layout(**graph_layout)

            # --- Add Ghost Points (Individual Traces) ---
            for idx, param_name in enumerate(variable_params_keys):
                param_color = colors[idx % len(colors)]
                param_index_in_canon = canonical_order.index(param_name)
                current_value = current_key[param_index_in_canon]
                possible_values = variable_params[param_name]

                try: current_value_index = possible_values.index(current_value)
                except ValueError: continue # Skip ghosts if current value not in list

                neighbors = []
                if current_value_index > 0: neighbors.append({'value': possible_values[current_value_index - 1], 'direction': '↓', 'opacity': 0.4}) # Lower opacity for prev
                if current_value_index < len(possible_values) - 1: neighbors.append({'value': possible_values[current_value_index + 1], 'direction': '↑', 'opacity': 0.7}) # Higher opacity for next

                for neighbor in neighbors:
                    neighbor_value, direction, opacity = neighbor['value'], neighbor['direction'], neighbor['opacity']
                    neighbor_key_list = list(current_key)
                    neighbor_key_list[param_index_in_canon] = neighbor_value
                    neighbor_key = tuple(neighbor_key_list)

                    if neighbor_key in transformed_data_dict:
                        neighbor_metrics, neighbor_info = transformed_data_dict[neighbor_key]
                        neighbor_x, neighbor_y = neighbor_metrics[0], neighbor_metrics[1]
                        neighbor_id = neighbor_info.get('experiment_id', 'unknown')

                        # Format neighbor value for display (similar to slider marks)
                        if isinstance(neighbor_value, tuple): label_val_str = f"({len(neighbor_value)} items)"
                        elif isinstance(neighbor_value, float): label_val_str = f"{neighbor_value:.1e}" if abs(neighbor_value)<1e-3 or abs(neighbor_value)>1e4 else f"{neighbor_value:.4g}"
                        else: label_val_str = str(neighbor_value)

                        # Legend label including accuracy
                        label = f"{param_name} {direction} {label_val_str} ({neighbor_y:.3f})" # << UPDATED LEGEND LABEL

                        hover_text = (f"<b>Experiment: {neighbor_id}</b><br>"
                                      f"{x_axis_label}: {neighbor_x:.2f}<br>"
                                      f"{y_axis_label}: {neighbor_y:.4f}<br>"
                                      f"<i>Change: {param_name} {direction} {label_val_str}</i>")
                        if len(neighbor_metrics) > 2 and not np.isnan(neighbor_metrics[2]):
                            hover_text += f"<br>{METRIC_LABELS[2]}: {neighbor_metrics[2]:.4f}"

                        custom_data = f"{param_name}:{neighbor_value}"

                        # << NEW: Add individual scatter trace for each neighbor >>
                        fig.add_scatter(
                            x=[neighbor_x], y=[neighbor_y],
                            mode='markers',
                            marker=dict(
                                size=12, # << CHANGED SIZE
                                color=param_color,
                                opacity=opacity,
                                symbol='circle' # << CHANGED SYMBOL (filled)
                            ),
                            name=label, # Use descriptive name for legend
                            legendgroup=param_name, # Group by parameter in legend
                            hovertext=hover_text,
                            hoverinfo='text',
                            customdata=[custom_data] # Ensure customdata is a list/array
                        )

                        # Add connecting line
                        fig.add_shape(type='line', x0=current_x, y0=current_y, x1=neighbor_x, y1=neighbor_y,
                                      line=dict(color=param_color, width=1.5, dash='dot'), layer='below')

        # Return figure and updated text elements
        return fig, subtitle_text, current_params_text


    @app.callback(
        Output({'type': 'param-slider', 'index': dash.dependencies.ALL}, 'value'),
        Input('output-graph', 'clickData'),
        State({'type': 'param-slider', 'index': dash.dependencies.ALL}, 'id'),
        State({'type': 'slider-mode', 'index': dash.dependencies.ALL}, 'data'),
        prevent_initial_call=True
    )
    def handle_click_data(clickData, slider_ids, slider_modes_list):
        ctx = dash.callback_context
        trigger_id = ctx.triggered_id

        if not isinstance(trigger_id, str) or trigger_id != 'output-graph' or not clickData or not clickData.get('points'):
            return [no_update] * len(slider_ids)

        point = clickData['points'][0]
        if 'customdata' not in point or not point['customdata']:
             # Might be click on main point, ignore for slider update
            return [no_update] * len(slider_ids)

        # Handle cases where customdata might be wrapped in a list/array from add_scatter
        custom_data_raw = point['customdata']
        custom_data_str = custom_data_raw[0] if isinstance(custom_data_raw, (list, np.ndarray)) and len(custom_data_raw) > 0 else custom_data_raw

        if not isinstance(custom_data_str, str) or ':' not in custom_data_str:
             print(f"Warning: Unexpected customdata format: {custom_data_raw}")
             return [no_update] * len(slider_ids)

        try:
            clicked_param_name, value_str = custom_data_str.split(':', 1)
        except ValueError:
             print(f"Error: Failed to split customdata string: {custom_data_str}")
             return [no_update] * len(slider_ids)

        # Process state to build slider_mode_map
        slider_mode_map = {}
        try:
            state_ids_full = ctx.states_list[0] # List of {'id': {'index': 'param', 'type': '...'}, 'property': 'value'}
            state_modes_full = ctx.states_list[1] # List of {'id': {'index': 'param', 'type': '...'}, 'property': 'data', 'value':{...}}

            for i in range(len(state_ids_full)):
                id_item = state_ids_full[i]['id']
                mode_item_value = state_modes_full[i]['value']
                if id_item['type'] == 'param-slider' and state_modes_full[i]['id']['type'] == 'slider-mode':
                     param_name = id_item['index']
                     if isinstance(mode_item_value, dict) and 'use_index' in mode_item_value:
                         slider_mode_map[param_name] = mode_item_value
                     else: print(f"Warning: Invalid mode data structure for {param_name}")
        except Exception as e:
             print(f"Error processing state in click handler: {e}")
             import traceback
             traceback.print_exc()
             return [no_update] * len(slider_ids)

        # Find the target slider and its mode
        target_slider_position = -1
        target_slider_mode = slider_mode_map.get(clicked_param_name)
        for i, id_dict in enumerate(slider_ids):
            if id_dict.get('index') == clicked_param_name:
                target_slider_position = i
                break

        if target_slider_position == -1 or target_slider_mode is None:
            print(f"Warning: Could not find slider or mode for clicked param '{clicked_param_name}'")
            return [no_update] * len(slider_ids)

        # Determine the new value for the slider
        new_slider_value = no_update
        if target_slider_mode.get('use_index'):
            possible_values = target_slider_mode.get('values', [])
            found_index = -1
            # Handle tuple conversion for lookup if needed
            target_value = None
            try:
                 # Attempt to parse value_str back to original type for matching possible_values
                 # This is tricky - safer to compare string representations
                 pass # Keep value_str as string for comparison below
            except: pass # Ignore parsing errors, rely on string match

            for i, v in enumerate(possible_values):
                 # Robust comparison using string representation
                if value_str == str(v):
                    found_index = i
                    break
            if found_index != -1: new_slider_value = found_index
            else: print(f"Warning: Clicked value '{value_str}' not found in index slider options for '{clicked_param_name}'")
        else:
            try: new_slider_value = float(value_str)
            except ValueError: print(f"Warning: Could not convert '{value_str}' to float for numeric slider '{clicked_param_name}'")

        # Construct output
        output_values = [no_update] * len(slider_ids)
        if new_slider_value is not no_update:
            output_values[target_slider_position] = new_slider_value

        return output_values

    return app

# --- Main Execution ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Interactive visualization for cipher classification experiments.")
    parser.add_argument('--max_experiments', type=int, default=None,
                      help="Maximum number of most recent experiments to load.")
    parser.add_argument('--filter', type=str,
                      help="Filter criteria: 'param=value;param2=v1,v2' string or path to JSON config.")
    args = parser.parse_args()

    all_experiments = load_data(max_experiments=args.max_experiments)

    if args.filter:
        filter_config = load_filter_config(args.filter)
        filtered_experiments = apply_filters(all_experiments, filter_config) if filter_config else all_experiments
    else:
        filtered_experiments = all_experiments

    if not filtered_experiments:
        print("No experiments remaining after loading/filtering. Exiting.")
    else:
        print(f"Proceeding to visualize {len(filtered_experiments)} experiments.")
        app = setup_dash_app(filtered_experiments)
        app.run(debug=True)
