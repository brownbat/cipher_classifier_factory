import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import glob
import json
import functools


# allow user to designate which metrics are sliders and which are x/y and which are simply ignored
# display in a text box somewhere those other metrics

# run some experiments repeatedly for percentage success?


USE_MAIN_DATASET = False  # restrict to only the main completed_experiments.json file, rather than pulling in every historical .json


def safe_json_load(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        # Handle empty or invalid JSON file
        print(f"WARNING: {file_path} is empty or invalid. An empty list will be used.")
        return []
    except FileNotFoundError:
        # Handle file not found and create a new empty file
        print(f"WARNING: {file_path} not found. Creating a new file.")
        with open(file_path, 'w') as file:
            json.dump([], file)
        return []


def filter_experiments(experiments, filter_criteria):
    """
    Filter out experiments that do not meet the filter criteria.

    Args:
    - experiments (list): List of experiment dictionaries.
    - filter_criteria (dict): Dictionary of parameters to accept.

    Returns:
    - (list): Filtered list of experiments.
    """
    filtered_experiments = []
    for exp in experiments:
        include_exp = True
        
        # Check that experiment has required data
        if 'hyperparams' not in exp or 'data_params' not in exp or 'metrics' not in exp:
            include_exp = False
            continue
            
        # Check that experiment has complete metrics
        if not exp['metrics'].get('val_accuracy') or not exp['metrics'].get('training_duration'):
            include_exp = False
            continue
            
        # Check filter criteria
        for param, legal_values in filter_criteria.items():
            if param in exp['hyperparams'] and exp['hyperparams'][param] not in legal_values:
                include_exp = False
                break
            if param in exp['data_params'] and exp['data_params'][param] not in legal_values:
                include_exp = False
                break
        
        if include_exp:
            filtered_experiments.append(exp)
            
    print(f"Found {len(filtered_experiments)} valid experiments out of {len(experiments)} total")
    return filtered_experiments


def calculate_avg_accuracy_per_value(data, param_name):
    # Group data by the value of the hyperparameter and collect accuracies
    value_accuracy_dict = {}
    for experiment in data:
        # Check both 'data_params' and 'hyperparams' for the parameter
        if param_name in experiment['data_params']:
            value = experiment['data_params'][param_name]
        elif param_name in experiment['hyperparams']:
            value = experiment['hyperparams'][param_name]
        else:
            # Handle cases where the parameter is missing
            continue  # or raise an error, or handle as needed
        # Obtain the last known accuracy value, ensuring it is not None
        accuracy = experiment['metrics']['val_accuracy'][-1] if experiment['metrics']['val_accuracy'] else None
        if accuracy is not None and accuracy > 0:  # Only consider non-null accuracies
            if value not in value_accuracy_dict:
                value_accuracy_dict[value] = []
            value_accuracy_dict[value].append(accuracy)

    # Calculate average accuracy for each value, excluding any null results
    avg_accuracy_per_value = {
        value: sum(accuracies) / len(accuracies) 
        for value, accuracies in value_accuracy_dict.items() if accuracies  # Ensure there are accuracies to average
    }
    return avg_accuracy_per_value



def generate_colors(num_colors, saturation=40, lightness=40):
    """Generate 'num_colors' distinct pastel colors in HSL format and return them as a list."""
    colors = []
    for i in range(num_colors):
        hue = int((360 / num_colors) * i)  # Evenly space the hue
        colors.append(f"hsl({hue}, {saturation}%, {lightness}%)")
    return colors


def generate_slider_marks(data, param_name):
    avg_accuracy_per_value = calculate_avg_accuracy_per_value(data, param_name)
    marks = {}
    for value, accuracy in avg_accuracy_per_value.items():
        # Format the accuracy as a string with 3 decimal places
        accuracy_str = f"{accuracy:.3f}"
        # Set the label with the value and average accuracy (avg acc)
        marks[value] = f"{value} ({accuracy_str})"
    return marks


def generate_slider_css(num_sliders, colors):
    css_rules = ""
    for i in range(num_sliders):
        color = colors[i % len(colors)]
        css_rules += f"""
        .slider-color-{i+1} .rc-slider-track {{ background-color: {color}; }}
        .slider-color-{i+1} .rc-slider-handle {{ border-color: {color}; }}
        """
    return css_rules


def generate_offsets(variable_params, offset_coefficient=0.1):
    """Generate a list of (x, y) offset tuples for each parameter."""
    # Assuming 'output1' and 'output2' are the metrics for x and y axes, respectively
    output1_range = max(variable_params['output1']) - min(variable_params['output1'])
    output2_range = max(variable_params['output2']) - min(variable_params['output2'])

    base_offset_x = offset_coefficient * output1_range
    base_offset_y = offset_coefficient * output2_range

    offsets = []
    for i in range(len(variable_params)):
        offset_x = (base_offset_x * i) - ((len(variable_params) // 2) * base_offset_x)
        offset_y = (base_offset_y * i) - ((len(variable_params) // 2) * base_offset_y)
        offsets.append((offset_x, offset_y))
    return offsets


def write_css_to_file(css_content, filename='assets/custom_styles.css'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(css_content)


def setup_dash_app(data=None):
    if data is None or not data:
        print("Warning: No experiment data available to visualize")
        # Create a simple app with a message
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H3("No experiment data available to visualize", style={'text-align': 'center'}),
            html.P("Complete some experiments first, then run visualization.py again.")
        ])
        return app
    
    # Display experiment count
    experiment_count = len(data)
    print(f"Visualizing {experiment_count} experiments")

    transformed_data = transform_data(data)
    if not transformed_data:
        print("Warning: No valid experiment data after transformation")
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H3("No valid experiment data to visualize", style={'text-align': 'center'}),
            html.P("Ensure your experiments have completed successfully.")
        ])
        return app

    # Create a dictionary mapping parameter tuples to (metrics, experiment_info)
    transformed_data_dict = {params: (metrics, exp_info) for params, metrics, exp_info in transformed_data}
    
    param_value_map = build_param_value_map(transformed_data)
    num_metrics = len(transformed_data[0][1]) if transformed_data else 0
    column_names = [f'output{i + 1}' for i in range(num_metrics)]
    
    # Create DataFrame for graph scaling
    df = pd.DataFrame([metrics for _, metrics, _ in transformed_data], columns=column_names)

    # Identify unchanging parameters
    constant_params = {k: v for k, v in param_value_map.items() if len(v) == 1}

    # Identifying variable parameters
    variable_params = {k: v for k, v in param_value_map.items() if len(v) > 1}
    num_params = len(variable_params)

    # Find most accurate parameters
    best_params_tuple = None
    best_metric = -float("inf")  
    best_exp_info = {}

    if transformed_data:
        for params_tuple, metrics_tuple, exp_info in transformed_data:
            if len(metrics_tuple) > 1:  # Ensure we have at least accuracy metric
                value_to_track = metrics_tuple[1]  
                if value_to_track > best_metric:
                    best_params_tuple = params_tuple
                    best_metric = value_to_track
                    best_exp_info = exp_info

    best_params_dict = {}
    if best_params_tuple:
        for i, param_name in enumerate(param_value_map.keys()):
            if i < len(best_params_tuple):
                best_params_dict[param_name] = best_params_tuple[i]

    # Reduce to the settings that are variable
    best_variables = {}
    for param_name, value in best_params_dict.items():
      if param_name in variable_params:
        best_variables[param_name] = value

    # Construct legend content
    constant_params_info = f"Fixed Parameters: {', '.join([f'{k}: {list(v)[0]}' for k, v in constant_params.items()])}"
    best_exp_id = best_exp_info.get('experiment_id', 'unknown')
    best_exp_uid = best_exp_info.get('uid', 'unknown')
    
    highest_accuracy_info = (f"Highest Accuracy: {best_exp_id} - "
                            f"Parameters: {best_variables} with Accuracy of {best_metric:.4f}")

    colors = generate_colors(num_params)
    slider_css = generate_slider_css(num_params, colors)
    write_css_to_file(slider_css)

    # Extracting metric values for graph scaling
    metrics = [values for _, values, _ in transformed_data]
    accuracy_values = [m[1] for m in metrics]  # Assuming accuracy is the second metric
    training_duration_values = [m[0] for m in metrics]  # Assuming training duration is the first metric

    min_accuracy, max_accuracy = min(accuracy_values), max(accuracy_values)
    min_duration, max_duration = min(training_duration_values), max(training_duration_values)

    # Adding a buffer for better visualization
    accuracy_buffer = (max_accuracy - min_accuracy) * 0.1
    duration_buffer = (max_duration - min_duration) * 0.1

    graph_layout = {
        'xaxis': {'range': [min_duration - duration_buffer, max_duration + duration_buffer]},
        'yaxis': {'range': [min_accuracy - accuracy_buffer, max_accuracy + accuracy_buffer]},
        # Include other layout properties as needed
    }

    app = dash.Dash(__name__)
    html_output = [
        html.H3(f"Transformer Experiments Dashboard - Showing {experiment_count} Experiments", 
                style={'text-align': 'center', 'margin-bottom': '20px'})
    ]
    
    # Title for the app
    ciphers_list = data[0]['data_params']['ciphers']
    title = "Distinguishing between " + ', '.join(ciphers_list)
    html_output.append(html.Div([
        html.H4(
            "Distinguishing between " + ', '.join(ciphers_list))],
            style={'text-align': 'center'}))

    # Adding the graph (first a line br)
    html_output.append(html.Br())
    html_output.append(dcc.Graph(id='output-graph', figure={'layout': graph_layout}))
    html_output.append(html.Br())

    # Add legend to HTML output
    html_output.append(html.Div([
        html.P(constant_params_info),
        html.P(highest_accuracy_info)
    ], style={'padding': '10px'}))

    for idx, (param_name, values) in enumerate(variable_params.items()):
        min_val, max_val = min(values), max(values)
        # marks = {val if isinstance(val, int) else float(val): str(val) for val in values}
        marks = generate_slider_marks(data, param_name)

        html_output.append(html.Div([
            html.Label(f'{param_name.capitalize()}'),
            dcc.Slider(
                id=f'slider-{param_name}',
                min=min_val,
                max=max_val,
                value=min_val,
                marks=marks,
                step=None,
                className=f'slider-color-{idx+1}'
            ),
        ], style={'padding': '20px 0px'}))

    # Add a hidden div to store clicked data state
    html_output.append(html.Div(id='clicked-data-store', style={'display': 'none'}))
    
    app.layout = html.Div(html_output)

    # Callback for handling click events on ghost points
    @app.callback(
        [Output(f'slider-{param_name}', 'value') for param_name in variable_params.keys()],
        [Input('output-graph', 'clickData')],
        prevent_initial_call=True
    )
    def handle_click_data(clickData):
        if not clickData or 'points' not in clickData or not clickData['points']:
            # No click data or no points in click data
            return [dash.no_update] * len(variable_params)
            
        point = clickData['points'][0]
        
        # Only process if point has customdata (ghost point)
        if 'customdata' not in point or not point['customdata']:
            return [dash.no_update] * len(variable_params)
            
        # Get customdata string (format: "param_name:param_value")
        custom_data = point['customdata'][0] if isinstance(point['customdata'], list) else point['customdata']
        
        # Check if this is a valid customdata string
        if not isinstance(custom_data, str) or ':' not in custom_data:
            return [dash.no_update] * len(variable_params)
            
        # Parse parameter name and value from custom data
        try:
            clicked_param_name, value_str = custom_data.split(':', 1)
            # Convert value to the appropriate type
            if '.' in value_str:
                clicked_param_value = float(value_str)
            else:
                clicked_param_value = int(value_str)
        except (ValueError, TypeError):
            return [dash.no_update] * len(variable_params)
        
        # Create the list of new slider values
        new_values = []
        for param_name in variable_params.keys():
            if param_name == clicked_param_name:
                # Update the clicked parameter's slider value
                new_values.append(clicked_param_value)
            else:
                # Keep other sliders unchanged
                new_values.append(dash.no_update)
                
        return new_values

    @app.callback(
        Output('output-graph', 'figure'),
        [Input(f'slider-{param_name}', 'value') for param_name in variable_params.keys()]
    )
    def update_graph(*slider_values):
        # Extract the first experiment's parameters as a template for constant values
        constant_values_template = transformed_data[0][0]

        # Map of variable parameter names to their slider values
        variable_values = dict(zip(variable_params.keys(), slider_values))

        # Reconstruct the complete key
        current_key = tuple(variable_values.get(param_name, constant_values_template[idx]) 
                            for idx, param_name in enumerate(param_value_map.keys()))
        
        if current_key not in transformed_data_dict:
            # Handle case where the parameter set hasn't been tested
            fig = px.scatter(x=[0], y=[0], labels={'x':'Training Time', 'y':'Accuracy'})
            fig.add_annotation(text="Untested parameter set",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="blue"))
        else:
            # Handle case where we have data for this parameter set
            (current_metrics, current_exp_info_dict) = transformed_data_dict[current_key]
            current_output1, current_output2 = current_metrics[0], current_metrics[1]
            
            # Extract experiment details for title
            exp_id = current_exp_info_dict.get('experiment_id', 'unknown')
            exp_uid = current_exp_info_dict.get('uid', 'unknown')
            
            # Create a detailed title with experiment info
            # Show just the experiment_id which should now be the date-based ID
            fig_title = f"Experiment: {exp_id} - Training: {current_output1:.2f}s, Accuracy: {current_output2:.4f}"
            
            fig = px.scatter(x=[current_output1], y=[current_output2], 
                             labels={'x':'Training Time', 'y':'Accuracy'},
                             title=fig_title)
            fig.update_traces(marker=dict(size=15, color='black'))

            # Graph scaling based on DataFrame
            min_x, max_x = df['output1'].min(), df['output1'].max()
            min_y, max_y = df['output2'].min(), df['output2'].max()
            margin_x = 0.1 * (max_x - min_x)
            margin_y = 0.1 * (max_y - min_y)
            fig.update_layout(
                xaxis_range=[min_x - margin_x, max_x + margin_x],
                yaxis_range=[min_y - margin_y, max_y + margin_y])

            # Adding ghost points

            # Iterate over variable parameters to identify both higher and lower parameter values
            for idx, (param_name, values) in enumerate(variable_params.items()):
                sorted_values = sorted(values)
                
                correct_idx = list(param_value_map.keys()).index(param_name)
                current_value = current_key[correct_idx]
                
                # Find next higher value (if any)
                next_values = [v for v in sorted_values if v > current_value]
                if next_values:
                    next_value = next_values[0]

                    # Construct the next_key by replacing the current parameter value with next_value
                    next_key = list(current_key)
                    next_key[correct_idx] = next_value
                    next_key = tuple(next_key)

                    # Check if the next_key exists in transformed_data_dict
                    if next_key in transformed_data_dict:
                        (next_metrics, next_exp_info) = transformed_data_dict[next_key]
                        ghost_output1, ghost_output2 = next_metrics[0], next_metrics[1]
                        
                        # Get experiment ID for ghost point
                        ghost_exp_id = next_exp_info.get('experiment_id', 'unknown')
                        
                        # Add ghost point to the plot with hover info displaying original values
                        label = f'{ghost_exp_id}: {param_name}={next_value}'
                        hover_text = (f"Experiment: {ghost_exp_id}<br>" + 
                                    f"Training Time: {ghost_output1:.2f}s<br>" +
                                    f"Accuracy: {ghost_output2:.4f}<br>" +
                                    f"Change: {param_name} ↑ {current_value} → {next_value}")
                        
                        # Store parameter information in customdata for click handling
                        custom_data = [f"{param_name}:{next_value}"]
                        
                        fig.add_scatter(
                            x=[ghost_output1],
                            y=[ghost_output2],
                            mode='markers',
                            marker=dict(
                                size=9,
                                color=colors[idx % len(colors)],
                                opacity=0.4),  # Higher opacity for higher values
                            name=label,
                            hovertext=hover_text,
                            hoverinfo='text',
                            customdata=custom_data)
                        # Add lines to ghost points for visibility
                        fig.add_shape(type='line',
                                      x0=current_output1, y0=current_output2,
                                      x1=ghost_output1, y1=ghost_output2,
                                      line=dict(color=colors[idx % len(colors)], width=2, dash='dot'))
                
                # Find previous lower value (if any)
                prev_values = [v for v in sorted_values if v < current_value]
                if prev_values:
                    prev_value = prev_values[-1]  # Get the closest lower value
                    
                    # Construct the prev_key by replacing the current parameter value with prev_value
                    prev_key = list(current_key)
                    prev_key[correct_idx] = prev_value
                    prev_key = tuple(prev_key)
                    
                    # Check if the prev_key exists in transformed_data_dict
                    if prev_key in transformed_data_dict:
                        (prev_metrics, prev_exp_info) = transformed_data_dict[prev_key]
                        ghost_output1, ghost_output2 = prev_metrics[0], prev_metrics[1]
                        
                        # Get experiment ID for ghost point
                        ghost_exp_id = prev_exp_info.get('experiment_id', 'unknown')
                        
                        # Add ghost point to the plot with hover info displaying original values
                        label = f'{ghost_exp_id}: {param_name}={prev_value}'
                        hover_text = (f"Experiment: {ghost_exp_id}<br>" + 
                                    f"Training Time: {ghost_output1:.2f}s<br>" +
                                    f"Accuracy: {ghost_output2:.4f}<br>" +
                                    f"Change: {param_name} ↓ {current_value} → {prev_value}")
                        
                        # Store parameter information in customdata for click handling
                        # We'll use a string format that can be parsed reliably, inside an array
                        custom_data = [f"{param_name}:{prev_value}"]
                        
                        fig.add_scatter(
                            x=[ghost_output1],
                            y=[ghost_output2],
                            mode='markers',
                            marker=dict(
                                size=9,
                                color=colors[idx % len(colors)],
                                opacity=0.2),  # Lower opacity for lower values
                            name=label,
                            hovertext=hover_text,
                            hoverinfo='text',
                            customdata=custom_data)
                        # Add lines to ghost points for visibility
                        fig.add_shape(type='line',
                                      x0=current_output1, y0=current_output2,
                                      x1=ghost_output1, y1=ghost_output2,
                                      line=dict(color=colors[idx % len(colors)], width=2, dash='dot'))
        return fig
    return app


# Cache for transformed data
_transform_cache = {}

def transform_data(data, use_cache=True):
    """
    Transform the experiment data into a format suitable for visualization.
    Dynamically extracts parameters and metrics from the experiment data.
    
    Args:
        data: List of experiment dictionaries
        use_cache: Whether to use cached results if available
        
    Returns:
        List of (parameters, metrics, experiment_info) tuples where experiment_info is a dict
    """
    # Use a simple cache key based on the number of items and first/last item hash
    if data:
        cache_key = (len(data), id(data[0]), id(data[-1]))
        if use_cache and cache_key in _transform_cache:
            return _transform_cache[cache_key]
    
    transformed_tuples = []
    for experiment in data:
        params = []
        
        # Dynamically extracting parameters from 'data_params'
        if 'data_params' in experiment:
            for key, value in sorted(experiment['data_params'].items()):
                if isinstance(value, list):
                    # For lists (like ciphers), store their length
                    params.append(len(value))
                else:
                    # For other types, store the value directly
                    params.append(value)
        
        # Extract hyperparameters
        if 'hyperparams' in experiment:
            # Common hyperparameters
            for key in ['batch_size', 'dropout_rate', 'epochs', 'learning_rate']:
                if key in experiment['hyperparams']:
                    params.append(experiment['hyperparams'][key])
                else:
                    params.append(None)  # Add placeholder for missing values
            
            # Transformer hyperparameters
            for key in ['d_model', 'nhead', 'num_encoder_layers', 'dim_feedforward']:
                if key in experiment['hyperparams']:
                    params.append(experiment['hyperparams'][key])
                else:
                    params.append(None)
        
        # Extracting metrics
        metrics = []
        if 'metrics' in experiment:
            for key in ['training_duration', 'val_accuracy', 'val_loss']:
                if key in experiment['metrics']:
                    value = experiment['metrics'][key]
                    # For 'val_accuracy' and 'val_loss', use the last value
                    if key in ['val_accuracy', 'val_loss'] and isinstance(value, list):
                        metrics.append(value[-1])
                    else:
                        metrics.append(value)
        
        # Extract experiment info for display
        experiment_info = {
            'experiment_id': experiment.get('experiment_id', 'unknown'),
            'uid': experiment.get('uid', 'unknown')
        }

        experiment_tuple = (tuple(params), tuple(metrics), experiment_info)
        transformed_tuples.append(experiment_tuple)
    
    # Cache the result
    if data:
        _transform_cache[cache_key] = transformed_tuples
    
    return transformed_tuples


def load_and_concatenate_json_files(pattern):
    all_data = []
    for file_name in glob.glob(pattern):
        data = safe_json_load(file_name)
        all_data.extend(data)
    return all_data


def load_data(max_experiments=None, use_strict_filters=False):
    """
    Load experiment data from JSON files and optionally limit to a maximum number of experiments.
    
    Args:
        max_experiments: Optional maximum number of experiments to include
        use_strict_filters: If True, use strict parameter filters; if False, use looser filters
        
    Returns:
        List of filtered experiment dictionaries
    """
    if USE_MAIN_DATASET:
        data = safe_json_load('data/completed_experiments.json')
    else:  # use all completed jsons in the data folder
        json_files_pattern = "data/completed_experiments*.json"
        data = load_and_concatenate_json_files(json_files_pattern)

    print(f"Loaded {len(data)} total experiments from JSON files")
    
    if use_strict_filters:
        # Define strict filter parameters for transformer models (exact match)
        params = {
            'ciphers': [[
                "english",
                "caesar",
                "vigenere",
                "beaufort",
                "autokey",
                "random_noise",
                "playfair",
                "bifid",
                "fractionated_morse",
                "columnar_transposition"]],
            'num_samples': [100000],
            'sample_length': [500],
            'epochs': [30],
            'd_model': [128, 256],
            'nhead': [4, 8],
            'num_encoder_layers': [2, 4],
            'dim_feedforward': [512, 1024],
            'batch_size': [32, 64],
            'dropout_rate': [0.1, 0.2],
            'learning_rate': [1e-4, 3e-4]
        }
    else:
        # Define loose filter parameters - just check that metrics exist
        params = {}
    
    data = filter_experiments(data, params)
    
    # Limit to max_experiments if specified
    if max_experiments is not None and max_experiments > 0:
        data = data[:max_experiments]
        
    return data


def build_param_value_map(transformed_data):
    """
    Build a map of parameter names to their possible values based on transformed data.
    
    Args:
        transformed_data: List of (parameters, metrics, exp_info) tuples
        
    Returns:
        Dict mapping parameter names to sets of possible values
    """
    # Define parameter names for transformer models
    param_names = [
        # Data parameters
        'num_ciphers', 'num_samples', 'sample_length',
        # Common hyperparameters
        'batch_size', 'dropout_rate', 'epochs', 'learning_rate',
        # Transformer hyperparameters
        'd_model', 'nhead', 'num_encoder_layers', 'dim_feedforward'
    ]
    
    # Create a map with empty sets
    param_value_map = {param_name: set() for param_name in param_names}
    
    # Fill the map with values from the transformed data
    for experiment in transformed_data:
        params = experiment[0]
        for idx, param_value in enumerate(params):
            if idx < len(param_names):
                param_name = param_names[idx]
                if param_value is not None:  # Only add non-None values
                    param_value_map[param_name].add(param_value)
    
    return param_value_map


# Parse command line arguments
import argparse

def parse_filter_string(filter_str):
    """
    Parse a filter string into a structured filter dictionary.
    Format: "param1=val1,val2;param2=val3,val4"
    
    Example: "epochs=20,40;d_model=128;dropout_rate=0.1"
    Returns: {
        'hyperparams': {
            'epochs': [20, 40],
            'd_model': [128],
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


def apply_filters(experiments, filter_params):
    """
    Filter experiments based on parameter criteria.
    
    Args:
        experiments: List of experiment dictionaries
        filter_params: Dictionary with 'hyperparams' and/or 'data_params' keys
                      containing filter criteria
    
    Returns:
        Filtered list of experiments
    """
    filtered = []
    
    for exp in experiments:
        include = True
        
        # Filter by hyperparameters
        if 'hyperparams' in filter_params:
            for param, values in filter_params['hyperparams'].items():
                if param in exp.get('hyperparams', {}):
                    if exp['hyperparams'][param] not in values:
                        include = False
                        break
        
        # Filter by data parameters
        if include and 'data_params' in filter_params:
            for param, values in filter_params['data_params'].items():
                if param in exp.get('data_params', {}):
                    if exp['data_params'][param] not in values:
                        include = False
                        break
        
        if include:
            filtered.append(exp)
    
    return filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualization tool for transformer experiment results")
    parser.add_argument('--max_experiments', type=int, default=100,
                      help="Maximum number of experiments to include")
    parser.add_argument('--strict', action='store_true',
                      help="Use strict filtering for experiments")
    parser.add_argument('--filter', type=str,
                      help="Filter criteria: either a filter string (e.g., \"epochs=40;d_model=128,256\") or path to a JSON config file")
    args = parser.parse_args()
    
    # Load data with specified limit
    data = load_data(max_experiments=args.max_experiments, use_strict_filters=args.strict)
    
    # Apply filter if provided
    if args.filter:
        filter_params = load_filter_config(args.filter)
        if filter_params:
            print(f"Applying filters: {filter_params}")
            data = apply_filters(data, filter_params)
            print(f"Showing {len(data)} experiments after filtering")
    
    app = setup_dash_app(data)
    
    # Run the app
    app.run(debug=True)

