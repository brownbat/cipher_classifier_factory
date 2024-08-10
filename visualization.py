import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
# from researcher import safe_json_load
import glob
import json

# TODO:
# 1. clicking ghost points should adjust settings (using a click handler)
# 2. greedily load nearby points only for efficiency
# 3. load multiple files
# 4. host in the cloud
# 5. provide some hint as to how much each param matters / does not
# - calculate average and std dev for all experiments at that param's
#   current value, show a graph of that beneath each slider?
# move from json to a database?

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
        for param, legal_values in filter_criteria.items():
            if param in exp['hyperparams'] and exp['hyperparams'][param] not in legal_values:
                include_exp = False
                break
            if param in exp['data_params'] and exp['data_params'][param] not in legal_values:
                include_exp = False
                break
        if include_exp:
            filtered_experiments.append(exp)
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


def load_subset_of_data(file_path='data/completed_experiments-subset.json', max_experiments=100):
    with open(file_path, 'r') as file:
        data = safe_json_load(file)
        # Assuming data is a list of experiments
        return data[:max_experiments]

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
    if data is None:
        raise ValueError("No data specified for setup_dash_app()")

    transformed_data = transform_data(data)
    transformed_data_dict = {params: metrics for params, metrics in transformed_data}
    param_value_map = build_param_value_map(transformed_data)
    num_metrics = len(transformed_data[0][1]) if transformed_data else 0
    column_names = [f'output{i + 1}' for i in range(num_metrics)]
    
    # Create DataFrame for graph scaling
    df = pd.DataFrame([metrics for _, metrics in transformed_data], columns=column_names)

    # Identify unchanging parameters
    constant_params = {k: v for k, v in param_value_map.items() if len(v) == 1}

    # Identifying variable parameters
    variable_params = {k: v for k, v in param_value_map.items() if len(v) > 1}
    num_params = len(variable_params)

    # Find most accurate parameters
    best_params_tuple = None
    best_metric = -float("inf")  

    for params_tuple, metrics_tuple in transformed_data:
      value_to_track = metrics_tuple[1]  
      if value_to_track > best_metric:
        best_params_tuple = params_tuple
        best_metric = value_to_track

    best_params_dict = {}
    for i, param_name in enumerate(param_value_map.keys()):
      best_params_dict[param_name] = best_params_tuple[i]

    # Reduce to the settings that are variable
    best_variables = {}
    for param_name, value in best_params_dict.items():
      if param_name in variable_params:
        best_variables[param_name] = value

    # Construct legend content
    constant_params_info = f"Fixed Parameters: {', '.join([f'{k}: {list(v)[0]}' for k, v in constant_params.items()])}"
    highest_accuracy_info = f"Highest Accuracy Parameters: { best_variables } with Accuracy of {best_metric}"

    colors = generate_colors(num_params)
    slider_css = generate_slider_css(num_params, colors)
    write_css_to_file(slider_css)

    # Extracting metric values for graph scaling
    metrics = [values for _, values in transformed_data]
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
    html_output = []
    
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

    app.layout = html.Div(html_output)

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
            fig = px.scatter(x=[0], y=[0], labels={'x':'Training Time', 'y':'Accuracy'})
            fig.add_annotation(text="Untested parameter set",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="blue"))
        if current_key in transformed_data_dict:
            current_output1, current_output2, _ = transformed_data_dict[current_key]
            fig = px.scatter(x=[current_output1], y=[current_output2], 
                             labels={'x':'Training Time', 'y':'Accuracy'})
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

            # Iterate over variable parameters to identify the next parameters/keys
            for idx, (param_name, values) in enumerate(variable_params.items()):
                sorted_values = sorted(values)
                
                correct_idx = list(param_value_map.keys()).index(param_name)
                current_value = current_key[correct_idx]
                next_values = [v for v in sorted_values if v > current_value]
                if next_values:
                    next_value = next_values[0]

                    # Construct the next_key by replacing the current parameter value with next_value
                    next_key = list(current_key)
                    next_key[correct_idx] = next_value
                    next_key = tuple(next_key)

                    # Check if the next_key exists in transformed_data_dict
                    if next_key in transformed_data_dict:
                        ghost_output1, ghost_output2, _ = transformed_data_dict[next_key]
                        output_values = {'output1':df['output1'], 'output2':df['output2']}

                        # Add ghost point to the plot with hover info displaying original values
                        label = f'Ghost Point for {param_name} (Param{idx + 1})'
                        fig.add_scatter(
                            x=[ghost_output1],  # replace with offset values if necessary
                            y=[ghost_output2],
                            mode='markers',
                            marker=dict(
                                size=9,
                                color=colors[idx % len(colors)],
                                opacity=0.4),
                            name=label,
                            hovertext=f'{label}: ({ghost_output1}, {ghost_output2})',
                            hoverinfo='text')
                        # Add lines to ghost points for visibility
                        fig.add_shape(type='line',
                                      x0=current_output1, y0=current_output2,  # add offsets here as appropriate
                                      x1=ghost_output1, y1=ghost_output2,
                                      line=dict(color=colors[idx % len(colors)], width=2, dash='dot'))
        return fig
    return app


def transform_data(data):
    transformed_tuples = []
    for experiment in data:
        # Extracting parameters
        params = (
            len(experiment['data_params']['ciphers']),
            experiment['data_params']['num_samples'],
            experiment['data_params']['sample_length'],
            experiment['hyperparams']['batch_size'],
            experiment['hyperparams']['dropout_rate'],
            experiment['hyperparams']['embedding_dim'],
            experiment['hyperparams']['epochs'],
            experiment['hyperparams']['hidden_dim'],
            experiment['hyperparams']['learning_rate'],
            experiment['hyperparams']['num_layers']
        )
        
        # Extracting metrics
        training_duration = experiment['metrics']['training_duration']
        final_accuracy = experiment['metrics']['val_accuracy'][-1]  # using the final validation accuracy
        final_loss = experiment['metrics']['val_loss'][-1]  # using the final validation loss

        experiment_tuple = (params, (training_duration, final_accuracy, final_loss))
        transformed_tuples.append(experiment_tuple)

    return transformed_tuples


def transform_data_dynamic(data):
    transformed_tuples = []
    for experiment in data:
        params = []
        # Dynamically extracting parameters from 'data_params' and 'hyperparams'
        for param_category in ['data_params', 'hyperparams']:
            if param_category in experiment:
                for key, value in experiment[param_category].items():
                    if isinstance(value, list):
                        # For lists (like ciphers), store their length
                        params.append(len(value))
                    else:
                        # For other types, store the value directly
                        params.append(value)

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

        experiment_tuple = (tuple(params), tuple(metrics))
        transformed_tuples.append(experiment_tuple)

    return transformed_tuples


def load_and_concatenate_json_files(pattern):
    all_data = []
    for file_name in glob.glob(pattern):
        data = safe_json_load(file_name)
        all_data.extend(data)
    return all_data


def load_data():
    if USE_MAIN_DATASET:
        data = safe_json_load('data/completed_experiments.json')
    else:  # use all completed jsons in the data folder
        json_files_pattern = "data/completed_experiments*.json"
        data = load_and_concatenate_json_files(json_files_pattern)

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
        'num_layers': [64, 128, 256],
        'batch_size': [16, 32, 64],
        'embedding_dim': [16, 32, 64],
        'hidden_dim': [128, 256, 512],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.002, 0.003, 0.004]
    }
    data = filter_experiments(data, params)
    return data


def build_param_value_map_dynamic(transformed_data):
    param_value_map = {}
    for params, _ in transformed_data:
        for idx, param in enumerate(params):
            # Assuming each parameter can be uniquely identified by its index
            param_name = f"param_{idx}"
            if param_name not in param_value_map:
                param_value_map[param_name] = set()
            param_value_map[param_name].add(param)
    return param_value_map


def build_param_value_map(transformed_data):
    param_value_map = {
        'num_ciphers': set(),
        'num_samples': set(),
        'sample_length': set(),
        'batch_size': set(),
        'dropout_rate': set(),
        'embedding_dim': set(),
        'epochs': set(),
        'hidden_dim': set(),
        'learning_rate': set(),
        'num_layers': set()
    }

    for experiment in transformed_data:
        hyperparams = experiment[0]
        for idx, param_name in enumerate(param_value_map.keys()):
            param_value_map[param_name].add(hyperparams[idx])

    return param_value_map


data = load_data()
app = setup_dash_app(data)

if __name__ == '__main__':
    app.run_server(debug=True)

