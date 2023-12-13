import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import random
import yaml


# TODO: add shapes for lines (compare with working version with tuples)

sample_data = [{'uid': 'exp_1_20231210_220748',
    'ciphers': [
        'english',
        'caesar',
        'vigenere',
        'beaufort',
        'autokey',
        'random_noise',
        'playfair',
        'columnar_transposition'],
    'num_samples': 1000,
    'sample_length': 200,
    'batch_size': 32,
    'dropout_rate': 0.015,
    'embedding_dim': 32,
    'epochs': 3,
    'hidden_dim': 64,
    'learning_rate': 0.002,
    'num_layers': 10,
    'training_duration': 1.4175951480865479,
    'val_accuracy': [0.105, 0.09, 0.14],
    'val_loss': [2.0956461088997975, 2.0918024608067105, 2.070784432547433]
    },
    {
    'uid': 'exp_2_20231210_220751',
    'ciphers': [
        'english',
        'caesar',
        'vigenere',
        'beaufort',
        'autokey',
        'random_noise',
        'playfair',
        'columnar_transposition'],
    'num_samples': 1000,
    'sample_length': 400,
    'batch_size': 32,
    'dropout_rate': 0.015,
    'embedding_dim': 32,
    'epochs': 3,
    'hidden_dim': 64,
    'learning_rate': 0.002,
    'num_layers': 10,
    'training_duration': 2.7952144145965576,
    'val_accuracy':[0.07, 0.07, 0.075],
    'val_loss':[2.095775161470686, 2.0921216351645335, 2.0317570311682567]
    },

    {'uid': 'exp_1_20231210_220748',
    'ciphers': [
        'english',
        'caesar',
        'vigenere',
        'beaufort',
        'autokey',
        'random_noise',
        'playfair',
        'columnar_transposition'],
    'num_samples': 2000,
    'sample_length': 200,
    'batch_size': 32,
    'dropout_rate': 0.015,
    'embedding_dim': 32,
    'epochs': 3,
    'hidden_dim': 64,
    'learning_rate': 0.002,
    'num_layers': 10,
    'training_duration': 1.92,
    'val_accuracy': [0.105, 0.09, 0.24],
    'val_loss': [2.0956461088997975, 2.0918024608067105, 2.03]
    },
    {
    'uid': 'exp_2_20231210_220751',
    'ciphers': [
        'english',
        'caesar',
        'vigenere',
        'beaufort',
        'autokey',
        'random_noise',
        'playfair',
        'columnar_transposition'],
    'num_samples': 2000,
    'sample_length': 400,
    'batch_size': 32,
    'dropout_rate': 0.015,
    'embedding_dim': 32,
    'epochs': 3,
    'hidden_dim': 64,
    'learning_rate': 0.002,
    'num_layers': 10,
    'training_duration': 3.2,
    'val_accuracy':[0.07, 0.07, 0.085],
    'val_loss':[2.095775161470686, 2.0921216351645335, 2.15]
    }
]

def load_subset_of_data(file_path='data/completed_experiments.yaml', max_experiments=100):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
        # Assuming data is a list of experiments
        return data[:max_experiments]

# Load a subset of the data
sample_data = load_subset_of_data(max_experiments=8)
print(sample_data)
input() 


def generate_colors(num_colors, saturation=40, lightness=40):
    """Generate 'num_colors' distinct pastel colors in HSL format and return them as a list."""
    colors = []
    for i in range(num_colors):
        hue = int((360 / num_colors) * i)  # Evenly space the hue
        colors.append(f"hsl({hue}, {saturation}%, {lightness}%)")
    return colors


def generate_slider_css(num_sliders, colors):
    css_rules = ""
    for i in range(num_sliders):
        color = colors[i % len(colors)]
        css_rules += f"""
        .slider-color-{i+1} .rc-slider-track {{ background-color: {color}; }}
        .slider-color-{i+1} .rc-slider-handle {{ border-color: {color}; }}
        """
    return css_rules


def generate_offsets(num_params, base_offset=0.04):
    """Generate a list of (x, y) offset tuples for each parameter."""
    first_offset = (num_params // 2) * base_offset
    offsets = []
    for i in range(num_params):
        # currently moves northeast, can adjust
        offsets.append(((base_offset * i) - first_offset, (base_offset * i) - first_offset))
    return offsets


def write_css_to_file(css_content, filename='assets/custom_styles.css'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        file.write(css_content)


def setup_dash_app(data=sample_data):
    transformed_data = transform_data(data)
    transformed_data_dict = {params: metrics for params, metrics in transformed_data}

    param_value_map = build_param_value_map(transformed_data)
    
    # Create DataFrame for graph scaling
    df = pd.DataFrame([metrics for _, metrics in transformed_data], columns=['output1', 'output2', 'additional_metric'])

    # Identifying variable parameters
    variable_params = {k: v for k, v in param_value_map.items() if len(v) > 1}
    num_params = len(variable_params)

    colors = generate_colors(num_params)
    slider_css = generate_slider_css(num_params, colors)
    write_css_to_file(slider_css)

    app = dash.Dash(__name__)
    html_output = [dcc.Graph(id='output-graph')]

    for idx, (param_name, values) in enumerate(variable_params.items()):
        min_val, max_val = min(values), max(values)
        marks = {int(val): str(val) for val in values}

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

        if current_key in transformed_data_dict:
            current_output1, current_output2, _ = transformed_data_dict[current_key]
            fig = px.scatter(x=[current_output1], y=[current_output2], 
                             labels={'x':'Output 1', 'y':'Output 2'})
            fig.update_traces(marker=dict(size=15, color='black'))

            # Graph scaling based on DataFrame
            min_x, max_x = df['output1'].min(), df['output1'].max()
            min_y, max_y = df['output2'].min(), df['output2'].max()
            margin = 5
            fig.update_layout(
                xaxis_range=[min_x - margin, max_x + margin],
                yaxis_range=[min_y - margin, max_y + margin])

            # Adding ghost points

            # Iterate over variable parameters to identify the next parameters/keys
            for idx, (param_name, values) in enumerate(variable_params.items()):
                correct_idx = list(param_value_map.keys()).index(param_name)
                current_value = current_key[correct_idx]
                next_values = [v for v in values if v > current_value]
                if next_values:
                    next_value = next_values[0]

                    # Construct the next_key by replacing the current parameter value with next_value
                    next_key = list(current_key)
                    next_key[correct_idx] = next_value
                    next_key = tuple(next_key)

                    # Check if the next_key exists in transformed_data_dict
                    if next_key in transformed_data_dict:
                        ghost_output1, ghost_output2, _ = transformed_data_dict[next_key]
                        offsets = generate_offsets(len(variable_params))
                        offset_x, offset_y = offsets[idx % len(offsets)]
                        display_output1 = ghost_output1 + offset_x
                        display_output2 = ghost_output2 + offset_y

                        # Add ghost point to the plot with hover info displaying original values
                        label = f'Ghost Point for Param{idx + 1}'
                        fig.add_scatter(
                            x=[display_output1],
                            y=[display_output2],
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
                                      x0=current_output1+offset_x, y0=current_output2+offset_y,
                                      x1=display_output1, y1=display_output2,
                                      line=dict(color=colors[idx % len(colors)], width=2, dash='dot'))
            return fig
    return app


"""('num_ciphers', 'num_samples', 'sample_length', 'batch_size', 'dropout_rate',
'embedding_dim', 'epochs', 'hidden_dim', 'learning_rate', 'num_layers'):
    ('training_duration', 'val_accuracy', 'val_loss')"""
def transform_data_old(data=sample_data):
    transformed_tuples = []
    for experiment in data:
        params = (
            len(experiment['ciphers']),
            experiment['num_samples'],
            experiment['sample_length'],
            experiment['batch_size'],
            experiment['dropout_rate'],
            experiment['embedding_dim'],
            experiment['epochs'],
            experiment['hidden_dim'],
            experiment['learning_rate'],
            experiment['num_layers']
        )
        training_duration = experiment['training_duration']
        final_accuracy = experiment['val_accuracy'][-1]  # using the final validation accuracy
        final_loss = experiment['val_loss'][-1]  # using the final validation loss

        experiment_tuple = (params, (training_duration, final_accuracy, final_loss))
        transformed_tuples.append(experiment_tuple)

    return transformed_tuples


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


def run_app(data=sample_data):
    num_params = len(list(data[0].keys()))
    app = setup_dash_app(data)
    app.run_server(debug=True)


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


if __name__ == '__main__':
    transformed_data = transform_data(sample_data)
    param_value_map = build_param_value_map(transformed_data)
    run_app()
