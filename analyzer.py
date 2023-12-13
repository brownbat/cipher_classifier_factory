import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import random


# issues transitioning to real data, results in "Error loading graph"
# DEPRECATED NOT USED, use visualization.py

def generate_colors(num_colors, saturation=40, lightness=40):
    """Generate 'num_colors' distinct pastel colors in HSL format."""
    for i in range(num_colors):
        hue = int((360 / num_colors) * i)  # Evenly space the hue
        yield f"hsl({hue}, {saturation}%, {lightness}%)"  # Adjusted Saturation and Lightness for pastel effect


def get_slider_config(df, columns):
    slider_config = {}

    for column in columns:
        if column in df.columns:
            unique_values = sorted(df[column].unique())
            marks = {int(val) if isinstance(val, float) and val.is_integer() else val: str(val) for val in unique_values}
            slider_config[column] = {
                'min': min(unique_values),
                'max': max(unique_values),
                'marks': marks
            }

    return slider_config



def generate_slider_css(num_sliders):
    css_rules = ""
    colors = list(generate_colors(num_sliders))
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


def get_varying_columns(df):
    varying_columns = [col for col in df.columns if len(df[col].unique()) > 1]
    return varying_columns


def print_layout_ids(component, prefix=""):
    # If the component has an 'id', print it
    if getattr(component, 'id', None):
        print(f"{prefix}{component.id}")

    # If the component has children, recursively print their IDs
    if getattr(component, 'children', None):
        if isinstance(component.children, list):
            for child in component.children:
                print_layout_ids(child, prefix + "  ")
        else:
            print_layout_ids(component.children, prefix + "  ")


def setup_dash_app(data):
    app = dash.Dash(__name__)

    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    # Only create sliders for columns that vary
    varying_columns = ['sample_length', 'num_samples']
    slider_config = get_slider_config(df, varying_columns)

    html_output = [dcc.Graph(id='output-graph')]

    for column in varying_columns:
        config = slider_config.get(column, {})
        html_output.append(html.Div([
            html.Label(f'{column.capitalize()}'),
            dcc.Slider(
                id=f'slider-{column}',  # Slider ID
                min=config.get('min', 0),
                max=config.get('max', 10),
                marks=config.get('marks', {0: '0', 10: '10'}),
                value=config.get('min', 0),
                className=f'slider-color-{column}'
            ),
        ], style={'padding': '20px 0px'}))

    # DEBUG
    print_layout_ids(app.layout)
    app.layout = html.Div([html.H1('My App')])
    # app.layout = html.Div(html_output)

    @app.callback(
        Output('output-graph', 'figure'),
        [Input(f'slider-{column}', 'value') for column in varying_columns]
    )
    def update_graph(sample_length, num_samples):
        # Filter the DataFrame based on slider values
        filtered_df = df[(df['sample_length'] == sample_length) & (df['num_samples'] == num_samples)]

        fig = px.scatter(filtered_df, x='final_val_loss', y='final_val_accuracy',
                         color='uid', labels={'x': 'Final Validation Loss', 'y': 'Final Validation Accuracy'})
        fig.update_traces(marker=dict(size=15))

        return fig

    return app




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
    }
]

def transform_data(raw_data):
    transformed_data = {}
    for experiment in raw_data:
        hyperparams = (
            experiment['hyperparams']['batch_size'],
            experiment['hyperparams']['dropout_rate'],
            experiment['hyperparams']['embedding_dim'],
            experiment['hyperparams']['epochs'],
            experiment['hyperparams']['hidden_dim'],
            experiment['hyperparams']['learning_rate'],
            experiment['hyperparams']['num_layers']
        )
        # Selecting training time and final loss as the metrics
        training_time = experiment['metrics']['training_duration']
        final_loss = experiment['metrics']['val_loss'][-1]  # assuming you want the final validation loss

        transformed_data[hyperparams] = (training_time, final_loss)
    return transformed_data


def flatten_data(experiments):
    flattened_data = []
    for exp in experiments:
        flattened = {
            'uid': exp['uid'],
            'num_ciphers': len(exp['ciphers']),
            'num_samples': exp['num_samples'],
            'sample_length': exp['sample_length'],
            # Include other relevant parameters you want to visualize
            'batch_size': exp['batch_size'],
            'dropout_rate': exp['dropout_rate'],
            'embedding_dim': exp['embedding_dim'],
            'epochs': exp['epochs'],
            'hidden_dim': exp['hidden_dim'],
            'learning_rate': exp['learning_rate'],
            'num_layers': exp['num_layers'],
            # Include metrics
            'final_val_accuracy': exp['val_accuracy'][-1],  # Taking the last value as final
            'final_val_loss': exp['val_loss'][-1],
            'training_duration': exp['training_duration']
        }
        flattened_data.append(flattened)
    return flattened_data    

def run_app(data=sample_data):
    flat_sample = flatten_data(data)
    df = pd.DataFrame(flat_sample)

    # Get columns that actually vary
    varying_columns = get_varying_columns(df)

    num_sliders = len(varying_columns)
    colors = list(generate_colors(num_sliders))

    app = setup_dash_app(df)
    app.run_server(debug=True)


if __name__ == '__main__':
    run_app(sample_data)
