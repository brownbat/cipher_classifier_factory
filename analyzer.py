import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import random


# issues transitioning to real data, results in "Error loading graph"


def generate_colors(num_colors, saturation=40, lightness=40):
    """Generate 'num_colors' distinct pastel colors in HSL format."""
    for i in range(num_colors):
        hue = int((360 / num_colors) * i)  # Evenly space the hue
        yield f"hsl({hue}, {saturation}%, {lightness}%)"  # Adjusted Saturation and Lightness for pastel effect


def get_slider_config(df):
    slider_config = {}

    for column in df.columns:
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


def setup_dash_app(data):
    app = dash.Dash(__name__)

    # Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data

    varying_columns = get_varying_columns(df)
    slider_config = get_slider_config(df)

    num_params = len(varying_columns)

    html_output = [dcc.Graph(id='output-graph')]

    for idx, column in enumerate(varying_columns):
        config = slider_config[column]
        html_output.append(html.Div([
            html.Label(f'{column.capitalize()}'),
            dcc.Slider(
                id=f'slider-{column}',
                min=config['min'],
                max=config['max'],
                marks=config['marks'],
                value=config['min'],
                className=f'slider-color-{idx+1}'
            ),
        ], style={'padding': '20px 0px'}))

    app.layout = html.Div(html_output)

    @app.callback(
        Output('output-graph', 'figure'),
        [Input(f'slider-{column}', 'value') for column in varying_columns]
    )
    def update_graph(*params):
        current_key = tuple(params)
        current_output1, current_output2 = data[current_key]
        fig = px.scatter(x=[current_output1], y=[current_output2], 
                         labels={'x':'Output 1', 'y':'Output 2'})
        fig.update_traces(marker=dict(size=15, color='black'))

        offsets = generate_offsets(num_params)

        colors = list(generate_colors(len(params)))
        for i, param in enumerate(params):
            next_param = list(params)
            if next_param[i] < 10:
                next_param[i] += 5
                next_key = tuple(next_param)
                if next_key in data:
                    ghost_output1, ghost_output2 = data[next_key]
                    # Adding random jitter
                    offset_x, offset_y = offsets[i % len(offsets)]
                    display_output1 = ghost_output1 + offset_x
                    display_output2 = ghost_output2 + offset_y

                    # Add ghost point to the plot with hover info displaying original values
                    label = f'Ghost Point for Param{i + 1}'
                    fig.add_scatter(
                        x=[display_output1],
                        y=[display_output2],
                        mode='markers',
                        marker=dict(
                            size=9,
                            color=colors[i % len(colors)],
                            opacity=0.4),
                        name=label,
                        hovertext=f'{label}: ({ghost_output1}, {ghost_output2})',
                        hoverinfo='text')

                    # Add lines to ghost points for visibility
                    fig.add_shape(type='line',
                                  x0=current_output1+offset_x, y0=current_output2+offset_y,
                                  x1=display_output1, y1=display_output2,
                                  line=dict(color=colors[i % len(colors)], width=2, dash='dot'))


        min_x = df['training_duration'].min()
        max_x = df['training_duration'].max()
        min_y = df['final_val_loss'].min()
        max_y = df['final_val_loss'].max()

        margin = 5
        fig.update_layout(
            xaxis_range=[min_x - margin, max_x + margin],
            yaxis_range=[min_y - margin, max_y + margin])
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
