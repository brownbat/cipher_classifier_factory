import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import random


# rolled back to working version with tuples

def generate_colors(num_colors, saturation=40, lightness=40):
    """Generate 'num_colors' distinct pastel colors in HSL format."""
    for i in range(num_colors):
        hue = int((360 / num_colors) * i)  # Evenly space the hue
        yield f"hsl({hue}, {saturation}%, {lightness}%)"  # Adjusted Saturation and Lightness for pastel effect


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


def setup_dash_app(data):
    df = pd.DataFrame(data).T  # convert to dataframe and transpose
    df.columns = ['output1', 'output2']
    num_params = len(list(data.keys())[0])
    slider_css = generate_slider_css(num_params)
    write_css_to_file(slider_css)

    app = dash.Dash(__name__)

    html_output = [dcc.Graph(id='output-graph')]

    for idx in range(num_params):
        html_output.append(html.Div([
            html.Label(f'Param{idx+1}'),
            dcc.Slider(
                id=f'slider-param{idx+1}',
                min=0,
                max=10,
                step=5,
                value=5,
                marks={i: str(i) for i in range(0, 11, 5)},
                className=f'slider-color-{idx+1}'
            ),
        ], style={'padding': '20px 0px'}))

    app.layout = html.Div(html_output)

    @app.callback(
        Output('output-graph', 'figure'),
        [Input(f'slider-param{i + 1}', 'value') for i in range(num_params)]
    )
    def update_graph(*params):
        colors = list(generate_colors(5))
        current_key = tuple(params)
        current_output1, current_output2 = data[current_key]
        fig = px.scatter(x=[current_output1], y=[current_output2], 
                         labels={'x':'Output 1', 'y':'Output 2'})
        fig.update_traces(marker=dict(size=15, color='black'))

        offsets = generate_offsets(num_params)

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


        min_x = df['output1'].min()
        max_x = df['output1'].max()
        min_y = df['output2'].min()
        max_y = df['output2'].max()

        margin = 5
        fig.update_layout(
            xaxis_range=[min_x - margin, max_x + margin],
            yaxis_range=[min_y - margin, max_y + margin])
        return fig
    return app


# keyed by param1, param2, param3 -> output_x, output_y
# Adjusted data dictionary with a fourth parameter
sample_data = {
    (5, 5, 5, 5): (5, 90),
    (5, 5, 5, 10): (10, 85),
    (5, 5, 10, 5): (10, 80),
    (5, 5, 10, 10): (15, 75),
    (5, 10, 5, 5): (15, 85),
    (5, 10, 5, 10): (20, 80),
    (5, 10, 10, 5): (20, 75),
    (5, 10, 10, 10): (25, 70),
    (10, 5, 5, 5): (20, 85),
    (10, 5, 5, 10): (25, 80),
    (10, 5, 10, 5): (25, 75),
    (10, 5, 10, 10): (30, 70),
    (10, 10, 5, 5): (30, 80),
    (10, 10, 5, 10): (35, 75),
    (10, 10, 10, 5): (35, 70),
    (10, 10, 10, 10): (40, 65)
}


def run_app(data=sample_data):
    colors = list(generate_colors(5))
    num_params = len(list(data.keys())[0])
    app = setup_dash_app(data)
    app.run_server(debug=True)


if __name__ == '__main__':
    run_app()
