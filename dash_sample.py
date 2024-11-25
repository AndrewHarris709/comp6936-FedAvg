import dash
import dash_bootstrap_components as dbc
from dash import Output, Input, State, dcc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from generators import CholeskyGenerator
from sklearn.linear_model import LinearRegression
import numpy as np

corr = np.array([[1, 0.1], [0.1, 1]])

plot_shape = (1, 1)

df = pd.DataFrame([], columns=['x0', 'y'])
generator = CholeskyGenerator(corr=corr, shifts=np.zeros(2), mults=[-3, 4])

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Card([
    dbc.CardHeader("Generated Data"),
    dbc.CardBody([
        dcc.Graph(id='main-graph', style={'width': '90vh', 'height': '90vh'}),
        dcc.Interval(id='graph-timer', interval=100)
    ])
])

def draw_slope_line_at_point(fig, y, slope, plot_index):
    x_vals = np.linspace(-15, 15, 100)
    y_vals = x_vals * float(slope) + y
    
    fig.add_trace(
            go.Scatter(
                x=x_vals, 
                y=y_vals,
            ), col=plot_index % plot_shape[0] + 1, row=plot_index // plot_shape[1] + 1
        )
    
    return x_vals, y_vals

@app.callback(
    Output("main-graph", "figure"),
    Input("graph-timer", "n_intervals"),
)
def graph_update(n):
    df.loc[len(df)] = generator.get().flatten()
    classifier = LinearRegression().fit(df.drop(columns='y').to_numpy(), df['y'].to_numpy())

    df2 = df.copy()
    df2 = pd.melt(df2, id_vars='y')

    fig = px.scatter(df2, x='value', y='y', facet_col='variable', facet_col_wrap=plot_shape[0])
    for i, coef in enumerate(classifier.coef_):
        draw_slope_line_at_point(fig, classifier.intercept_, float(coef), i)
    fig.layout.height = 1200
    fig.update_yaxes(matches=None)
    fig.update_xaxes(matches=None)

    return fig


if __name__ == "__main__":
    app.run_server()
