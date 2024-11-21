import dash
import dash_bootstrap_components as dbc
from dash import dcc, Output, Input
import plotly.express as px

from linearRegression.utils import get_weights_dejsonified

from sklearn.linear_model import LinearRegression, SGDRegressor
import requests
import numpy as np
import pandas as pd
from generators import CholeskyGenerator

corr = np.array([[1, 0.9, 0.8], [0.9, 1, 0.92], [0.8, 0.92, 1]])
test_data = CholeskyGenerator(corr).get(10000)

score_results = pd.DataFrame([], columns=['All Data', 'Federated'])

server_ip = "http://localhost:5000"

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Card([
    dbc.CardHeader("Federated Learning Evaluation"),
    dbc.CardBody([
        dcc.Graph(id='main-graph', style={'width': '90vh', 'height': '90vh'}),
        dcc.Interval(id='graph-timer', interval=1000)
    ])
])

@app.callback(
    Output("main-graph", "figure"),
    Input("graph-timer", "n_intervals"),
)
def graph_update(_):
    all_data_result = requests.get(f"{server_ip}/data/all").json()
    fed_model_params = get_weights_dejsonified(requests.get(f"{server_ip}/model").json())

    all_X, all_Y = [], []
    for name, client_data in all_data_result.items():
        all_X.extend(client_data['X'])
        all_Y.extend(client_data['Y'])

    all_X, all_Y = np.array(all_X), np.array(all_Y)

    # Train a model on all available data
    all_model = LinearRegression().fit(all_X, all_Y)

    # Recreate the server model from given parameters
    fed_model = SGDRegressor()
    fed_model.coef_ = fed_model_params[0]
    fed_model.intercept_ = fed_model_params[1]

    all_score = all_model.score(test_data[:-1].T, test_data[-1])
    fed_model = fed_model.score(test_data[:-1].T, test_data[-1])

    score_results.loc[len(score_results)] = [all_score, fed_model]

    # Create a plot of all_scores & fed_scores over iterations
    fig = px.scatter(data_frame=score_results, x=score_results.index, y=['All Data', 'Federated'])
    fig.update_yaxes(range=[0, 1])
    return fig

if __name__ == "__main__":
    app.run_server()
