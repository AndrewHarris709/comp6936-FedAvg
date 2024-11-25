import dash
import dash_bootstrap_components as dbc
from dash import dcc, Output, Input, no_update, ctx, State
import plotly.express as px
import plotly.graph_objects as go

from linearRegression.utils import get_weights_dejsonified

from sklearn.linear_model import LinearRegression, SGDRegressor
import requests
import numpy as np
import pandas as pd
from generators import CholeskyGenerator

corr = np.array([[1, 0.9, 0.8], [0.9, 1, 0.92], [0.8, 0.92, 1]])
test_data = CholeskyGenerator(corr).get(10000)

score_results = pd.DataFrame([], columns=['All Data', 'Federated'])
client_results = pd.DataFrame([], columns=[])

server_ip = "http://fed-server:5000"

app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.layout = dbc.Card([
    dbc.CardHeader("Federated Learning Evaluation"),
    dbc.CardBody([
        dcc.Interval(id='graph-timer', interval=100),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='main-graph', style={'width': '100%', 'height': '90vh'}),
            ], width=True),
            dbc.Col([
                dbc.Button("Reset", id="reset-button", color="danger"),
                dbc.Button("Save", id="save-button", color="success"),
                dbc.Checklist(
                    options=[
                        {"label": "Show Clients", "value": 1}
                    ],
                    value=[],
                    id="checklist",
                ),
                dcc.Download(id="download-dataframe"),
            ], width="auto")
        ])
    ])
])

@app.callback(
    Output("main-graph", "figure"),
    Input("graph-timer", "n_intervals"),
    Input("reset-button", "n_clicks"),
    State("checklist", "value"),
)
def update(n_intervals, n_clicks, check_value):
    if ctx.triggered_id == 'reset-button':
        if n_clicks is not None:
            requests.get(f"{server_ip}/reset")
            global score_results
            score_results = pd.DataFrame([], columns=['All Data', 'Federated'])
            global client_results
            client_results = pd.DataFrame([])
            return {}
        return no_update

    if n_intervals % 3 == 0:
        return graph_update(1 in check_value)
    elif n_intervals % 3 == 1:
        requests.get(f"{server_ip}/generate")
    else:
        requests.get(server_ip)
    return no_update

@app.callback(
    Output("download-dataframe", "data"),
    Input("save-button", "n_clicks"),
    prevent_initial_call=True
)
def download_data(n_clicks):
    return dcc.send_data_frame(score_results.to_csv, "fed_trial_data.csv")

def graph_update(list_clients):
    all_data_result = requests.get(f"{server_ip}/data/all").json()
    fed_model_params = get_weights_dejsonified(requests.get(f"{server_ip}/model").json())
    client_model_params = requests.get(f"{server_ip}/model/clients").json()
    client_model_params = {client: get_weights_dejsonified(weights) for client, weights in client_model_params.items()}

    all_X, all_Y = [], []
    for name, client_data in all_data_result.items():
        all_X.extend(client_data['X'])
        all_Y.extend(client_data['Y'])

    all_X, all_Y = np.array(all_X), np.array(all_Y)

    # Train a model on all available data
    all_model = LinearRegression().fit(all_X, all_Y)
    all_score = all_model.score(test_data[:-1].T, test_data[-1])

    # Recreate the server model from given parameters
    fed_model = SGDRegressor()
    fed_model.coef_ = fed_model_params[0]
    fed_model.intercept_ = fed_model_params[1]
    fed_score = fed_model.score(test_data[:-1].T, test_data[-1])

    # Test each client model against the test dataset
    client_row = {}
    for client, params in client_model_params.items():
        client_model = SGDRegressor()
        client_model.coef_ = params[0]
        client_model.intercept_ = params[1]
        client_score = client_model.score(test_data[:-1].T, test_data[-1])
        client_row[client] = client_score

    score_results.loc[len(score_results)] = [all_score, fed_score]

    global client_results
    client_results = pd.concat([client_results, pd.DataFrame([client_row])], ignore_index=True)

    # Create a plot of all_scores & fed_scores over iterations
    if list_clients:
        fig = px.scatter(client_results, x=client_results.index, y=client_results.columns)
        fig.add_trace(go.Scatter(x=score_results.index, y=score_results['All Data'], mode='markers', name='All Data'))
    else:
        fig = px.scatter(data_frame=score_results, x=score_results.index, y=['All Data', 'Federated'])

    fig.update_yaxes(range=[0, 1])
    return fig

if __name__ == "__main__":
    app.run_server(port=8000, host="0.0.0.0")
