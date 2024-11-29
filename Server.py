#!/usr/bin/env python
import os
print(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request
from flask_socketio import SocketIO, emit
from federated import FederatedServer
from linearRegression.utils import get_initial_weights, get_code_params, get_weights_jsonified, get_weights_dejsonified
import sys
import numpy as np


if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

server = FederatedServer(
    participationRatio = codeParams["participation_ratio"],
    initialWeights = [np.random.rand(2), np.array([np.random.rand()])]
)

app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/", methods=['GET'])
def start_clients():
    selections = server.select_clients()
    if selections is False:
        return "No Clients Yet!\n"
    
    weights = get_weights_jsonified(server.get_weights())
    for client in selections:
        emit('client_update', weights, to=client, namespace="/")

    return "Training Requests Sent!\n"

@app.route("/reset", methods=['GET'])
def test_model():
    global global_data
    global_data = {}
    server.reset()
    emit("reset", namespace="/", broadcast=True)
    return "System has been reset."


global_data = {}

@app.route("/data/all", methods=['GET'])
def get_all_data():
    return global_data

@app.route("/model", methods=['GET'])
def get_model():
    return get_weights_jsonified(server.get_weights())

@app.route('/model/clients', methods=['GET'])
def get_client_models():
    clients_weights = server.get_client_weights()
    print(clients_weights, flush=True)
    return {client: get_weights_jsonified(weights) for client, weights in clients_weights.items()}

@app.route("/generate", methods=['GET'])
def generate_data():
    emit('generate_data', namespace="/", broadcast=True)
    return "Generation Request Sent"

@socketio.event
def data_update(data):
    global_data[request.sid] = data

@socketio.event
def training_complete(report):
    report['weights'] = get_weights_dejsonified(report['weights'])
    server.update_weights(report, request.sid)
    print(f'Weights updated!')

@socketio.event
def connect(auth):
    server.add_client(request.sid)
    print('Client connected')

@socketio.event
def disconnect():
    server.remove_client(request.sid)
    print('Client disconnected')

if __name__ == "__main__":
    port = codeParams['port']
    socketio.run(app, host = '0.0.0.0', port = port, allow_unsafe_werkzeug=True)
