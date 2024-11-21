#!/usr/bin/env python
from flask import Flask, request
from flask_socketio import SocketIO, emit
from flaskFederated.Server import Server
from linearRegression.utils import get_initial_weights, get_code_params, get_weights_jsonified, get_weights_dejsonified
import sys
import requests

'''
Training Clients: curl http://server_address/
Getting Centralized Model Params: curl http://server_address/test
'''

if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

server = Server(
    participationRatio = codeParams["participation_ratio"],
    initialWeights = get_initial_weights()
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

@app.route("/test", methods=['GET'])
def collect_model():
    return server.test_model()

@socketio.event
def training_complete(report):
    report['weights'] = get_weights_dejsonified(report['weights'])
    server.update_weights(report)
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
    ip, port = codeParams['ip'].split(":")
    socketio.run(app, host = ip, port = int(port))
