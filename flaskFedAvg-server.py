#!/usr/bin/env python
from flask import Flask, request
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

@app.route("/new_client", methods=['POST'])
def new_client():
    server.add_client(request.json["ip"])
    return "New Client!\n"

@app.route("/update_weights", methods=['POST'])
def update_weights():
    report = request.json
    report['weights'] = get_weights_dejsonified(request.json['weights'])
    server.update_weights(report)
    return "Update Weights Called!\n"

@app.route("/", methods=['GET'])
def start_clients():
    selections = server.start_clients()
    if selections is False:
        return "No Clients Yet!\n"
    
    weights = get_weights_jsonified(server.get_weights())
    for clientIP in selections:
        requests.post(f"http://{clientIP}/train", json = {"new_weights": weights})

    return "Trained Clients!\n"

@app.route("/test", methods=['GET'])
def collect_model():
    return server.test_model()

if __name__ == "__main__":
    ip, port = codeParams['ip'].split(":")
    app.run(host = ip, port = int(port))
