#!/usr/bin/env python

from flaskFederated.Client import Client
from linearRegression.utils import get_code_params, get_weights_dejsonified, get_weights_jsonified
import sys
import socketio

if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

client = Client(name = codeParams["name"], failure_rate = codeParams["failure_rate"])

sio = socketio.Client()

@sio.event
def client_update(new_weights):
    print("Training")
    weights = get_weights_dejsonified(new_weights)

    report = client.train(weights)
    if not report:
        return None

    report["weights"] = get_weights_jsonified(report["weights"])
    sio.emit('training_complete', report)

@sio.event
def reset():
    client.reset()

@sio.event
def generate_data():
    print("Adding Data")
    client.add_data()
    sio.emit('data_update', {'X': client.data_X.tolist(), 'Y': client.data_Y.tolist()})

if __name__ == '__main__':
    print("Starting")
    sio.connect(f"http://{codeParams['server_ip']}")
    sio.wait()
