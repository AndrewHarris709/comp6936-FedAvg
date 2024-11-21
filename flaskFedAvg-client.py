#!/usr/bin/env python

from flaskFederated.Client import Client
from linearRegression.utils import get_splitted_dataset, get_code_params, get_weights_dejsonified, get_weights_jsonified
import sys
from numpy.random import randint
import socketio
import time

if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

numSplits = 5
data_X, data_Y = get_splitted_dataset(numSplits = numSplits)

client = Client(name = codeParams["name"])

rand_idx = randint(numSplits)
client.add_data(new_X = data_X[rand_idx], new_Y = data_Y[rand_idx])

sio = socketio.Client()

@sio.event
def client_update(new_weights):
    weights = get_weights_dejsonified(new_weights)

    report = client.train(weights)
    if not report:
        return None

    report["weights"] = get_weights_jsonified(report["weights"])
    sio.emit('training_complete', report)

if __name__ == '__main__':
    sio.connect(f"http://{codeParams['server_ip']}")

    while sio.connected:
        time.sleep(2)
        sio.emit('data_update', {'X': client.data_X.tolist(), 'Y': client.data_Y.tolist()})
