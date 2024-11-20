#!/usr/bin/env python

from flaskFederated.Client import Client
from linearRegression.utils import get_splitted_dataset, get_code_params, get_weights_dejsonified, get_weights_jsonified
from flask import Flask, request
import sys
from numpy.random import randint
import requests

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
data_X, data_Y = get_splitted_dataset(numSplits = numSplits, columnIdx = codeParams["columnIdx"])

client = Client(name = codeParams["name"])

rand_idx = randint(numSplits)
client.add_data(new_X = data_X[rand_idx], new_Y = data_Y[rand_idx])

app = Flask(__name__)

@app.route("/train", methods=["POST"])
def train_model():
    if(not request.json):
        return None

    weights = get_weights_dejsonified(request.json["new_weights"])

    report = client.train(weights)
    if not report:
        return None
    
    report["weights"] = get_weights_jsonified(report["weights"])
    requests.post(f"http://{codeParams['server_ip']}/update_weights", json = report)
    return report

if __name__ == '__main__':
    ip, port = codeParams["ip"].split(":")
    requests.post(f"http://{codeParams['server_ip']}/new_client", json = {"ip": codeParams["ip"]})
    app.run(host = ip, port = int(port))

