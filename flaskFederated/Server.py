import concurrent.futures
import numpy as np
from linearRegression.models import get_model
from linearRegression.utils import get_dataset_shape, get_params
from flask import Flask, request, jsonify
import requests

class Server:
    def __init__(self, mode, participationRatio, initialWeights, clients):
        self.mode = mode
        self.C = participationRatio
        self.clients = clients
        self.weights = initialWeights
        self.m = 0
        self.model = None
        self.clientWs = []
        self.app = Flask(__name__)
        self.app.add_url_rule("/send_weights", view_func = self.send_weights, methods = ["POST"])
        self.app.add_url_rule("/update_weights", view_func = self.update_weights, methods = ["POST"])

    def start_clients(self):
        self.m = int(max(round(self.C * len(self.clients)), 1))
        selectedClients = np.random.choice(self.clients, self.m, replace = False)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.send_weights, selectedClients)
        
    def update_weights(self):
        if(len(self.clientWs) == self.m):
            Nr = sum(res["numRecords"] for res in self.clientWs)
            newWeights = []
            for weights in self.clientWs[0]["weights"]:
                newWeights.append(np.zeros(shape = weights.shape))
            
            for res in self.clientWs:
                for idx in range(len(res["weights"])):
                    newWeights[idx] += (res["weights"][idx] * (res["numRecords"] / Nr))

            self.weights = newWeights
            self.clientWs = []
        else:
            self.clientWs.append(request.json)

    def update_centralized_model(self, columnIdx):
        if(not self.model):
            inputDim, outputDim = get_dataset_shape(columnIdx = columnIdx)
            self.model = get_model(mode = self.mode, inputDim = inputDim, outputDim = outputDim)
        
        if(self.mode == "keras"):
            self.model.set_weights(self.weights)
        else:
            self.model.coef_ = self.weights[0]
            self.model.intercept_ = self.weights[1]        

    def test_model(self, columnIdx = -1):
        self.update_centralized_model(columnIdx)
        w, b = get_params(mode = self.mode, model = self.model)
        print(f"Server Weight: {w}")
        print(f"Server Bias: {b}")
        return w, b

    def get_clients(self):
        return self.clients
    
    def send_weights(self, client):
        requests.post(f"{client.ip}/new_weights", json = {"new_weights": self.weights})