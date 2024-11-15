from linearRegression.models import get_model
from linearRegression.utils import get_dataset_shape, get_params, get_weights_jsonified, get_weights_dejsonified
from flask import Flask, request
import numpy as np
import requests

class Server:
    def __init__(self, mode, ip, participationRatio, initialWeights):
        self.mode = mode
        self.C = participationRatio
        self.clients = []
        self.weights = initialWeights
        self.m = 0
        self.model = None
        self.clientWs = []
        self.app = Flask(__name__)
        self.app.add_url_rule("/new_client", view_func = self.new_client, methods = ["POST"])
        self.app.add_url_rule("/update_weights", view_func = self.update_weights, methods = ["POST"])
        self.app.add_url_rule("/", view_func = self.start_clients, methods = ["GET"])
        self.app.add_url_rule("/test", view_func = self.test_model, methods = ["GET"])
        self.ip = ip

    def start_clients(self):
        if(len(self.clients)):
            self.m = int(max(round(self.C * len(self.clients)), 1))
            selectedClients = np.random.choice(self.clients, self.m, replace = False)
            for clientIP in selectedClients:
                requests.post(f"http://{clientIP}/train", json = {"new_weights": get_weights_jsonified(self.weights)})
            return "Trained Clients!\n"
        return "No Clients Yet!\n"
        
    def update_weights(self):
        res = request.json
        res["weights"] = get_weights_dejsonified(res["weights"])
        self.clientWs.append(res)

        if(len(self.clientWs) == self.m):
            Nr = sum(res["numRecords"] for res in self.clientWs)
            newWeights = []
            for w in self.clientWs[0]["weights"]:
                newWeights.append(np.zeros(shape = w.shape))
            
            for res in self.clientWs:
                for idx in range(len(res["weights"])):
                    newWeights[idx] += (res["weights"][idx] * (res["numRecords"] / Nr))

            self.weights = newWeights
            self.clientWs = []

        return "Update Weights Called!\n"

    def update_centralized_model(self, columnIdx):
        if(not self.model):
            inputDim, outputDim = get_dataset_shape(columnIdx = columnIdx)
            self.model = get_model(mode = self.mode, inputDim = inputDim, outputDim = outputDim)
        
        if(self.mode == "keras"):
            self.model.set_weights(self.weights)
        else:
            self.model.coef_ = self.weights[0]
            self.model.intercept_ = self.weights[1]        

    def test_model(self):
        # Have to send a JSON with columnIdx as a key to get only the desired column!
        columnIdx = request.json["columnIdx"] if request.is_json else -1
        self.update_centralized_model(columnIdx)
        w, b = get_params(mode = self.mode, model = self.model)
        print(f"Server Weight: {w}")
        print(f"Server Bias: {b}")
        return f"{w}, {b}\n"

    def get_clients(self):
        return self.clients

    def new_client(self):
        self.clients.append(request.json["ip"])
        return "New Client!\n"

    def start_flask(self):
        ip, port = self.ip.split(":")
        self.app.run(host = ip, port = int(port))