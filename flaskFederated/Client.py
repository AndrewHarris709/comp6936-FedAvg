import numpy as np
from linearRegression.models import get_model
from linearRegression.utils import fit_model, get_loss, get_params, get_weights_dejsonified, get_weights_jsonified
from flask import Flask, request
import requests

class Client():
    def __init__(self, name, ip, server_ip):
        self.name = name
        self.data_X = []
        self.data_Y = []
        self.model = None
        self.app = Flask(__name__)
        self.app.add_url_rule("/train", view_func = self.train, methods = ["POST"])
        self.ip = ip
        self.server_ip = server_ip

    def add_data(self, new_X, new_Y):
        if(new_X.shape[0] == 1):
            self.data_X.append(new_X)
            self.data_Y.append(new_Y)
        else:
            self.data_X.extend(new_X)
            self.data_Y.extend(new_Y)            

    def train(self):
        if(len(self.data_X) == 0 or len(self.data_Y) == 0):
            print("Waiting for data...")
            return None
        
        if(not request.json):
            return None

        weights = get_weights_dejsonified(request.json["new_weights"])

        print(f"Start Training {self.name}...")
        report = {}
        X = np.array(self.data_X)
        Y = np.array(self.data_Y)
        report["numRecords"] = X.shape[0]

        if(not self.model):
            self.model = get_model()

        self.model = fit_model(model = self.model, X = X, Y = Y, weights = weights[0], biases = weights[1])
        report["weights"] = get_params(model = self.model)

        print(f"Loss {self.name}: {get_loss(pred_Y = self.model.predict(X), target_Y = Y)}")
        w, b = get_params(model = self.model)
        print(f"Weight {self.name}: {w}")
        print(f"Bias {self.name}: {b}")

        report["weights"] = get_weights_jsonified(report["weights"])
        requests.post(f"http://{self.server_ip}/update_weights", json = report)
        return report

    def get_model(self):
        return self.model
    
    def start_flask(self):
        ip, port = self.ip.split(":")
        requests.post(f"http://{self.server_ip}/new_client", json = {"ip": self.ip})
        self.app.run(host = ip, port = int(port))
