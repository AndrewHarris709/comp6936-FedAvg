import numpy as np
from linearRegression.models import get_model
from linearRegression.utils import fit_model, get_loss, get_params
from flask import Flask

class Client():
    def __init__(self, name, mode, ip):
        self.name = name
        self.mode = mode
        self.data_X = []
        self.data_Y = []
        self.model = None
        self.app = Flask(__name__)
        self.ip = ip

    def add_data(self, new_X, new_Y):
        if(new_X.shape[0] == 1):
            self.data_X.append(new_X)
            self.data_Y.append(new_Y)
        else:
            self.data_X.extend(new_X)
            self.data_Y.extend(new_Y)            

    def train(self, weights):
        if(len(self.data_X) == 0 or len(self.data_Y) == 0):
            print("Waiting for data...")
            return None

        print(f"Start Training {self.name}...")
        report = {}
        X = np.array(self.data_X)
        Y = np.array(self.data_Y)
        report["numRecords"] = X.shape[0]

        if(not self.model):
            self.model = get_model(mode = self.mode, inputDim = X.shape[1], outputDim = 1 if len(Y.shape) == 1 else Y.shape[1])

        if(self.mode == "keras"):
            self.model.set_weights(weights)
            fit_model(mode = self.mode, model = self.model, X = X, Y = Y)
            report["weights"] = self.model.get_weights()
        else:
            self.model = fit_model(mode = self.mode, model = self.model, X = X, Y = Y, weights = weights[0], biases = weights[1])
            report["weights"] = get_params(mode = self.mode, model = self.model)

        print(f"{self.mode} Loss {self.name}: {get_loss(mode = self.mode, model = self.model, X = X, Y = Y, pred_Y = self.model.predict(X), target_Y = Y)}")
        weight, bias = get_params(mode = self.mode, model = self.model)
        print(f"Weight {self.name}: {weight}")
        print(f"Bias {self.name}: {bias}")

        return report

    def get_model(self):
        return self.model