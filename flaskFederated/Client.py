import numpy as np
from linearRegression.models import get_model
from linearRegression.utils import fit_model, get_loss, get_params

class Client():
    def __init__(self, name):
        self.name = name
        self.data_X = []
        self.data_Y = []
        self.model = None

    def add_data(self, new_X, new_Y):
        if(new_X.shape[0] == 1):
            self.data_X.append(new_X)
            self.data_Y.append(new_Y)
        else:
            self.data_X.extend(new_X)
            self.data_Y.extend(new_Y)            

    def train(self, weights: np.ndarray):
        if(len(self.data_X) == 0 or len(self.data_Y) == 0):
            print("Waiting for data...")
            return None

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

        return report

    def get_model(self):
        return self.model
