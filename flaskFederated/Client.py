import numpy as np
from linearRegression.models import get_model
from linearRegression.utils import fit_model, get_loss, get_params
from generators import CholeskyGenerator

corr = np.array([[1, 0.9, 0.8], [0.9, 1, 0.92], [0.8, 0.92, 1]])

class Client():
    def __init__(self, name):
        self.name = name
        self.model = None
        self.generator = CholeskyGenerator(corr)

        initial = self.generator.get(2)
        self.data_X = initial[:-1].T
        self.data_Y = initial[-1]

    def add_data(self):
        new_data = self.generator.get(1)
        self.data_X = np.append(self.data_X, new_data[:-1].T, axis=0)
        self.data_Y = np.append(self.data_Y, new_data[-1])

    def train(self, weights: np.ndarray):
        if(len(self.data_X) == 0 or len(self.data_Y) == 0):
            print("Waiting for data...")
            return None

        print(f"Start Training {self.name}...")
        report = {}
        X = self.data_X
        Y = self.data_Y
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
