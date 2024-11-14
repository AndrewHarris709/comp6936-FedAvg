import numpy as np
from linearRegression.models import get_keras_model
from linearRegression.utils import fit_keras_model, get_keras_loss, get_keras_params

class Client():
    def __init__(self, name, mode):
        self.name = name
        self.data_X = []
        self.data_Y = []
        self.mode = mode
        self.model = None

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
        X = np.array(self.data_X)
        Y = np.array(self.data_Y)
        if(len(Y.shape) == 1):
            outDim = 1
        else:
            outDim = Y.shape[1]

        if(not self.model):
            self.model = get_keras_model(inputDim = X.shape[1], outputDim = outDim)
        self.model.set_weights(weights)
        fit_keras_model(self.model, X, Y)
        print(f"Loss {self.name}: {get_keras_loss(self.model, X, Y)}")
        weight, bias = get_keras_params(self.model)
        print(f"Weight {self.name}: {weight}")
        print(f"Bias {self.name}: {bias}")

        return {"weights": self.model.get_weights(), "numRecords": X.shape[0]}

    def get_model(self):
        return self.model