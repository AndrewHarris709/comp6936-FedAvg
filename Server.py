import concurrent.futures
import numpy as np
from linearRegression.models import get_model
from linearRegression.utils import get_dataset_shape, get_params
from copy import deepcopy

class Server:
    def __init__(self, mode, participationRatio, initialWeights, clients):
        self.mode = mode
        self.C = participationRatio
        self.clients = clients
        self.weights = initialWeights
        self.model = None

    def start_clients(self):
        clientsRes = []
        m = int(max(round(self.C * len(self.clients)), 1))
        selectedClients = np.random.choice(self.clients, m, replace = False)

        # Deepcopy must be removed when moving to multi-device!!!
        # Deepcopy must only be used when the code is running in parallel on a single device simulating a multi-device senario
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outcomes = [executor.submit(lambda c: c.train(deepcopy(self.weights)), client) for client in selectedClients]
            for outcome in concurrent.futures.as_completed(outcomes):
                clientsRes.append(outcome.result())
        
        self.update_weights(clientsRes)

    def update_weights(self, clientsResults):
        Nr = sum(res["numRecords"] for res in clientsResults)
        newWeights = []
        for weights in clientsResults[0]["weights"]:
            newWeights.append(np.zeros(shape = weights.shape))
        
        for res in clientsResults:
            for idx in range(len(res["weights"])):
                newWeights[idx] += (res["weights"][idx] * (res["numRecords"] / Nr))

        self.weights = newWeights

    def test_model(self, columnIdx = -1):
        if(not self.model):
            inputDim, outputDim = get_dataset_shape(columnIdx = columnIdx)
            self.model = get_model(mode = self.mode, inputDim = inputDim, outputDim = outputDim)
        
        if(self.mode == "keras"):
            self.model.set_weights(self.weights)
        else:
            self.model.coef_ = self.weights[0]
            self.model.intercept_ = self.weights[1]
        w, b = get_params(mode = self.mode, model = self.model)
        print(f"Server Weight: {w}")
        print(f"Server Bias: {b}")
        return w, b

    def get_clients(self):
        return self.clients
