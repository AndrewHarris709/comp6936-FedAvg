import concurrent.futures
import numpy as np
from linearRegression.models import get_keras_model
from linearRegression.utils import get_dataset_shape, get_keras_params, get_keras_loss

class Server:
    def __init__(self, participationRatio, initialWeights, clients):
        self.C = participationRatio
        self.clients = clients
        self.weights = initialWeights
        self.model = None

    def start_clients(self):
        clientsRes = []
        m = int(max(self.C * len(self.clients), 1))
        selectedClients = np.random.choice(self.clients, m, replace = False)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            outcomes = [executor.submit(lambda c: c.train(self.weights), client) for client in selectedClients]
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

    def test_model(self):
        if(not self.model):
            inputDim, outputDim = get_dataset_shape(columnIdx = 3)
            self.model = get_keras_model(inputDim = inputDim, outputDim = outputDim)
        self.model.set_weights(self.weights)
        w, b = get_keras_params(self.model)
        print(f"Server Weight: {w}")
        print(f"Server Bias: {b}")        
