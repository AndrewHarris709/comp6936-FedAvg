from linearRegression.models import get_model
from linearRegression.utils import get_params
import numpy as np

class Server:
    def __init__(self, participationRatio, initialWeights):
        self.C = participationRatio
        self.clients = []
        self.weights = initialWeights
        self.m = 0
        self.model = None
        self.clientWs = []

    def start_clients(self):
        if(len(self.clients)):
            self.m = int(max(round(self.C * len(self.clients)), 1))
            selectedClients = np.random.choice(self.clients, self.m, replace = False)
            return selectedClients
        return False
        
    def update_weights(self, res):
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

    def update_centralized_model(self):
        if(not self.model):
            self.model = get_model()
        
        self.model.coef_ = self.weights[0]
        self.model.intercept_ = self.weights[1]        

    def test_model(self):
        self.update_centralized_model()
        w, b = get_params(model = self.model)
        print(f"Server Weight: {w}")
        print(f"Server Bias: {b}")
        return f"{w}, {b}\n"
    
    def get_weights(self):
        return self.weights

    def add_client(self, ip):
        self.clients.append(ip)
