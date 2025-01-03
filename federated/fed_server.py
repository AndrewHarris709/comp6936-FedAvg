from linearRegression.models import get_model
import numpy as np

class FederatedServer:
    def __init__(self, participationRatio, initialWeights, max_iter: int = 1):
        self.C = participationRatio
        self.clients = []
        self.m = 0
        self.model = None
        self.clientWs = []
        self.clientWRecord = {}
        self.initialWeights = initialWeights
        self.max_iter = max_iter
        self.update_centralized_model(initialWeights)

    def select_clients(self):
        if(len(self.clients)):
            self.m = int(max(round(self.C * len(self.clients)), 1))
            selectedClients = np.random.choice(self.clients, self.m, replace = False)
            return selectedClients
        return False
        
    def update_weights(self, res, client):
        self.clientWs.append(res)
        self.clientWRecord[client] = res['weights']

        if len(self.clientWs) == self.m:
            Nr = sum(res["numRecords"] for res in self.clientWs)
            newWeights = []
            for w in self.clientWs[0]["weights"]:
                newWeights.append(np.zeros(shape = w.shape))
            
            for res in self.clientWs:
                for idx in range(len(res["weights"])):
                    newWeights[idx] += (res["weights"][idx] * (res["numRecords"] / Nr))

            self.update_centralized_model(newWeights)
            self.clientWs = []

    def update_centralized_model(self, weights):
        if(not self.model):
            self.model = get_model(max_iter=self.max_iter)
        
        self.model.coef_ = weights[0]
        self.model.intercept_ = weights[1]
    
    def get_weights(self):
        return [self.model.coef_, self.model.intercept_]

    def get_client_weights(self):
        return self.clientWRecord

    def add_client(self, id):
        self.clients.append(id)

    def remove_client(self, id):
        self.clients.remove(id)

    def reset(self):
        self.m = 0
        self.clientWs = []
        self.clientWRecord = {}
        self.update_centralized_model(self.initialWeights)
