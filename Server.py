import concurrent.futures
import numpy as np

class Server:
    def __init__(self, participationRatio, initialWeights, clients):
        self.C = participationRatio
        self.clients = clients
        self.weights = initialWeights

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
        newWeights = np.zeros(shape = clientsResults[0]["weights"].shape)
        Nr = sum(res["numRecords"] for res in clientsResults)
        for res in clientsResults:
            newWeights += res["weights"] * (res["numRecords"] / Nr)
        self.weights = newWeights