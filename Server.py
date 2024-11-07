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
            futures = [executor.submit(lambda c: c.train(self.weights), client) for client in selectedClients]
            for future in concurrent.futures.as_completed(futures):
                clientsRes.append(future.result())
        self.update_weights(clientsRes)

    def update_weights(self, clientsResults):
        print("-------------")
        print(clientsResults)