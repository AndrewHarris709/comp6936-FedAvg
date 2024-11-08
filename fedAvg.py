from Client import Client
from Server import Server
from linearRegression.utils import get_keras_initial_weights, get_splitted_dataset

data_X, data_Y = get_splitted_dataset(numSplits = 5, columnIdx = 3)

clientAli = Client("Ali")
clientAndrew = Client("Andrew")
clientMahdi = Client("Mahdi")
clientJames = Client("James")
clientVictoria = Client("Victoria")

clientAli.add_data(new_X = data_X[0], new_Y = data_Y[0])
clientAndrew.add_data(new_X = data_X[1], new_Y = data_Y[1])
clientMahdi.add_data(new_X = data_X[2], new_Y = data_Y[2])
clientJames.add_data(new_X = data_X[3], new_Y = data_Y[3])
clientVictoria.add_data(new_X = data_X[4], new_Y = data_Y[4])

server = Server(
    participationRatio = 0.8,
    initialWeights = get_keras_initial_weights(columnIdx = 3),
    clients = [clientAli, clientAndrew, clientMahdi, clientJames, clientVictoria]
)

server.start_clients()
server.test_model()