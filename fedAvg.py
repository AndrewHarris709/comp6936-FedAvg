#!/usr/bin/env python

from Client import Client
from Server import Server
from linearRegression.utils import get_initial_weights, get_splitted_dataset, get_code_params
import sys


if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

data_X, data_Y = get_splitted_dataset(numSplits = 5, columnIdx = codeParams["columnIdx"])

clientAli = Client(name = codeParams["clients"]["names"][0], mode = codeParams["mode"])
clientAndrew = Client(name = codeParams["clients"]["names"][1], mode = codeParams["mode"])
clientMahdi = Client(name = codeParams["clients"]["names"][2], mode = codeParams["mode"])
clientJames = Client(name = codeParams["clients"]["names"][3], mode = codeParams["mode"])
clientVictoria = Client(name = codeParams["clients"]["names"][4], mode = codeParams["mode"])

clientAli.add_data(new_X = data_X[0], new_Y = data_Y[0])
clientAndrew.add_data(new_X = data_X[1], new_Y = data_Y[1])
clientMahdi.add_data(new_X = data_X[2], new_Y = data_Y[2])
clientJames.add_data(new_X = data_X[3], new_Y = data_Y[3])
clientVictoria.add_data(new_X = data_X[4], new_Y = data_Y[4])

server = Server(
    mode = codeParams["mode"],
    participationRatio =codeParams["server"]["participation_ratio"],
    initialWeights = get_initial_weights(codeParams["mode"], codeParams["columnIdx"]),
    clients = [clientAli, clientAndrew, clientMahdi, clientJames, clientVictoria]
)

server.start_clients()
server.test_model(columnIdx = codeParams["columnIdx"])