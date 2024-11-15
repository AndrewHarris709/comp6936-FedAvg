#!/usr/bin/env python

from flaskFederated.Client import Client
from linearRegression.utils import get_splitted_dataset, get_code_params
import sys
from numpy.random import randint


if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

numSplits = 5
data_X, data_Y = get_splitted_dataset(numSplits = numSplits, columnIdx = codeParams["columnIdx"])

client = Client(
    name = codeParams["name"],
    mode = codeParams["mode"],
    ip = codeParams["ip"],
    server_ip = codeParams["server_ip"]
)

rand_idx = randint(numSplits)
client.add_data(new_X = data_X[rand_idx], new_Y = data_Y[rand_idx])

client.start_flask()
