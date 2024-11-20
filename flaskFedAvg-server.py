#!/usr/bin/env python

from flaskFederated.Server import Server
from linearRegression.utils import get_initial_weights, get_code_params
import sys

'''
Training Clients: curl http://server_address/
Getting Centeralizaed Model Params: curl http://server_address/test
'''


if(not (len(sys.argv) > 1 and len(sys.argv) < 4)):
    print("Wrong format! Exiting...")
    sys.exit(0)
if(sys.argv[1] == "-p"):
    jsonPath = sys.argv[2]
else:
    print("Wrong format! Exiting...")
    sys.exit(0)

codeParams = get_code_params(jsonPath)

server = Server(
    ip = codeParams["ip"],
    participationRatio = codeParams["participation_ratio"],
    initialWeights = get_initial_weights(codeParams["columnIdx"])
)

server.start_flask()