from sklearn.metrics import mean_squared_error
from sklearn import datasets
import numpy as np
import json

def fit_model(model, X, Y, weights, biases):
    return model.fit(X, Y, coef_init = weights, intercept_init = biases)

def get_loss(pred_Y, target_Y):
    return mean_squared_error(pred_Y, target_Y)

def get_params(model):
    return model.coef_, model.intercept_

def get_initial_weights():
    inDim, outDim = get_dataset_shape()
    return [np.random.rand(outDim, inDim), np.random.rand(outDim)]

def get_dataset():
    data_X, data_Y = datasets.load_diabetes(return_X_y = True)
    return data_X, data_Y

def get_splitted_dataset(numSplits):
    data_X, data_Y = get_dataset()
    return np.array_split(data_X, numSplits), np.array_split(data_Y, numSplits)

def get_dataset_shape():
    data_X, data_Y = get_dataset()
    if(len(data_Y.shape) == 1):
        outDim = 1
    else:
        outDim = data_Y.shape[1]
    return data_X.shape[1], outDim

def get_code_params(path):
    with open(path, mode = "r", encoding = "utf-8") as f:
        data = json.load(f)
    return data

def get_weights_jsonified(list_of_arrays):
    return [arr.tolist() for arr in list_of_arrays]

def get_weights_dejsonified(list_of_lists):
    return [np.array(l) for l in list_of_lists]