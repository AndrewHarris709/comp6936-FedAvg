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

def get_initial_weights(columnIdx):
    inDim, outDim = get_dataset_shape(columnIdx)
    return [np.random.rand(outDim, inDim), np.random.rand(outDim)]

def get_dataset(columnIdx = -1):
    data_X, data_Y = datasets.load_diabetes(return_X_y = True)
    if(columnIdx >= 0):
        data_X = data_X[:, np.newaxis, columnIdx]
    return data_X, data_Y

def get_splitted_dataset(numSplits, columnIdx = -1):
    data_X, data_Y = get_dataset(columnIdx)
    return np.array_split(data_X, numSplits), np.array_split(data_Y, numSplits)

def get_dataset_shape(columnIdx = -1):
    data_X, data_Y = get_dataset(columnIdx)
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