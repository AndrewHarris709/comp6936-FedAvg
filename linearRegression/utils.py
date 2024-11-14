from sklearn.metrics import mean_squared_error
from sklearn import datasets
from linearRegression.models import get_keras_model
import numpy as np
import json

def fit_keras_model(model, X, Y, batchSize, epochs):
    return model.fit(
        X,
        Y,
        batch_size = batchSize,
        epochs = epochs
    )

def fit_sklearn_model(model, X, Y):
    return model.fit(X, Y)

def fit_SGD_sklearn_model(model, X, Y, weights, biases):
    return model.fit(X, Y, coef_init = weights, intercept_init = biases)

def get_keras_loss(model, X, Y):
    return model.evaluate(X, Y)

def get_sklearn_loss(pred_Y, target_Y):
    return mean_squared_error(pred_Y, target_Y)

def get_keras_params(model):
    w1, b1 = model.layers[1].get_weights()
    w2, b2 = model.layers[2].get_weights()
    wTotal = np.multiply(w1, w2)
    bTotal = w2 * b1 + b2
    return [wTotal, bTotal]

def get_sklearn_params(model):
    return model.coef_, model.intercept_

def get_keras_initial_weights(columnIdx):
    inDim, outDim = get_dataset_shape(columnIdx)
    return get_keras_model(inputDim = inDim, outputDim = outDim).get_weights()

def get_SGD_sklearn_initial_weights(columnIdx):
    inDim, outDim = get_dataset_shape(columnIdx)
    return [np.random.rand(outDim, inDim), np.random.rand(outDim)]

def fit_model(mode, model, X, Y, weights = None, biases = None, batchSize = 1, epochs = 200):
    if(mode == "keras"):
        return fit_keras_model(model, X, Y, batchSize, epochs)
    elif(mode == "sklearn"):
        return fit_sklearn_model(model, X, Y)
    elif(mode == "sklearnSGD"):
        return fit_SGD_sklearn_model(model, X, Y, weights, biases)
    else:
        return None
    
def get_loss(mode, model, X, Y, pred_Y, target_Y):
    if(mode == "keras"):
        return get_keras_loss(model, X, Y)
    elif(mode == "sklearn" or mode == "sklearnSGD"):
        return get_sklearn_loss(pred_Y, target_Y)
    else:
        return None
    
def get_params(mode, model):
    if(mode == "keras"):
        return get_keras_params(model)
    elif(mode == "sklearn" or mode == "sklearnSGD"):
        return get_sklearn_params(model)
    else:
        return None
    
def get_initial_weights(mode, columnIdx = -1):
    if(mode == "keras"):
        return get_keras_initial_weights(columnIdx)
    elif(mode == "sklearnSGD"):
        return get_SGD_sklearn_initial_weights(columnIdx)
    else:
        return None

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