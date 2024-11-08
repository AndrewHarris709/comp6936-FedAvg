from sklearn.metrics import mean_squared_error
from sklearn import datasets
from linearRegression.models import get_keras_model
import numpy as np

def fit_keras_model(model, X, Y, batchSize = 1, epochs = 200):
    return model.fit(
        X,
        Y,
        batch_size = batchSize,
        epochs = epochs
    )

def get_keras_loss(model, X, Y):
    return model.evaluate(X, Y)

def get_keras_params(model):
    w1, b1 = model.layers[1].get_weights()
    w2, b2 = model.layers[2].get_weights()
    wTotal = np.multiply(w1, w2)
    bTotal = w2 * b1 + b2
    return wTotal, bTotal

def get_keras_initial_weights(columnIdx = -1):
    data_X, data_Y = get_dataset(columnIdx)
    if(len(data_Y.shape) == 1):
        outDim = 1
    else:
        outDim = data_Y.shape[1]    
    return get_keras_model(inputDim = data_X.shape[1], outputDim = outDim).get_weights()

def fit_sklearn_model(model, X, Y):
    return model.fit(X, Y)

def get_sklearn_loss(pred_Y, target_Y):
    return mean_squared_error(pred_Y, target_Y)

def get_sklearn_params(model):
    return model.coef_, model.intercept_

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
