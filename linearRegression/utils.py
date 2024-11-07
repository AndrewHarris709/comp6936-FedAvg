from sklearn.metrics import mean_squared_error
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

def fit_sklearn_model(model, X, Y):
    return model.fit(X, Y)

def get_sklearn_loss(pred_Y, target_Y):
    return mean_squared_error(pred_Y, target_Y)

def get_sklearn_params(model):
    return model.coef_, model.intercept_