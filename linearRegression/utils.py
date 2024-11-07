from sklearn.metrics import mean_squared_error

def fit_keras_model(model, X, Y, batchSize = 1, epochs = 100):
    return model.fit(
        X,
        Y,
        batch_size = batchSize,
        epochs = epochs,
    )

def get_keras_loss(model, X, Y):
    return model.evaluate(X, Y)

def fit_sklearn_model(model, X, Y):
    return model.fit(X, Y)

def get_sklearn_loss(pred_Y, target_Y):
    return mean_squared_error(pred_Y, target_Y)