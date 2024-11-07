def fit_keras_model(model, X, Y, batchSize = 1, epochs = 100):
    return model.fit(
        X,
        Y,
        batch_size = batchSize,
        epochs = epochs,
    )