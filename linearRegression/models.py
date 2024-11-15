import keras
from sklearn import linear_model

def get_keras_model(inputDim, outputDim):
    inputs = keras.Input(shape = (inputDim,))
    x = keras.layers.Dense(outputDim, activation = "linear")(inputs)
    outputs = keras.layers.Dense(outputDim)(x)

    model = keras.Model(inputs = inputs, outputs = outputs, name = "LinearRegression")
    model.compile(
        loss = keras.losses.MeanSquaredError(),
        optimizer = keras.optimizers.Adam(learning_rate = 0.01),
        metrics = []
    )
    return model

def get_sklearn_model():
    return linear_model.LinearRegression()

def get_SGD_sklearn_model():
    return linear_model.SGDRegressor(
        penalty = None,
        alpha = 0.0,
        l1_ratio = 0,
        max_iter = 50000,
        tol = None,
    )

def get_model(mode, inputDim = None, outputDim = None):
    if(mode == "keras"):
        return get_keras_model(inputDim, outputDim)
    elif(mode == "sklearn"):
        return get_sklearn_model()
    elif(mode == "sklearnSGD"):
        return get_SGD_sklearn_model()
    else:
        return None