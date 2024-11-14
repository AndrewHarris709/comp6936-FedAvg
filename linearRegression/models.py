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
        l1_ratio = 0,
        max_iter = 10000
    )