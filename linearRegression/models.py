import keras

def get_model(inputDim, outputDim):
    inputs = keras.Input(shape = (inputDim,))
    x = keras.layers.Dense(outputDim, activation = "linear")(inputs)
    outputs = keras.layers.Dense(outputDim)(x)

    model = keras.Model(inputs = inputs, outputs = outputs, name = "LinearRegression")
    model.compile(
        loss = keras.losses.MeanSquaredError(),
        optimizer = keras.optimizers.Adam(),
        metrics = []
    )
    return model