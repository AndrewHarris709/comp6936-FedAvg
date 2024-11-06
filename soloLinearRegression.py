import keras
from sklearn import datasets
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


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



data_X, data_Y = datasets.load_diabetes(return_X_y = True)

data_X = data_X[:, np.newaxis, 3] # Choosing only one feature

train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, test_size = 0.1, random_state = 1)

# scaler = StandardScaler()
# train_X = scaler.fit(train_X).transform(train_X)
# test_X = scaler.fit(test_X).transform(test_X)

if(len(train_Y.shape) == 1):
    outDim = 1
else:
    outDim = train_Y.shape[1]

model = get_model(inputDim = train_X.shape[1], outputDim = outDim)
history = model.fit(
    train_X,
    train_Y,
    batch_size = 1,
    epochs = 150,
    validation_data = (test_X, test_Y)
)

weightDense1 = model.layers[1].get_weights()[0]
biasDense1 = model.layers[1].get_weights()[1]
weightDense2 = model.layers[2].get_weights()[0]
biasDense2 = model.layers[2].get_weights()[1]
finalWeight = weightDense1 * weightDense2
finalBias = weightDense2 * biasDense1 + biasDense2

pred_loss = model.evaluate(test_X, test_Y)
print(f"Prediction Loss: {pred_loss}")
print(f"Coefficients (weights): {finalWeight}")
print(f"Independent Term (bias): {finalBias}")

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title(f"Linear Regression MSE")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Val"], loc = "upper left")
# plt.show()

plt.scatter(test_X, test_Y)
plt.plot(test_X, model.predict(test_X), color = "black")

plt.show()