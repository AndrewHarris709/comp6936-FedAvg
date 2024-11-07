from sklearn import datasets
import numpy as np
# import matplotlib.pyplot as plt


from linearRegression.models import get_model

columnIdx = 3

data_X, data_Y = datasets.load_diabetes(return_X_y = True)
if(columnIdx >= 0):
    data_X = data_X[:, np.newaxis, columnIdx]

if(len(data_Y.shape) == 1):
    outDim = 1
else:
    outDim = data_Y.shape[1]

model = get_model(inputDim = data_X.shape[1], outputDim = outDim)
history = model.fit(
    data_X,
    data_Y,
    batch_size = 1,
    epochs = 150,
)

# weightDense1 = model.layers[1].get_weights()[0]
# biasDense1 = model.layers[1].get_weights()[1]
# weightDense2 = model.layers[2].get_weights()[0]
# biasDense2 = model.layers[2].get_weights()[1]
# finalWeight = weightDense1 * weightDense2
# finalBias = weightDense2 * biasDense1 + biasDense2

pred_loss = model.evaluate(data_X, data_Y)
print(f"Prediction Loss: {pred_loss}")
# print(f"Coefficients (weights): {finalWeight}")
# print(f"Independent Term (bias): {finalBias}")

# plt.plot(history.history["loss"])
# plt.plot(history.history["val_loss"])
# plt.title(f"Linear Regression MSE")
# plt.ylabel("Loss")
# plt.xlabel("Epoch")
# plt.legend(["Train", "Val"], loc = "upper left")
# plt.show()

# plt.scatter(test_X, test_Y)
# plt.plot(test_X, model.predict(test_X), color = "black")

# plt.show()