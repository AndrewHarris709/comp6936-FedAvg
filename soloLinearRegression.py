from sklearn import datasets
import numpy as np
# import matplotlib.pyplot as plt

from linearRegression.models import *
from linearRegression.utils import *

columnIdx = 3
modelType = "sklearn"

data_X, data_Y = datasets.load_diabetes(return_X_y = True)
if(columnIdx >= 0):
    data_X = data_X[:, np.newaxis, columnIdx]

if(modelType == "keras"):
    if(len(data_Y.shape) == 1):
        outDim = 1
    else:
        outDim = data_Y.shape[1]
    model = get_keras_model(inputDim = data_X.shape[1], outputDim = outDim)
    history = fit_keras_model(model, data_X, data_Y)
    print(f"Keras Loss: {get_keras_loss(model, data_X, data_Y)}")
    weight, bias = get_keras_params(model)
    print(f"Keras Weight: {weight}")
    print(f"Keras Bias: {bias}")
else:
    model = get_sklearn_model()
    model = fit_sklearn_model(model, data_X, data_Y)
    print(f"Sklearn Loss: {get_sklearn_loss(model.predict(data_X), data_Y)}")
    weight, bias = get_sklearn_params(model)
    print(f"Sklearn Weights: {weight}")
    print(f"Sklearn Biases: {bias}")

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