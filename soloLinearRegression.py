import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

trainTestSplit = 0.1

data_X, data_Y = datasets.load_diabetes(return_X_y = True)

numTest = int(data_X.shape[0] * trainTestSplit)
X_train = data_X[:-numTest]; X_test = data_X[-numTest:]
Y_train = data_Y[:-numTest]; Y_test = data_Y[-numTest:]

regressor = linear_model.LinearRegression()
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

print(f"Coefficients (weights): {regressor.coef_}")
print(f"Independent Term (bias): {regressor.intercept_}")
print(f"MSE: {mean_squared_error(Y_test, Y_pred)}")
