# Linear Regression

In this package, you can find files related to local models.

A simple linear regression model is used. The structure here allows to easily add more sophisticated models as well such as neural network, decision trees, etc. Initially, linear regression was emulated by implementing it using neural networks (Keras).

When transferring weights from the client to the server, and from the server to the inspector, an additional SGDRegressor is created. This model is not fitted with data, but provided with `coef_` & `intercept_` values. This allows us to evaluate our remotely-defined weights.

## models.py

We are using SGDRegressor from sklearn as the local model. Defining the model is as simple as calling ```get_model()```. This methods will return a SGDRegressor model. Number of training iterations can also be specified by passing ```max_iter``` to the method. Hyperparameters for this model are set by default to find the best possible linear fit.

## utils.py

This file provides utility functions for model interaction. The utility functions are as follows:

- ```fit_model```

    Calling ```fit_model``` will start the training process for SGDRegressor model. Training can be done either by using batches or the whole dataset at once. To use batches ```batchSize``` must be an integer number greater than 0. By default, the model will train on the whole dataset at once if batchSize is not passed as an input. Data points must be passed as ```X``` and the labels as ```Y```. Also, the new weights and biases for the model must be passed to the method by using ```weights``` and ```biases```.

- ```get_loss```
    
    Calling ```get_loss``` will return the mean squared error (MSE) loss value. The predicted values must be passed as ```pred_Y``` and the ground-truth labels must be passed as ```target_Y```.

- ```get_params```

    Calling ```get_params``` will return weights and biases of the model. This is used in sending model parameters from client to server or vice versa. The model must be passed as ```model```.

- ```get_code_params```

    This method is used for reading code parameters from a JSON file. It will return the code parameters and the code will use them for execution. Path of the JSON file must be specified by passing it as ```path```.

- ```get_weights_jsonified```

    Calling this method will convert model parameters to a JSON-compatible format. This is used in emitting model parameters from client to the server or vice versa. Model parameters are passed as ```list_of_arrays```.

- ```get_weights_dejsonified```

    Calling this method will convert model parameters to a NumPy-compatible format. This is used in working with model parameters after emission from client to the server or vice versa. Model parameters are passed as ```list_of_lists```.

- ```get_initial_weights```

    This method is used for getting the initial model parameters. The initial model parameters are random numbers. The biases are of size 1. The number of weights is passed as ```n```.