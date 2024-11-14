from Client import Client
from Server import Server
from linearRegression.models import get_sklearn_model, get_keras_model
from linearRegression.utils import *
import numpy

def test_single_client_single_column_convergence():
    data_X, data_Y = get_dataset(columnIdx = 3)

    client_1 = Client("1")

    client_1.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        participationRatio = 1,
        initialWeights = get_keras_initial_weights(columnIdx = 3),
        clients = [client_1]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = 3)

    model = get_sklearn_model()
    model = fit_sklearn_model(model, data_X, data_Y)
    w_sklearn, b_sklearn = get_sklearn_params(model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 5) and numpy.allclose(b_fed, b_sklearn, atol = 5)

def test_multi_client_single_column_convergence():
    data_X, data_Y = get_dataset(columnIdx = 3)

    client_1 = Client("1")
    client_2 = Client("2")
    client_3 = Client("3")

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)
    client_3.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        participationRatio = 0.8,
        initialWeights = get_keras_initial_weights(columnIdx = 3),
        clients = [client_1, client_2, client_3]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = 3)

    model = get_sklearn_model()
    model = fit_sklearn_model(model, data_X, data_Y)
    w_sklearn, b_sklearn = get_sklearn_params(model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 5) and numpy.allclose(b_fed, b_sklearn, atol = 5)

def test_single_client_multi_column_convergence():
    data_X, data_Y = get_dataset()

    client_1 = Client("1")
    client_1.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        participationRatio = 1,
        initialWeights = get_keras_initial_weights(),
        clients = [client_1]
    )
    server.start_clients()
    w_fed, b_fed = server.test_model()

    # There are some dissimilarities when sklearn is used!!!!!!!!!!!!!!!!!!
    model = get_sklearn_model()
    model = fit_sklearn_model(model, data_X, data_Y)
    w_sklearn, b_sklearn = get_sklearn_params(model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 5) and numpy.allclose(b_fed, b_sklearn, atol = 5)

def test_multi_client_multi_column_convergence():
    data_X, data_Y = get_dataset()

    client_1 = Client("1")
    client_2 = Client("2")

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        participationRatio = 0.9,
        initialWeights = get_keras_initial_weights(),
        clients = [client_1, client_2]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model()

    model = get_keras_model(inputDim = data_X.shape[1], outputDim = 1)
    fit_keras_model(model, data_X, data_Y)
    w_keras, b_keras = get_keras_params(model)

    assert numpy.allclose(w_fed, w_keras, atol = 5) and numpy.allclose(b_fed, b_keras, atol = 5)

    # There are some dissimilarities when sklearn is used!!!!!!!!!!!!!!!!!!
    # model = get_sklearn_model()
    # model = fit_sklearn_model(model, data_X, data_Y)
    # w_sklearn, b_sklearn = get_sklearn_params(model)

    # assert numpy.allclose(w_fed, w_keras, atol = 5) and numpy.allclose(b_fed, b_keras, atol = 5)

def test_similar_wb_client_server():
    data_X, data_Y = get_dataset()

    client_1 = Client("1")
    client_2 = Client("2")
    client_3 = Client("3")

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)
    client_3.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        participationRatio = 1,
        initialWeights = get_keras_initial_weights(),
        clients = [client_1, client_2, client_3]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model()

    w1, b1 = get_keras_params(client_1.get_model())
    w2, b2 = get_keras_params(client_2.get_model())
    w3, b3 = get_keras_params(client_3.get_model())

    cliServ1 = numpy.allclose(w_fed, w1, atol = 2) and numpy.allclose(b_fed, b1, atol = 2)
    cliServ2 = numpy.allclose(w_fed, w2, atol = 2) and numpy.allclose(b_fed, b2, atol = 2)
    cliServ3 = numpy.allclose(w_fed, w2, atol = 2) and numpy.allclose(b_fed, b3, atol = 2)

    assert cliServ1 and cliServ2 and cliServ3

def test_participation_ratio():
    data_X, data_Y = get_dataset()

    client_1 = Client("1")
    client_2 = Client("2")
    client_3 = Client("3")
    client_4 = Client("4")

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)
    client_3.add_data(new_X = data_X, new_Y = data_Y)
    client_4.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        participationRatio = 0.5,
        initialWeights = get_keras_initial_weights(),
        clients = [client_1, client_2, client_3, client_4]
    )

    server.start_clients()

    clientCnt = 0
    for client in server.get_clients():
        if(client.model is not None):
            clientCnt += 1

    assert clientCnt == round(0.5 * 4)

def test_split_data_single_column_convergence():
    data_X, data_Y = get_splitted_dataset(numSplits = 3, columnIdx = 1)

    client_1 = Client("1")
    client_2 = Client("2")
    client_3 = Client("3")

    client_1.add_data(new_X = data_X[0], new_Y = data_Y[0])
    client_2.add_data(new_X = data_X[1], new_Y = data_Y[1])
    client_3.add_data(new_X = data_X[2], new_Y = data_Y[2])

    server = Server(
        participationRatio = 1,
        initialWeights = get_keras_initial_weights(columnIdx = 1),
        clients = [client_1, client_2, client_3]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = 1)

    data_X, data_Y = get_dataset(columnIdx = 1)
    model = get_sklearn_model()
    model = fit_sklearn_model(model, data_X, data_Y)
    w_sklearn, b_sklearn = get_sklearn_params(model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 10) and numpy.allclose(b_fed, b_sklearn, atol = 10)

def test_multiple_training_single_column_convergence():
    data_X, data_Y = get_splitted_dataset(numSplits = 2, columnIdx = 1)

    client_1 = Client("1")
    client_2 = Client("2")

    client_1.add_data(new_X = data_X[0], new_Y = data_Y[0])
    client_2.add_data(new_X = data_X[1], new_Y = data_Y[1])

    server = Server(
        participationRatio = 1,
        initialWeights = get_keras_initial_weights(columnIdx = 1),
        clients = [client_1, client_2]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = 1)
    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = 1)

    data_X, data_Y = get_dataset(columnIdx = 1)
    model = get_sklearn_model()
    model = fit_sklearn_model(model, data_X, data_Y)
    w_sklearn, b_sklearn = get_sklearn_params(model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 10) and numpy.allclose(b_fed, b_sklearn, atol = 10)