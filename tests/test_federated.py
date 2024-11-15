from Client import Client
from Server import Server
from linearRegression.models import get_model
from linearRegression.utils import get_initial_weights, fit_model, get_params, get_dataset, get_splitted_dataset

from pytest import fixture
import numpy

@fixture
def mode():
    return "sklearnSGD"

@fixture
def idx():
    return 3

def test_single_client_single_column_convergence(mode, idx):
    data_X, data_Y = get_dataset(columnIdx = idx)

    client_1 = Client(name = "1", mode = mode)
    client_1.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = get_initial_weights(mode = mode, columnIdx = idx),
        clients = [client_1]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = idx)

    model = get_model(mode = "sklearn")
    model = fit_model("sklearn", model, data_X, data_Y)
    w_sklearn, b_sklearn = get_params("sklearn", model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 5) and numpy.allclose(b_fed, b_sklearn, atol = 5)

def test_multi_client_single_column_convergence(mode, idx):
    data_X, data_Y = get_dataset(columnIdx = idx)

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)
    client_3 = Client(name = "3", mode = mode)

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)
    client_3.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        mode = mode,
        participationRatio = 0.8,
        initialWeights = get_initial_weights(mode = mode, columnIdx = idx),
        clients = [client_1, client_2, client_3]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = idx)

    model = get_model(mode = "sklearn")
    model = fit_model("sklearn", model, data_X, data_Y)
    w_sklearn, b_sklearn = get_params("sklearn", model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 5) and numpy.allclose(b_fed, b_sklearn, atol = 5)

def test_similar_wb_client_server(mode):
    data_X, data_Y = get_dataset()

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)
    client_3 = Client(name = "3", mode = mode)

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)
    client_3.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = get_initial_weights(mode = mode),
        clients = [client_1, client_2, client_3]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model()

    w1, b1 = get_params(mode = mode, model = client_1.get_model())
    w2, b2 = get_params(mode = mode, model = client_2.get_model())
    w3, b3 = get_params(mode = mode, model = client_3.get_model())

    cliServ1 = numpy.allclose(w_fed, w1, atol = 2) and numpy.allclose(b_fed, b1, atol = 2)
    cliServ2 = numpy.allclose(w_fed, w2, atol = 2) and numpy.allclose(b_fed, b2, atol = 2)
    cliServ3 = numpy.allclose(w_fed, w3, atol = 2) and numpy.allclose(b_fed, b3, atol = 2)

    assert cliServ1 and cliServ2 and cliServ3

def test_participation_ratio(mode):
    data_X, data_Y = get_dataset()

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)
    client_3 = Client(name = "3", mode = mode)
    client_4 = Client(name = "4", mode = mode)

    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = data_X, new_Y = data_Y)
    client_3.add_data(new_X = data_X, new_Y = data_Y)
    client_4.add_data(new_X = data_X, new_Y = data_Y)

    server = Server(
        mode = mode,
        participationRatio = 0.5,
        initialWeights = get_initial_weights(mode = mode),
        clients = [client_1, client_2, client_3, client_4]
    )

    server.start_clients()

    clientCnt = 0
    for client in server.get_clients():
        if(client.model is not None):
            clientCnt += 1

    assert clientCnt == round(0.5 * 4)

def test_split_data_single_column_convergence(mode, idx):
    data_X, data_Y = get_splitted_dataset(numSplits = 3, columnIdx = idx)

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)
    client_3 = Client(name = "3", mode = mode)

    client_1.add_data(new_X = data_X[0], new_Y = data_Y[0])
    client_2.add_data(new_X = data_X[1], new_Y = data_Y[1])
    client_3.add_data(new_X = data_X[2], new_Y = data_Y[2])

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = get_initial_weights(mode = mode, columnIdx = idx),
        clients = [client_1, client_2, client_3]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = idx)

    data_X, data_Y = get_dataset(columnIdx = idx)
    model = get_model(mode = "sklearn")
    model = fit_model("sklearn", model, data_X, data_Y)
    w_sklearn, b_sklearn = get_params("sklearn", model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 15) and numpy.allclose(b_fed, b_sklearn, atol = 15)

def test_multiple_training_single_column_convergence(mode, idx):
    data_X, data_Y = get_splitted_dataset(numSplits = 2, columnIdx = idx)

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)

    client_1.add_data(new_X = data_X[0], new_Y = data_Y[0])
    client_2.add_data(new_X = data_X[1], new_Y = data_Y[1])

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = get_initial_weights(mode = mode, columnIdx = idx),
        clients = [client_1, client_2]
    )

    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = idx)
    server.start_clients()
    w_fed, b_fed = server.test_model(columnIdx = idx)

    data_X, data_Y = get_dataset(columnIdx = idx)
    model = get_model(mode = "sklearn")
    model = fit_model("sklearn", model, data_X, data_Y)
    w_sklearn, b_sklearn = get_params("sklearn", model)

    assert numpy.allclose(w_fed, w_sklearn, atol = 10) and numpy.allclose(b_fed, b_sklearn, atol = 10)