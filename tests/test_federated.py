from Client import Client
from Server import Server
from linearRegression.utils import get_dataset, get_keras_initial_weights
from linearRegression.models import get_sklearn_model
from linearRegression.utils import fit_sklearn_model, get_sklearn_params
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

    assert numpy.allclose(w_fed, w_sklearn, atol = 3) and numpy.allclose(b_fed, b_sklearn, atol = 3)

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

    assert numpy.allclose(w_fed, w_sklearn, atol = 3) and numpy.allclose(b_fed, b_sklearn, atol = 3)

def test_single_client_multi_column_convergence():
    pass

def test_multi_client_multi_column_convergence():
    pass

def test_similar_wb_client_server():
    pass