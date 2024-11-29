from charset_normalizer.cd import filter_alt_coherence_matches

from flaskFederated.Client import Client
from flaskFederated.Server import Server
from linearRegression.models import get_model
from linearRegression.utils import fit_model, get_params, get_initial_weights

from pytest import fixture
import numpy
from sklearn import datasets

@fixture
def single_X():
    return datasets.load_diabetes(return_X_y=True)[0][:,4].reshape(-1, 1)

@fixture
def single_Y():
    return datasets.load_diabetes(return_X_y=True)[1]

def test_single_client_single_column_convergence(single_X, single_Y):
    client_1 = Client(name = "1", failure_rate=0)
    client_1.data_X = single_X
    client_1.data_Y = single_Y

    initial_weights = get_initial_weights(1)
    server = Server(
        participationRatio = 1,
        initialWeights = initial_weights,
        max_iter=10000
    )
    server.add_client(client_1.name)
    assert server.select_clients() == [client_1.name]

    client_result = client_1.train(server.get_weights())
    server.update_weights(client_result, client_1.name)

    w_fed, b_fed = server.get_weights()

    model = get_model(max_iter=10000)
    model = fit_model(model, single_X, single_Y, initial_weights[0], initial_weights[1], batchSize=16)
    w_sklearn, b_sklearn = get_params(model)

    assert numpy.allclose(w_fed, w_sklearn)
    assert numpy.allclose(b_fed, b_sklearn)

def test_multi_client_single_column_convergence():
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

def test_similar_wb_client_server():
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

def test_participation_ratio():
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

def test_split_data_single_column_convergence(idx):
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

def test_multiple_training_single_column_convergence(idx):
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

def test_no_training_data():
    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)

    initW = get_initial_weights(mode = mode)

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = initW,
        clients = [client_1, client_2]
    )
    server.start_clients()
    w_fed_1, b_fed_1 = server.test_model()
    compare_1 = numpy.array_equal(w_fed_1, initW[0]) and numpy.array_equal(b_fed_1, initW[1])
    server.start_clients()
    w_fed_2, b_fed_2 = server.test_model()
    compare_2 = numpy.array_equal(w_fed_2, initW[0]) and numpy.array_equal(b_fed_2, initW[1])
    server.start_clients()
    w_fed_3, b_fed_3 = server.test_model()
    compare_3 = numpy.array_equal(w_fed_3, initW[0]) and numpy.array_equal(b_fed_3, initW[1])

    assert compare_1 and compare_2 and compare_3

def test_client_selection():
    data_X, data_Y = get_splitted_dataset(numSplits = 5)

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)
    client_3 = Client(name = "3", mode = mode)
    client_4 = Client(name = "4", mode = mode)
    client_5 = Client(name = "5", mode = mode)
    client_1.add_data(new_X = data_X[0], new_Y = data_Y[0])
    client_2.add_data(new_X = data_X[1], new_Y = data_Y[1])
    client_3.add_data(new_X = data_X[2], new_Y = data_Y[2])
    client_4.add_data(new_X = data_X[3], new_Y = data_Y[3])
    client_5.add_data(new_X = data_X[4], new_Y = data_Y[4])

    server = Server(
        mode = mode,
        participationRatio = 0.2,
        initialWeights = get_initial_weights(mode = mode),
        clients = [client_1, client_2, client_3, client_4, client_5]
    )
    server.start_clients()
    server.start_clients()
    server.start_clients()

    cnt = 0
    for client in server.get_clients():
        if(client.get_model() is not None):
            cnt += 1
    
    assert cnt == 1 or cnt == 2 or cnt == 3

def test_training_lean():
    split_data_X, split_data_Y = get_splitted_dataset(numSplits = 10)
    data_X, data_Y = get_dataset()

    client_1 = Client(name = "1", mode = mode)
    client_2 = Client(name = "2", mode = mode)
    client_1.add_data(new_X = data_X, new_Y = data_Y)
    client_2.add_data(new_X = split_data_X[0], new_Y = split_data_Y[0])

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = get_initial_weights(mode = mode),
        clients = [client_1, client_2]
    )
    server.start_clients()
    w_fed, b_fed = server.test_model()
    w1, b1 = get_params(mode = mode, model = client_1.get_model())
    w2, b2 = get_params(mode = mode, model = client_2.get_model())

    assert numpy.linalg.norm(w1 - w_fed) < numpy.linalg.norm(w2 - w_fed)
    assert numpy.linalg.norm(b1 - b_fed) < numpy.linalg.norm(b2 - b_fed)

def test_continuous_new_client_data(idx):
    data_X, data_Y = get_splitted_dataset(numSplits = 3, columnIdx = idx)

    client_1 = Client(name = "1", mode = mode)

    initW = get_initial_weights(mode = mode, columnIdx = idx)

    server = Server(
        mode = mode,
        participationRatio = 1,
        initialWeights = initW,
        clients = [client_1]
    )
    
    server.start_clients()
    w_fed_1, b_fed_1 = server.test_model(columnIdx = idx)
    model_1 = client_1.get_model()
    client_1.add_data(new_X = data_X[0], new_Y = data_Y[0])
    server.start_clients()
    w_fed_2, b_fed_2 = server.test_model(columnIdx = idx)
    client_1.add_data(new_X = data_X[1], new_Y = data_Y[1])
    server.start_clients()
    w_fed_3, b_fed_3 = server.test_model(columnIdx = idx)
    client_1.add_data(new_X = data_X[2], new_Y = data_Y[2])
    server.start_clients()
    w_fed_4, b_fed_4 = server.test_model(columnIdx = idx)

    data_X, data_Y = get_dataset(columnIdx = idx)
    model = get_model(mode = "sklearn")
    model = fit_model("sklearn", model, data_X, data_Y)
    w_sklearn, b_sklearn = get_params("sklearn", model)

    assert model_1 is None
    assert numpy.array_equal(w_fed_1, initW[0]) and numpy.array_equal(b_fed_1, initW[1])
    assert numpy.linalg.norm(w_sklearn - w_fed_2) < numpy.linalg.norm(w_sklearn - w_fed_1)
    assert numpy.linalg.norm(w_sklearn - w_fed_3) < numpy.linalg.norm(w_sklearn - w_fed_1)
    assert numpy.linalg.norm(w_sklearn - w_fed_3) < numpy.linalg.norm(w_sklearn - w_fed_2)
    assert numpy.linalg.norm(w_sklearn - w_fed_4) < numpy.linalg.norm(w_sklearn - w_fed_1)
    assert numpy.linalg.norm(w_sklearn - w_fed_4) < numpy.linalg.norm(w_sklearn - w_fed_2)
    assert numpy.linalg.norm(w_sklearn - w_fed_4) < numpy.linalg.norm(w_sklearn - w_fed_3)
    assert numpy.allclose(b_fed_2, b_sklearn, atol = 2)
    assert numpy.allclose(b_fed_3, b_sklearn, atol = 2)
    assert numpy.allclose(b_fed_4, b_sklearn, atol = 2)
   