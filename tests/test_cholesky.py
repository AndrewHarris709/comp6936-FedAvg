from generators import CholeskyGenerator
import numpy as np
from pytest import raises, fixture
from unittest.mock import Mock

@fixture
def rng_mock():
    rng_mock = Mock()
    rng_mock.normal = Mock(return_value=np.zeros(2))
    return rng_mock

def test_corr_not_symmetric():
    with raises(TypeError):
        CholeskyGenerator(np.array([[1, 0], [0.5, 1]]), np.array([0, 0]))


def test_corr_excessive_values():
    with raises(np.linalg.LinAlgError):
        CholeskyGenerator(np.array([[1, 1.2], [1.2, 1]]), np.array([0, 0]))


def test_corr_impossible_correlation():
    # With this matrix variable 3 is strongly correlated with variables 1 & 2, but 
    # variables 1 & 2 are not correlated with each other.
    # This arrangement is not possible.
    corr = np.array([[1, 0, 0.8], [0, 1, 0.8], [0.8, 0.8, 1]])

    with raises(np.linalg.LinAlgError):
        CholeskyGenerator(corr, np.array([0, 0]))


def test_non_numpy_arrays():
    CholeskyGenerator([[1, 0], [0, 1]], [0, 0])


def test_simple_gen(rng_mock):
    simple_gen = CholeskyGenerator(np.array([[1, 0], [0, 1]]), np.array([0, 0]), rng=rng_mock)

    assert np.array_equal(simple_gen.get(), np.array([[0], [0]]))
    rng_mock.normal.assert_called_with(loc=0, scale=1, size=(2, 1))

    rng_mock.normal = Mock(return_value=np.zeros((2, 2)))
    assert np.array_equal(simple_gen.get(2), np.zeros((2, 2)))
    rng_mock.normal.assert_called_with(loc=0, scale=1, size=(2, 2))

    rng_mock.normal = Mock(return_value=np.zeros((2, 100)))
    assert np.array_equal(simple_gen.get(100), np.zeros((2, 100)))
    rng_mock.normal.assert_called_with(loc=0, scale=1, size=(2, 100))


def test_simple_gen_shift(rng_mock):
    simple_gen = CholeskyGenerator(np.array([[1, 0], [0, 1]]), np.array([1, 2]), rng=rng_mock)

    assert np.array_equal(simple_gen.get(), np.array([[1], [2]]))
    rng_mock.normal.assert_called_with(loc=0, scale=1, size=(2, 1))

    rng_mock.normal = Mock(return_value=np.zeros((2, 2)))
    assert np.array_equal(simple_gen.get(2), [np.ones(2), np.full(2, 2)])
    rng_mock.normal.assert_called_with(loc=0, scale=1, size=(2, 2))

    rng_mock.normal = Mock(return_value=np.zeros((2, 100)))
    assert np.array_equal(simple_gen.get(100), [np.ones(100), np.full(100, 2)])
    rng_mock.normal.assert_called_with(loc=0, scale=1, size=(2, 100))


def test_full_shift(rng_mock):
    rng_mock.normal = Mock(return_value=np.zeros((5, 1)))
    full_gen = CholeskyGenerator(np.diagflat(np.ones(5)), np.array([0, 1, 2, 3, 4]), rng=rng_mock)
    assert np.array_equal(full_gen.get(), np.array([[0], [1], [2], [3], [4]]))

    rng_mock.normal = Mock(return_value=np.zeros((5, 100)))
    assert np.array_equal(full_gen.get(100), np.full((5, 100), [[0], [1], [2], [3], [4]]))


def test_distributions():
    full_gen = CholeskyGenerator(np.diagflat(np.ones(5)), np.array([0, 1, 2, 3, 4]))
    result = full_gen.get(1000)
    assert not np.array_equal(result, np.full((5, 100), [[0], [1], [2], [3], [4]]))

    closeness = np.isclose(result.mean(axis=1), [0, 1, 2, 3, 4], atol=0.1)
    assert closeness.all()

def test_mult(rng_mock):
    simple_gen = CholeskyGenerator(np.array([[1, 0], [0, 1]]), np.array([0, 0]), np.array([2, -1]), rng=rng_mock)

    rng_mock.normal = Mock(return_value=np.ones((2, 1)))
    assert np.array_equal(simple_gen.get(), np.array([[2], [-1]]))

    rng_mock.normal = Mock(return_value=np.ones((2, 2)))
    assert np.array_equal(simple_gen.get(2), [np.full(2, 2), np.full(2, -1)])

    rng_mock.normal = Mock(return_value=np.ones((2, 100)))
    assert np.array_equal(simple_gen.get(100), [np.full(100, 2), np.full(100, -1)])
