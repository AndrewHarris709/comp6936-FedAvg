from data_gen import NormalVariable
from pytest import fixture, approx
import numpy as np


@fixture
def static_var():
    return NormalVariable(5, 0)

@fixture
def unit_var():
    return NormalVariable(0, 1)


def test_generate_one(static_var):
    assert static_var.get() == 5


def test_generate_five(static_var):
    assert np.array_equal(static_var.get(5), [5, 5, 5, 5, 5])


def test_generate_std(unit_var):
    result = unit_var.get(1000)
    assert not np.array_equal(result, np.zeros(1000))
    assert result.mean() == approx(0, abs=0.1)
