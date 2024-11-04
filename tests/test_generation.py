from data_gen import NormalVariable, VariableRelation
from pytest import fixture, approx, raises
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


def test_error_on_no_weights(static_var, unit_var):
    with raises(TypeError):
        rel = VariableRelation([static_var], unit_var)


def test_direct_relation():
    var1 = NormalVariable(3, 0)
    var2 = NormalVariable(0, 0)

    rel = VariableRelation([(var1, 2)], var2)
    assert np.array_equal(rel.get(), [[3], [6]])
    assert np.array_equal(rel.get(5), [[3, 3, 3, 3, 3], [6, 6, 6, 6, 6]])

