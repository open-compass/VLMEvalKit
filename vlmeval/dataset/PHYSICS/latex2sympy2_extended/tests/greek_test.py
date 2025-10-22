import pytest
from latex2sympy2_extended import latex2sympy
from sympy import Symbol
from tests.context import assert_equal

epsilon_lower = Symbol('epsilon')
varepsilon = Symbol('varepsilon')


def test_greek_epsilon():
    assert_equal("\\epsilon", epsilon_lower)


def test_greek_epsilon_upper():
    assert_equal('\\char"000190', epsilon_lower)


def test_greek_varepsilon():
    assert_equal('\\varepsilon', varepsilon)
