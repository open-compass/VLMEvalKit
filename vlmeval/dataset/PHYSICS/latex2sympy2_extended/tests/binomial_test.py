import pytest
from latex2sympy2_extended import latex2sympy
from sympy import binomial, Symbol
from tests.context import assert_equal, _Add, _Mul, _Pow

x = Symbol('x')
y = Symbol('y')
theta = Symbol('theta')
gamma = Symbol('gamma')


def test_binomial_numeric():
    assert_equal("\\binom{16}{2}", binomial(16, 2))


def test_binomial_symbols():
    assert_equal("\\binom{x}{y}", binomial(x, y))


def test_binomial_greek_symbols():
    assert_equal("\\binom{\\theta}{\\gamma}", binomial(theta, gamma))


def test_binomial_expr():
    assert_equal("\\binom{16+2}{\\frac{4}{2}}", binomial(_Add(16, 2), _Mul(4, _Pow(2, -1)), evaluate=False))


def test_choose_numeric():
    assert_equal("{16 \\choose 2}", binomial(16, 2))


def test_choose_symbols():
    assert_equal("{x \\choose y}", binomial(x, y))


def test_choose_greek_symbols():
    assert_equal("{\\theta \\choose \\gamma}", binomial(theta, gamma))
