from tests.context import assert_equal
import pytest
from sympy import sin, Symbol

x = Symbol('x')


def test_overline():
    assert_equal("\\frac{\\sin(x)}{\\overline{x}_n}", sin(x) / Symbol('bar{x}_n'))
