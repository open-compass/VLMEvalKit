import pytest
from latex2sympy2_extended import latex2sympy
from sympy import sin, Symbol
from tests.context import assert_equal

x = Symbol('x')


def test_left_right_cdot():
    assert_equal("\\sin\\left(x\\right)\\cdot x", sin(x) * x)
