from sympy import Add, Float, Mul, Rational, Symbol
from tests.context import assert_equal
import pytest

@pytest.mark.parametrize('latex, latex2sympy', [
    ('1 \\frac{1}{2}', Rational(3, 2)),
    ('1 \\frac{1}{2} + 3', Add(Rational(3, 2), 3)),
    ('3 1\\frac{1}{2}', Rational(9, 2)),
    # For float's we interpret it as multiplication
    ('3.1 \\frac{1}{2}', Mul(Float(3.1), Rational(1, 2))),
    # Negative numbers
    ('-3 \\frac{1}{2}', Rational(-7, 2)),

    # We didn't break other postfix
    ('ab', Mul(Symbol('a'), Symbol('b'))),

])
def test_mixed_fraction(latex, latex2sympy):
    assert_equal(latex, latex2sympy)
