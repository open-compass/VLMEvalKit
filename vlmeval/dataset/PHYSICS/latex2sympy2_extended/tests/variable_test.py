from tests.context import assert_equal
import pytest
from sympy import Rational, UnevaluatedExpr, Symbol, Mul, Pow, Max, Min, gcd, lcm, floor, ceiling

x = Symbol('x')
y = Symbol('y')


def test_variable_letter():
    assert_equal("\\variable{x}", Symbol('x'))


def test_variable_digit():
    assert_equal("\\variable{1}", Symbol('1'))


def test_variable_letter_subscript():
    assert_equal("\\variable{x_y}", Symbol('x_y'))


def test_variable_letter_comma_subscript():
    assert_equal("\\variable{x_{i,j}}", Symbol('x_{i,j}'))


def test_variable_digit_subscript():
    assert_equal("\\variable{x_1}", Symbol('x_1'))


def test_variable_after_subscript_required():
    with pytest.raises(Exception):
        assert_equal("\\variable{x_}", Symbol('x_'))


def test_variable_before_subscript_required():
    with pytest.raises(Exception):
        assert_equal("\\variable{_x}", Symbol('_x'))


def test_variable_bad_name():
    with pytest.raises(Exception):
        assert_equal("\\variable{\\sin xy}", None)


def test_variable_in_expr():
    assert_equal("4\\cdot\\variable{x}", 4 * Symbol('x'))


def test_variable_greek_letter():
    assert_equal("\\variable{\\alpha }\\alpha", Symbol('\\alpha ') * Symbol('alpha'))


def test_variable_greek_letter_subscript():
    assert_equal("\\variable{\\alpha _{\\beta }}\\alpha ", Symbol('\\alpha _{\\beta }') * Symbol('alpha'))


def test_variable_bad_unbraced_long_subscript():
    with pytest.raises(Exception):
        assert_equal("\\variable{x_yz}", None)


def test_variable_bad_unbraced_long_complex_subscript():
    with pytest.raises(Exception):
        assert_equal("\\variable{x\\beta 10_y\\alpha 20}", None)


def test_variable_braced_subscript():
    assert_equal("\\variable{x\\beta 10_{y\\alpha 20}}", Symbol('x\\beta 10_{y\\alpha 20}'))


def test_variable_complex_expr():
    assert_equal("4\\cdot\\variable{value1}\\frac{\\variable{value_2}}{\\variable{a}}\\cdot x^2", 4 * Symbol('value1') * Symbol('value_2') / Symbol('a') * x**2)


def test_variable_dollars():
    assert_equal("\\$\\variable{x}", Symbol('x'))


def test_variable_percentage():
    assert_equal("\\variable{x}\\%", Symbol('x') * Rational(1, 100))


def test_variable_single_arg_func():
    assert_equal("\\floor(\\variable{x})", floor(Symbol('x')))
    assert_equal("\\ceil(\\variable{x})", ceiling(Symbol('x')))


def test_variable_multi_arg_func():
    assert_equal("\\gcd(\\variable{x}, \\variable{y})", UnevaluatedExpr(gcd(Symbol('x'), Symbol('y'))))
    assert_equal("\\lcm(\\variable{x}, \\variable{y})", UnevaluatedExpr(lcm(Symbol('x'), Symbol('y'))))
    assert_equal("\\max(\\variable{x}, \\variable{y})", Max(Symbol('x'), Symbol('y'), evaluate=False))
    assert_equal("\\min(\\variable{x}, \\variable{y})", Min(Symbol('x'), Symbol('y'), evaluate=False))
