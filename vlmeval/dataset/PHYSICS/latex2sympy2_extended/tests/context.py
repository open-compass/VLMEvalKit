from latex2sympy2_extended import latex2sympy
from sympy import (
    Add, Mul, Pow, Symbol, Rational, pi, sqrt,
    simplify, srepr
)

def _Add(a, b):
    return Add(a, b, evaluate=False)

def _Mul(a, b):
    return Mul(a, b, evaluate=False)

def _Pow(a, b):
    return Pow(a, b, evaluate=False)

def assert_equal(latex, expr, symbolically=False, variables: dict | None = None):
    parsed = latex2sympy(latex, variable_values=variables)
    if symbolically:
        assert simplify(parsed - expr) == 0
    else:
        actual_exp_tree = srepr(parsed)
        expected_exp_tree = srepr(expr)
        try:
            assert actual_exp_tree == expected_exp_tree
        except Exception:
            if (isinstance(parsed, (int, float)) or parsed.is_number) and \
               (isinstance(expr, (int, float)) or expr.is_number):
                assert parsed == expr or parsed - expr == 0 or simplify(parsed - expr) == 0
            else:
                print('expected_exp_tree = ', expected_exp_tree)
                print('actual exp tree = ', actual_exp_tree)
                raise

def get_simple_examples(func):
    '''
    Returns an array of tuples, containing the string `input`, sympy `output` using the provided sympy `func`, and `symbolically` boolean
    for calling `compare`.
    '''
    x = Symbol('x', real=None)
    y = Symbol('y', real=None)
    return [
        ("1.1", func(1.1), False),
        ("6.9", func(6.9), False),
        ("3.5", func(3.5), False),
        ("8", func(8), False),
        ("0", func(0), False),
        ("x", func(x), True),
        ("x + y", func(x + y), True),
        ("2y-y-y", func(2 * y - y - y), True)
    ]
