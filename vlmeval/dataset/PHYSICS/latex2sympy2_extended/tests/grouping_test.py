from latex2sympy2_extended import latex2sympy
import pytest
from sympy import Integral, sin, Symbol, Mul, Integer, Pow
from tests.context import assert_equal, _Add, _Mul, _Pow

a = Symbol('a')
b = Symbol('b')
x = Symbol('x')
theta = Symbol('theta')


func_arg_examples = [
    ('\\int ', 'x dx', Integral(x, x)),
    ('\\sin', '\\theta ', sin(theta))
]

example_groups = [
    ('1+2', '3-4', _Mul(_Add(1, 2), _Add(3, _Mul(-1, 4))))
]

modifiable_delimiter_pairs = {
    '(': ')',
    '\\lgroup': '\\rgroup',
    '\\{': '\\}',
    '\\lbrace': '\\rbrace',
    '[': ']',
    '\\lbrack': '\\rbrack',
}


@pytest.mark.parametrize('func, args, output', func_arg_examples)
def test_func_arg_groupings(func, args, output):
    # none
    assert_equal("{func} {args}".format(func=func, args=args), output)
    # normal brace (not modifiable)
    assert_equal("{func}{{{args}}}".format(func=func, args=args), output)
    # rest of delimiters, with modifications
    for left, right in modifiable_delimiter_pairs.items():
        assert_equal("{func}{left}{args}{right}".format(left=left, right=right, func=func, args=args), output)
        assert_equal("{func}\\left{left}{args}\\right{right}".format(left=left, right=right, func=func, args=args), output)
        assert_equal("{func}\\mleft{left}{args}\\mright{right}".format(left=left, right=right, func=func, args=args), output)


@pytest.mark.parametrize('group1, group2, output', example_groups)
def test_delimiter_groupings(group1, group2, output):
    # normal brace (not modifiable)
    assert_equal("{{{group1}}}{{{group2}}}".format(group1=group1, group2=group2), output)
    # rest of delimiters, with modifications
    for left, right in modifiable_delimiter_pairs.items():
        assert_equal("{left}{group1}{right}{left}{group2}{right}".format(left=left, right=right, group1=group1, group2=group2), output)
        assert_equal("\\left{left}{group1}\\right{right}\\left{left}{group2}\\right{right}".format(left=left, right=right, group1=group1, group2=group2), output)
        assert_equal("\\mleft{left}{group1}\\mright{right}\\mleft{left}{group2}\\mright{right}".format(left=left, right=right, group1=group1, group2=group2), output)
