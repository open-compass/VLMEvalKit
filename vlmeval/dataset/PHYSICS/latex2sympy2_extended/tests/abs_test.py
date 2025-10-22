import pytest
from latex2sympy2_extended import latex2sympy
from sympy import Abs

from tests.context import assert_equal, get_simple_examples

examples = get_simple_examples(Abs)

delimiter_pairs = {
    '|': '|',
    '\\vert': '\\vert',
    '\\lvert': '\\rvert'
}

@pytest.mark.parametrize('input, output, symbolically', examples)
def test_abs(input, output, symbolically):
    for left, right in delimiter_pairs.items():
        assert_equal("{left}{input}{right}".format(left=left, right=right, input=input), output, symbolically=symbolically)
        assert_equal("\\left{left}{input}\\right{right}".format(left=left, right=right, input=input), output, symbolically=symbolically)
        assert_equal("\\mleft{left}{input}\\mright{right}".format(left=left, right=right, input=input), output, symbolically=symbolically)
