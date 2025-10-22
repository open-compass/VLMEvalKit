import pytest
from latex2sympy2_extended import latex2sympy
from sympy import FiniteSet, Tuple
from tests.context import assert_equal, get_simple_examples

examples = get_simple_examples(lambda x: x)


@pytest.mark.parametrize('input, output, symbolically', examples)
def test_boxed_func(input, output, symbolically):
    assert_equal("\\boxed{{{input}}}".format(input=input), output, symbolically=symbolically)


@pytest.mark.parametrize('input, output, symbolically', [
    ("\\boxed{1,2,3}", FiniteSet(1, 2, 3), False),
    ("\\boxed{(1,2,3)}", Tuple(1, 2, 3), False),
    ("\\boxed{\\{1,2,3\\}}", FiniteSet(1, 2, 3), False),
])

def test_boxed_func_with_braces(input, output, symbolically):
    assert_equal(input, output, symbolically=symbolically)
