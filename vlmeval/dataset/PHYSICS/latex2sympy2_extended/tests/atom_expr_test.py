import pytest
from latex2sympy2_extended import latex2sympy
from sympy import Symbol, Integer, Pow
from tests.context import assert_equal

# label, text, symbol_text
symbols = [
    ('letter', 'x', 'x'),
    ('greek letter', '\\lambda', 'lambda'),
    ('greek letter w/ space', '\\alpha ', 'alpha'),
    ('accented letter', '\\overline{x}', 'bar{x}')
]

subscripts = [
    ('2'),
    ('{23}'),
    ('i'),
    ('{ij}'),
    ('{i,j}'),
    ('{good}'),
    ('{x^2}')
]

examples = []
for symbol in symbols:
    for subscript in subscripts:
        examples.append(tuple(list(symbol) + [subscript]))


@pytest.mark.parametrize('label, text, symbol_text, subscript', examples)
def test_with_supexpr(label, text, symbol_text, subscript):
    assert_equal(text + '^2', Pow(Symbol(symbol_text), Integer(2)))


@pytest.mark.parametrize('label, text, symbol_text, subscript', examples)
def test_with_subexpr(label, text, symbol_text, subscript):
    assert_equal(text + '_' + subscript, Symbol(symbol_text + '_' + subscript))


@pytest.mark.parametrize('label, text, symbol_text, subscript', examples)
def test_with_subexpr_before_supexpr(label, text, symbol_text, subscript):
    assert_equal(text + '_' + subscript + '^2', Pow(Symbol(symbol_text + '_' + subscript), Integer(2)))


@pytest.mark.parametrize('label, text, symbol_text, subscript', examples)
def test_with_subexpr_before_supexpr_with_braces(label, text, symbol_text, subscript):
    wrapped_subscript = subscript if '{' in subscript else '{' + subscript + '}'
    assert_equal(text + '_' + wrapped_subscript + '^{2}', Pow(Symbol(symbol_text + '_' + subscript), Integer(2)))


@pytest.mark.parametrize('label, text, symbol_text, subscript', examples)
def test_with_supexpr_before_subexpr(label, text, symbol_text, subscript):
    assert_equal(text + '^2_' + subscript, Pow(Symbol(symbol_text + '_' + subscript), Integer(2)))


@pytest.mark.parametrize('label, text, symbol_text, subscript', examples)
def test_with_supexpr_before_subexpr_with_braces(label, text, symbol_text, subscript):
    wrapped_subscript = subscript if '{' in subscript else '{' + subscript + '}'
    assert_equal(text + '^{2}_' + wrapped_subscript, Pow(Symbol(symbol_text + '_' + subscript), Integer(2)))
