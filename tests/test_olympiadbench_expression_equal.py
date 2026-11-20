"""Offline regressions for OlympiadBench's symbolic answer comparison.

MathJudger.expression_equal calls self.sympy_sub_pi before the numeric and
simplify comparisons. The method's only definition was commented out, so every
call raised AttributeError; is_equal swallows it with `except Exception: pass`,
which made the whole evalf / simplify block unreachable and scored every
answer that was not character-identical to the gold answer as wrong.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytest.importorskip('sympy', reason='OlympiadBench scoring needs sympy')
pytest.importorskip('antlr4', reason='sympy.parsing.latex needs the antlr4 runtime')


@pytest.fixture(scope='module')
def judger():
    spec = importlib.util.spec_from_file_location(
        'olympiadbench_under_test', REPO / 'vlmeval/dataset/utils/olympiadbench.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MathJudger()


def test_sympy_sub_pi_is_defined(judger):
    """expression_equal calls it unconditionally on the non-identical path."""
    assert callable(getattr(judger, 'sympy_sub_pi', None))


def test_expression_equal_does_not_raise_attributeerror(judger):
    """is_equal swallows exceptions, so the raise has to be asserted directly."""
    assert judger.expression_equal(r'\frac{1}{2}', '0.5') is True


@pytest.mark.parametrize('gold, prediction', [
    # Same value, different spelling -- the case the numeric branch exists for.
    (r'\frac{1}{2}', '0.5'),
    (r'\frac{3}{6}', '0.5'),
    (r'\sqrt{4}', '2'),
    # pi substituted for its numeric value, which is what sympy_sub_pi is for.
    (r'\frac{\pi}{3}', '1.0471975512'),
    (r'2\pi', '6.283185307'),
    # Symbolic answers, which reach the simplify branch.
    (r'\frac{1}{2}mv^2', '0.5mv^2'),
    ('2x', 'x+x'),
])
def test_equivalent_expressions_are_equal(judger, gold, prediction):
    assert judger.is_equal(gold, prediction) is True


@pytest.mark.parametrize('gold, prediction', [
    (r'\frac{1}{2}', '0.6'),
    ('1', '2'),
    ('2x', '3x'),
    (r'\frac{\pi}{3}', '2.5'),
])
def test_inequivalent_expressions_are_not_equal(judger, gold, prediction):
    assert judger.is_equal(gold, prediction) is False


def test_identical_expressions_still_short_circuit(judger):
    """The structural equality fast path above the call must be untouched."""
    assert judger.is_equal(r'\frac{1}{2}', r'\frac{1}{2}') is True
