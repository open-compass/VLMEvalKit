"""Regression tests for the deterministic VMCBench fallback (#1867).

The parser module only needs numpy/pandas, so it is loaded by path to keep
the unit tests runnable without the full VLMEvalKit dependency set.
"""
import importlib.util
import random
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'vlmeval' / 'dataset' / 'utils' / 'vmcbench.py')
_spec = importlib.util.spec_from_file_location('vmcbench_under_test', MODULE_PATH)
vmcbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vmcbench)

ALL_CHOICES = ['A', 'B', 'C', 'D']
INDEX2ANS = {'A': 'alpha', 'B': 'bravo', 'C': 'charlie', 'D': 'delta'}
UNPARSABLE = 'I am not sure about this one.'


def _parse(response, row_key=None):
    return vmcbench.parse_multi_choice_response(
        response, ALL_CHOICES, INDEX2ANS, row_key=row_key)


def test_fallback_is_identical_across_repeated_parses():
    results = {_parse(UNPARSABLE, row_key='42') for _ in range(200)}
    assert len(results) == 1


def test_fallback_does_not_depend_on_global_random_state():
    random.seed(0)
    first = _parse(UNPARSABLE, row_key='42')
    random.seed(12345)
    for _ in range(10):
        random.random()
    assert _parse(UNPARSABLE, row_key='42') == first


def test_identical_responses_do_not_collapse_without_row_key():
    """The case response-only hashing misses: same string, different rows.

    Hashing the response alone would tie every byte-identical fallback
    response to one option; including the row key keeps the draws
    independent.
    """
    drawn = {_parse(UNPARSABLE, row_key=str(i)) for i in range(64)}
    assert len(drawn) > 1


def test_row_key_changes_the_draw_stably():
    a1 = _parse(UNPARSABLE, row_key='7')
    a2 = _parse(UNPARSABLE, row_key='7')
    b = _parse(UNPARSABLE, row_key='8')
    assert a1 == a2


def test_fallback_still_spreads_over_options():
    drawn = {_parse(f'unparsable response number {i}', row_key=str(i))
             for i in range(200)}
    assert drawn <= set(ALL_CHOICES)
    assert len(drawn) > 1


def test_parseable_responses_are_unaffected():
    assert _parse('The answer is (B).', row_key='1') == 'B'
    assert _parse('A. alpha', row_key='2') == 'A'
    assert _parse('Answer: D. delta', row_key='3') == 'D'
