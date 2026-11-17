"""Regression tests for the VMCBench multiple-choice fallback (#1867).

The parser is a pure function with only numpy/pandas dependencies, so the
module is loaded by path: this keeps the unit test runnable without the full
VLMEvalKit dependency set.
"""
import importlib.util
import random
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'vlmeval' / 'dataset' / 'utils' / 'vmcbench.py')
_SPEC = importlib.util.spec_from_file_location('vmcbench_under_test',
                                               MODULE_PATH)
vmcbench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vmcbench)

ALL_CHOICES = ['A', 'B', 'C', 'D']
INDEX2ANS = {
    'A': 'Semi-circular',
    'B': 'Oval',
    'C': 'Gothic',
    'D': 'Arched',
}
UNPARSABLE = 'I am not sure about this one.'


def _parse(response):
    return vmcbench.parse_multi_choice_response(response, ALL_CHOICES,
                                                INDEX2ANS)


def test_fallback_is_identical_across_repeated_parses():
    """Rescoring one predictions file must give one answer, not a spread."""
    results = {_parse(UNPARSABLE) for _ in range(200)}

    assert results == {_parse(UNPARSABLE)}
    assert len(results) == 1


def test_fallback_does_not_depend_on_global_random_state():
    """The global RNG used to leak into the score; it must not any more."""
    random.seed(0)
    first = _parse(UNPARSABLE)

    random.seed(12345)
    for _ in range(10):
        random.random()
    second = _parse(UNPARSABLE)

    assert first == second


def test_fallback_still_spreads_over_the_options():
    """A fixed seed would collapse every guess onto one option."""
    results = {_parse(f'unparsable response number {index}')
               for index in range(200)}

    assert results <= set(ALL_CHOICES)
    assert len(results) > 1


@pytest.mark.parametrize('response,expected', [
    ('The answer is (B).', 'B'),
    ('A. Semi-circular', 'A'),
    ('Answer: D. Arched', 'D'),
    ('The answer is (C)', 'C'),
])
def test_parseable_responses_are_unaffected(response, expected):
    assert _parse(response) == expected
