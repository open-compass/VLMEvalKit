"""Regression tests for the VMCBench option tie-break (#1869).

The parser is a pure function with only numpy/pandas dependencies, so the
module is loaded by path: this keeps the unit test runnable without the full
VLMEvalKit dependency set.
"""
import importlib.util
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


def _parse(response):
    return vmcbench.parse_multi_choice_response(response, ALL_CHOICES,
                                                INDEX2ANS)


def test_restated_options_resolve_to_the_last_one():
    """The case from the issue: every option restated as 'X. ', gold D."""
    response = (
        'The image shows a building with a prominent window. The options '
        'given are: A. Semi-circular B. Oval C. Gothic D. Arched. '
        'The window in the image has a rectangular shape with a slightly '
        'curved top.')

    assert _parse(response) == 'D'


def test_restated_options_are_ordered_by_position_not_alphabet():
    """Two options restated without parentheses must follow the response."""
    response = 'C. Gothic comes before A. Semi-circular in this answer.'

    # A is mentioned last, so it wins even though C comes first alphabetically
    assert _parse(response) == 'A'


def test_mixed_spellings_take_the_later_mention():
    """A parenthesised option followed by a restated one resolves to the latter."""
    response = 'First (B), but on reflection the answer is D. Arched.'

    assert _parse(response) == 'D'


def test_parenthesised_tie_break_is_unchanged():
    """The original '(X)' path keeps working."""
    response = '(A) looks right, though (C) is also plausible.'

    assert _parse(response) == 'C'


@pytest.mark.parametrize('response,expected', [
    ('The answer is (B).', 'B'),
    ('Only C. Gothic applies here.', 'C'),
    ('A', 'A'),
])
def test_single_candidate_cases_are_unchanged(response, expected):
    assert _parse(response) == expected
