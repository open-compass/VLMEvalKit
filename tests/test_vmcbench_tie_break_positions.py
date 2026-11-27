"""Regression tests for the VMCBench option tie-break (#1869).

The parser module only needs numpy, so it is loaded by path to keep the unit
tests runnable without the full VLMEvalKit dependency set.
"""
import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'vlmeval' / 'dataset' / 'utils' / 'vmcbench.py')
_spec = importlib.util.spec_from_file_location('vmcbench_under_test', MODULE_PATH)
vmcbench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vmcbench)

ALL_CHOICES = ['A', 'B', 'C', 'D']
INDEX2ANS = {
    'A': 'Semi-circular',
    'B': 'Oval',
    'C': 'Gothic',
    'D': 'Arched',
}


def _parse(response):
    return vmcbench.parse_multi_choice_response(response, ALL_CHOICES, INDEX2ANS)


def test_restated_options_resolve_by_position():
    """The example from the issue: gold D, every option restated as 'X. '."""
    response = (
        'The image shows a building with a prominent window. The options '
        'given are: A. Semi-circular B. Oval C. Gothic D. Arched '
        'The window in the image has a rectangular shape with a slightly '
        'curved top.')
    assert _parse(response) == 'D'


def test_order_follows_the_response_not_the_alphabet():
    response = 'C. Gothic comes before A. Semi-circular in this answer.'
    assert _parse(response) == 'A'


def test_mixed_spellings_take_the_later_mention():
    response = 'First (B), but on reflection the answer is D. Arched.'
    assert _parse(response) == 'D'


def test_parenthesised_tie_break_is_unchanged():
    response = '(A) looks right, though (C) is also plausible.'
    assert _parse(response) == 'C'


def test_single_candidate_cases_are_unchanged():
    assert _parse('The answer is (B).') == 'B'
    assert _parse('Only C. Gothic applies here.') == 'C'
