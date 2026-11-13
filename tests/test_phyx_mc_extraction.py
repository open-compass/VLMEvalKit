"""Offline regressions for PhyX multiple-choice answer extraction.

PhyX_process_line_MC used `re.search(r'\\b(?:correct|answer|option|...)\\b[\\s\\S]*?([A-D])')`
-- the FIRST trigger word, a lazy gap and an unanchored letter -- so it
captured the first A-D character anywhere after the first trigger. On a
rationale that enumerates the options before concluding ("...each answer
choice. A) 10 N ... the correct option is D.") that is the "A" of the
enumeration, not the answer. A bare-letter prediction was not extracted at all.

The contract is stated by the repo's own in-context examples in get_ICE_MC:
`Predicted answer: A` -> judgement 1, and
`Predicted answer: The correct option is D.` -> judgement 1.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def phyx():
    pkg = sys.modules.setdefault('vlmeval', types.ModuleType('vlmeval'))
    pkg.__path__ = []
    smp = types.ModuleType('vlmeval.smp')
    smp.load = lambda *a, **k: None       # only used by PhyX_acc
    sys.modules['vlmeval.smp'] = smp

    spec = importlib.util.spec_from_file_location(
        'phyx_under_test', REPO / 'vlmeval/dataset/utils/phyx.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(phyx, answer, prediction):
    line = pd.Series({'index': 0, 'answer': answer, 'prediction': prediction})
    return phyx.PhyX_process_line_MC(line)


ENUMERATING_RATIONALES = [
    ('D', 'Let me check each answer choice. A) 10 N assumes the pulley is frictionless. '
          'Therefore the correct option is D.'),
    ('C', 'Option A is 5 m/s, option B is 7 m/s. The correct answer is C.'),
    ('B', 'Evaluating each option: A) 2 J, B) 4 J, C) 6 J, D) 8 J. The correct answer is B.'),
    ('D', 'A is wrong because the field cancels. The correct option is D.'),
]


@pytest.mark.parametrize('answer, prediction', ENUMERATING_RATIONALES)
def test_enumerated_options_do_not_shadow_the_conclusion(phyx, answer, prediction):
    ret = _row(phyx, answer, prediction)
    assert ret['extracted'] == answer
    assert ret['match'] == 1


@pytest.mark.parametrize('answer', ['A', 'B', 'C', 'D'])
def test_bare_option_letter_is_extracted(phyx, answer):
    """get_ICE_MC example_1 pairs a predicted answer of 'A' with judgement 1."""
    ret = _row(phyx, answer, answer)
    assert ret['extracted'] == answer
    assert ret['match'] == 1


@pytest.mark.parametrize('answer, prediction', [
    ('A', 'The correct answer is A.'),
    ('B', 'The correct answer is B.'),
    ('C', 'The correct option is C.'),
    ('D', 'Answer: D'),
    ('C', 'C: 12 m/s'),
    ('B', 'Comparing all four, the answer must be B, since A and C violate conservation.'),
])
def test_shapes_that_already_worked_keep_working(phyx, answer, prediction):
    ret = _row(phyx, answer, prediction)
    assert ret['match'] == 1


@pytest.mark.parametrize('answer, prediction', [
    # A genuinely wrong conclusion must stay wrong.
    ('D', 'Option A is 5 m/s, option B is 7 m/s. The correct answer is C.'),
    ('A', 'The correct answer is B.'),
    ('C', 'A'),
    ('B', 'Evaluating each option: A) 2 J, B) 4 J. The correct answer is D.'),
])
def test_wrong_answers_stay_wrong(phyx, answer, prediction):
    assert _row(phyx, answer, prediction)['match'] == 0


def test_api_failure_is_still_reported(phyx):
    ret = _row(phyx, 'A', phyx.FAIL_MSG)
    assert ret['extracted'] == 'Fail to Call API'
    assert ret['match'] == 0


def test_response_without_any_letter_is_not_matched(phyx):
    ret = _row(phyx, 'A', 'I cannot determine the answer from this diagram.')
    assert ret['match'] == 0
