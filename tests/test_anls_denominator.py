"""Offline regressions for the ANLS denominator on DocVQA / InfoVQA.

anls_compute normalised both operands (strip, lowercase, collapse runs of
whitespace) before measuring the Levenshtein distance, but then divided by the
length of the *raw* strings:

    dist   = levenshtein_distance(gt_answer, det_answer)   # normalised
    length = max(len(groundtruth.upper()), len(prediction.upper()))   # raw

ANLS is defined as edit_distance(a, b) / max(|a|, |b|) over the same pair of
strings used for the distance, so mixing the two inflates the score. On short
answers it crosses the 0.5 threshold and credits a wrong answer.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def vqa_eval():
    """Load vqa_eval with the real smp helpers it depends on."""
    spec = importlib.util.spec_from_file_location(
        'anls_test_smp_misc', REPO / 'vlmeval/smp/misc.py'
    )
    misc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(misc)

    pkg = sys.modules.setdefault('vlmeval', types.ModuleType('vlmeval'))
    pkg.__path__ = []
    smp = types.ModuleType('vlmeval.smp')
    smp.process_punctuation = misc.process_punctuation
    smp.istype = misc.istype
    smp.listinstr = misc.listinstr
    sys.modules['vlmeval.smp'] = smp

    spec = importlib.util.spec_from_file_location(
        'vqa_eval_under_test', REPO / 'vlmeval/dataset/utils/vqa_eval.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(vqa_eval, answer, prediction, dataset='DocVQA_VAL'):
    """Mirror ImageVQADataset.evaluate_heuristic for one row."""
    line = pd.Series({'answer': str(answer), 'prediction': str(prediction)})
    res = vqa_eval.process_line(line, method='anls')
    return vqa_eval.hit_calculate([res], dataset)[0]


@pytest.mark.parametrize('dataset', ['DocVQA_VAL', 'InfoVQA_VAL'])
def test_short_wrong_answer_with_trailing_space_scores_zero(vqa_eval, dataset):
    """gt '2' vs prediction '3 ' must not be credited.

    The raw denominator is max(len('2'), len('3 ')) == 2, so the normalised
    distance of 1 becomes 0.5 and `1 - 0.5` is not < 0.5 -- half a point for a
    wrong digit. With the normalised denominator it is 1/1 == 1.0 -> 0.0.
    """
    assert _score(vqa_eval, '2', '3 ', dataset) == 0.0


@pytest.mark.parametrize('answer, prediction', [
    ('2', '3 '),
    ('2', ' 3'),
    ('5', '  7  '),
    ('No', 'Yes '),
])
def test_wrong_short_answers_are_never_credited(vqa_eval, answer, prediction):
    assert _score(vqa_eval, answer, prediction) == 0.0


def test_whitespace_padding_does_not_inflate_a_near_miss(vqa_eval):
    """A padded prediction must score the same as its unpadded form."""
    padded = _score(vqa_eval, 'Fairfield', '  fairfeild  ')
    bare = _score(vqa_eval, 'Fairfield', 'fairfeild')
    assert padded == bare == pytest.approx(1 - 2 / 9)


def test_internal_whitespace_runs_do_not_inflate(vqa_eval):
    collapsed = _score(vqa_eval, '17 Oct 2002', '17 Oct 2001')
    spaced = _score(vqa_eval, '17 Oct 2002', '17  Oct  2001')
    assert spaced == collapsed


@pytest.mark.parametrize('answer, prediction', [
    # Whitespace and case are the only difference: still a perfect match.
    ('Fairfield', 'Fairfield'),
    ('Fairfield', '  Fairfield  '),
    ('Total', '  total'),
    ('2', '2'),
    ('17 Oct 2002', '17  Oct  2002'),
])
def test_exact_matches_still_score_one(vqa_eval, answer, prediction):
    assert _score(vqa_eval, answer, prediction) == 1.0


@pytest.mark.parametrize('answer, prediction', [
    ('2', '3'),
    ('No', 'Yes'),
    ('Fairfield', 'Manchester'),
])
def test_clearly_wrong_answers_stay_zero(vqa_eval, answer, prediction):
    assert _score(vqa_eval, answer, prediction) == 0.0


def test_anls_stays_a_soft_metric(vqa_eval):
    """ANLS deliberately credits a close miss; the fix must not harden it.

    'Fairfield' vs 'Springfield' is edit distance 5 over max length 11, so it
    scores 1 - 5/11 both before and after. Only the operands of the
    denominator change, not the metric.
    """
    assert _score(vqa_eval, 'Fairfield', 'Springfield') == pytest.approx(1 - 5 / 11)


def test_multi_answer_ground_truth_takes_the_best(vqa_eval):
    """hit_calculate uses min(distance), i.e. the closest of the gt variants."""
    line = pd.Series({'answer': "['Fairfield', 'Springfield']", 'prediction': 'Springfield'})
    res = vqa_eval.process_line(line, method='anls')
    assert vqa_eval.hit_calculate([res], 'DocVQA_VAL')[0] == 1.0
