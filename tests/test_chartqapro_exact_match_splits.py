"""Offline regressions for ChartQAPro's exact-match splits.

`relaxed_correctness_chartqapro` takes an `always_use_exact_match` flag so that
the closed-label splits -- Fact Checking (true/false/unanswerable) and
Multi Choice (a/b/c/d/unanswerable) -- are scored by equality rather than by
ANLS similarity or the 5% numeric tolerance.
"""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def chartqapro():
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/utils/chartqapro.py'
    spec = importlib.util.spec_from_file_location('chartqapro_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(answer, prediction, question_type, year=None):
    return {
        'Answer': [answer],
        'prediction': prediction,
        'Question Type': question_type,
        'Year': year if year is not None else ['NO'],
    }


@pytest.mark.parametrize('question_type, answer, prediction', [
    # A near-miss spelling of the opposite verdict must not earn partial ANLS
    # credit on a true/false split.
    ('Fact Checking', 'true', 'untrue'),
    ('Fact Checking', 'unanswerable', 'Not answerable'),
    ('Multi Choice', 'unanswerable', 'unanswerabel'),
])
def test_closed_label_splits_do_not_earn_partial_credit(chartqapro, question_type, answer, prediction):
    scores = chartqapro.evaluate_predictions_chartqapro([_row(answer, prediction, question_type)])
    assert scores['Overall'] == 0.0
    assert scores[question_type] == 0.0


@pytest.mark.parametrize('question_type, answer, prediction', [
    ('Fact Checking', 'true', 'true'),
    ('Fact Checking', 'true', 'True'),
    ('Fact Checking', 'false', 'false'),
    ('Fact Checking', 'unanswerable', 'unanswerable'),
    ('Multi Choice', 'a', 'a'),
    ('Multi Choice', 'c', 'C'),
    ('Multi Choice', 'unanswerable', 'Unanswerable'),
])
def test_correct_closed_label_answers_still_score_one(chartqapro, question_type, answer, prediction):
    scores = chartqapro.evaluate_predictions_chartqapro([_row(answer, prediction, question_type)])
    assert scores['Overall'] == 1.0


@pytest.mark.parametrize('answer, prediction, expected', [
    # Factoid keeps the 5% relative tolerance and the ANLS text fallback.
    ('100', '102', 1.0),
    ('100', '150', 0.0),
    ('Fairfield', 'Fairfield', 1.0),
])
def test_open_ended_splits_keep_relaxed_scoring(chartqapro, answer, prediction, expected):
    scores = chartqapro.evaluate_predictions_chartqapro([_row(answer, prediction, 'Factoid')])
    assert scores['Overall'] == pytest.approx(expected)


def test_year_rows_remain_exact_match(chartqapro):
    """The pre-existing Year flag must keep forcing exact match on Factoid."""
    scores = chartqapro.evaluate_predictions_chartqapro(
        [_row('2020', '2021', 'Factoid', year=['YES'])]
    )
    assert scores['Overall'] == 0.0
