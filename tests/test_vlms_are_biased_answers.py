"""Offline regressions for VLMs Are Biased answer parsing (no model imports)."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(scope='module')
def process_results():
    smp = types.ModuleType('vlmeval.smp')
    smp.load = lambda data: data
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/utils/vlmsarebiased.py'
    with mock.patch.dict(sys.modules, {'vlmeval': types.ModuleType('vlmeval'), 'vlmeval.smp': smp}):
        spec = importlib.util.spec_from_file_location('vlmsarebiased_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.vlms_are_biased_process_results


def score(process_results, prediction, ground_truth, expected_bias):
    row = {
        'prediction': prediction,
        'ground_truth': ground_truth,
        'expected_bias': expected_bias,
        'topic': 'Animals',
    }
    result = process_results([row])[0]
    return result['is_correct'], result['matches_bias']


@pytest.mark.parametrize('prediction, ground_truth, expected_bias, expected', [
    # The prompts ask for the answer in curly brackets; the bracketed answer is what counts.
    ('Counting each leg: 1, 2, 3, 4, 5. So the answer is {5}.', '5', '4', (True, False)),
    ('A typical dog has 4 legs, but this one has {5}', '5', '4', (True, False)),
    ('The answer is {No}.', 'No', 'Yes', (True, False)),
    ('{3} stripes, not 13', '3', '13', (True, False)),
    ('Dogs have 4 legs, so {4}.', '5', '4', (False, True)),
])
def test_bracketed_answer_in_longer_response(process_results, prediction, ground_truth, expected_bias,
                                             expected):
    assert score(process_results, prediction, ground_truth, expected_bias) == expected


@pytest.mark.parametrize('prediction, ground_truth, expected_bias, expected', [
    ('{5}', '5', '4', (True, False)),
    ('{4}', '5', '4', (False, True)),
    ('5', '5', '4', (True, False)),
    ('yes', 'Yes', 'No', (True, False)),
])
def test_bare_answers_still_score(process_results, prediction, ground_truth, expected_bias, expected):
    assert score(process_results, prediction, ground_truth, expected_bias) == expected
