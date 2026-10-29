import json

import numpy as np
import pandas as pd
import pytest

from vlmeval.dataset import qvhighlights as qvh
from vlmeval.smp import dump


def test_parse_windows():
    assert qvh.parse_windows('[[82, 150], [10, 20]]') == [(82.0, 150.0), (10.0, 20.0)]
    assert qvh.parse_windows('It happens from 20 to 10 seconds.') == [(10.0, 20.0)]
    assert qvh.parse_windows('No matching moment.') == []


def test_parse_highlights():
    answer = ('The highlight timestamps are in the 82, 84, 86 seconds. '
              'Their saliency scores are 1.3, 1.7, 2.3.')
    assert qvh.parse_highlights(answer) == ([82.0, 84.0, 86.0], [1.3, 1.7, 2.3])
    assert qvh.parse_highlights('Highlights at 1:02 and 1:04.') == ([62.0, 64.0], [1.0, 1.0])
    assert qvh.parse_highlights('Nothing stands out.') == ([], [])


def test_highlights_to_clip_scores():
    scores = qvh.highlights_to_clip_scores([0, 1, 3, 149, 150], [1, 3, 2, 4, 5], duration=150)
    assert len(scores) == 75
    assert scores[0] == 2.0 and scores[1] == 2.0 and scores[74] == 4.0
    assert scores[2:74].sum() == 0


def test_window_ap():
    assert np.allclose(qvh.window_ap([(0, 10)], [[0, 10]]), 1.0)
    # The correct window ranked second: precision 1/2 at full recall.
    assert np.allclose(qvh.window_ap([(20, 30), (0, 10)], [[0, 10]]), 0.5)
    # Half of the ground-truth windows found, with precision 1.
    assert np.allclose(qvh.window_ap([(0, 10)], [[0, 10], [20, 30]]), 0.5)
    assert np.allclose(qvh.window_ap([], [[0, 10]]), 0.0)


def test_moment_retrieval_metrics():
    preds = [[(0, 10)], [(0, 6), (20, 30)], []]
    gts = [[[0, 10]], [[0, 10]], [[40, 50]]]
    full = qvh.moment_retrieval_metrics(preds, gts)['full']
    # IoUs of the top-1 window: 1.0, 0.6, 0 (no prediction).
    assert full['R1@0.5'] == pytest.approx(200 / 3)
    assert full['R1@0.7'] == pytest.approx(100 / 3)
    assert full['mAP@0.5'] == pytest.approx(200 / 3)
    assert full['num_queries'] == 3


def test_highlight_metrics():
    gt = np.array([[4, 4, 4], [0, 0, 0], [4, 0, 4]], dtype=float)
    res = qvh.highlight_metrics([np.array([3.0, 2.0, 1.0])], [gt])['VeryGood']
    assert res['HIT@1'] == 100.0
    # Per annotator AP: 5/6, 1, 5/6.
    assert res['mAP'] == pytest.approx(100 * (5 / 6 + 1 + 5 / 6) / 3)


@pytest.mark.parametrize('task', ['MR', 'HD'])
def test_evaluate(tmp_path, task):
    eval_file = tmp_path / f'model_QVHighlights-{task}_2fps.xlsx'
    prediction = ('[[0, 10]]' if task == 'MR' else
                  'The highlight timestamps are in the 0, 2, 4 seconds. Their saliency scores are 3, 2, 1.')
    dump(
        pd.DataFrame({
            'index': [0],
            'video': ['v'],
            'question': ['q'],
            'duration': [150.0],
            'answer': ['[[0, 10]]'],
            'relevant_clip_ids': [json.dumps([0, 1, 2, 3, 4])],
            'saliency_scores': [json.dumps([[4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4], [4, 4, 4]])],
            'prediction': [prediction],
        }), str(eval_file))
    dataset = object.__new__(qvh.QVHighlights)
    dataset.dataset_name = f'QVHighlights-{task}'
    metrics = dataset.evaluate(str(eval_file))
    assert metrics['num_samples'] == 1 and metrics['num_unparsed'] == 0
    if task == 'MR':
        assert metrics['mAP'] == 100.0 and metrics['R1@0.7'] == 100.0
    else:
        assert metrics['HIT@1'] == 100.0
