"""US-007 — Tier 0.5 rule-based evaluation lane.

Exercises the post-inference scoring path with NO LLM judge: build a tiny
prediction table and run a rule-based scorer end-to-end. ChartQA's
``evaluate_heuristic`` uses ``relaxed_accuracy`` (numeric tolerance /
exact string match) and never calls ``build_judge``, so it is the
cheapest honest check that the evaluate half of the harness works.
"""

from __future__ import annotations
import os

import pytest

from vlmeval.smp import dump

pytestmark = pytest.mark.skipif(
    not os.path.isfile(
        os.path.join(os.environ.get('LMUData', ''), 'ChartQA_TEST.tsv')
    ),
    reason='ChartQA_TEST.tsv not cached under LMUData',
)


def _tiny_eval_file(tmp_path, *, perfect: bool):
    from vlmeval.dataset import build_dataset

    dataset = build_dataset('ChartQA_TEST')
    data = dataset.data.iloc[:5].copy()
    if perfect:
        # Predictions equal to gold answers -> relaxed_accuracy should be 100.
        data['prediction'] = [str(a) for a in data['answer']]
    else:
        data['prediction'] = ['definitely-wrong-xyz'] * len(data)
    eval_file = str(tmp_path / 'ChartQA_TEST.xlsx')
    dump(data, eval_file)
    return dataset, eval_file


def test_chartqa_perfect_predictions_score_100(tmp_path):
    dataset, eval_file = _tiny_eval_file(tmp_path, perfect=True)
    # No judge_kwargs -> evaluate() routes to evaluate_heuristic (rule-based).
    ret = dataset.evaluate(eval_file)
    overall = float(ret['Overall'].iloc[0])
    assert overall == pytest.approx(100.0), f'expected 100, got {overall}'


def test_chartqa_wrong_predictions_score_low(tmp_path):
    dataset, eval_file = _tiny_eval_file(tmp_path, perfect=False)
    ret = dataset.evaluate(eval_file)
    overall = float(ret['Overall'].iloc[0])
    # Garbage predictions must not score full marks; this proves the scorer
    # actually discriminates rather than rubber-stamping.
    assert overall < 100.0, f'wrong predictions scored {overall}'


def test_chartqa_evaluate_uses_no_judge(tmp_path, monkeypatch):
    """Guard: the rule-based path must not instantiate an LLM judge."""
    import vlmeval.dataset.image_vqa as image_vqa

    def _boom(*a, **k):
        raise AssertionError('build_judge must not be called on the heuristic path')

    monkeypatch.setattr(image_vqa, 'build_judge', _boom)
    dataset, eval_file = _tiny_eval_file(tmp_path, perfect=True)
    ret = dataset.evaluate(eval_file)
    assert 'Overall' in ret
