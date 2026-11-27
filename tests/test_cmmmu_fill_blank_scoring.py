"""Offline regressions for CMMMU fill-in-the-blank scoring.

The official CMMMU evaluation counts a question as correct at most once. These tests run
``CMMMU.evaluate`` on a tiny prediction file, with the heavy imports stubbed out.
"""
import importlib.util
import os.path as osp
import sys
import types
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

CATEGORIES = ['技术与工程', '科学', '健康与医学', '商业', '艺术与设计', '人文社会科学']


def _get_intermediate_file_path(eval_file, suffix, fmt):
    return osp.splitext(eval_file)[0] + suffix + '.' + fmt


def _load_cmmmu():
    smp = types.ModuleType('vlmeval.smp')
    smp.d2df = lambda d: pd.DataFrame({k: [v] for k, v in d.items()})
    smp.decode_base64_to_image_file = mock.MagicMock()
    smp.dump = lambda data, path: data.to_csv(path, index=False)
    smp.get_intermediate_file_path = _get_intermediate_file_path
    smp.load = lambda path: pd.read_csv(path, sep='\t')
    smp.read_ok = mock.MagicMock()

    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    image_base = types.ModuleType('vlmeval.dataset.image_base')
    image_base.ImageBaseDataset = type('ImageBaseDataset', (), {})

    modules = {
        'vlmeval': package,
        'vlmeval.smp': smp,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.image_base': image_base,
    }
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/cmmmu.py'
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.cmmmu', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _evaluate(tmp_path, rows):
    """rows: one (answer, prediction) fill-in-the-blank pair per category."""
    data = pd.DataFrame([{
        'index': i,
        'category': category,
        'type': '填空',
        'answer': answer,
        'prediction': prediction,
    } for i, (category, (answer, prediction)) in enumerate(zip(CATEGORIES, rows))])
    eval_file = tmp_path / 'preds.tsv'
    data.to_csv(eval_file, sep='\t', index=False)
    result = _load_cmmmu().CMMMU.evaluate(str(eval_file))
    return result.iloc[0].to_dict()


CORRECT = ('苹果', '答案是苹果')
WRONG = ('上海', '答案是北京')
# Two different sentences each end in the right answer ("北京市" and "北京" both contain "北京").
REPEATED = ('北京', '所以答案是北京市。\n因此最终答案是北京。')


def test_answer_found_in_several_candidates_counts_once(tmp_path):
    scores = _evaluate(tmp_path, [REPEATED, CORRECT, CORRECT, CORRECT, CORRECT, CORRECT])

    assert scores['技术与工程'] == pytest.approx(1.0)
    assert scores['总准确率'] == pytest.approx(1.0)


def test_repeated_answer_does_not_hide_wrong_answers(tmp_path):
    scores = _evaluate(tmp_path, [REPEATED, CORRECT, WRONG, WRONG, CORRECT, CORRECT])

    assert scores['技术与工程'] == pytest.approx(1.0)
    assert scores['健康与医学'] == pytest.approx(0.0)
    assert scores['总准确率'] == pytest.approx(4 / 6)


def test_single_match_and_wrong_answers_are_unchanged(tmp_path):
    scores = _evaluate(tmp_path, [CORRECT, WRONG, CORRECT, WRONG, CORRECT, WRONG])

    assert scores['科学'] == pytest.approx(0.0)
    assert scores['总准确率'] == pytest.approx(0.5)


def test_numeric_answers_are_still_matched(tmp_path):
    numeric = ('3.5', '所以答案是3.5')
    scores = _evaluate(tmp_path, [numeric] * 6)

    assert scores['总准确率'] == pytest.approx(1.0)
