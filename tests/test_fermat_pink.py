from io import BytesIO
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from datasets import Dataset, Image
from PIL import Image as PILImage

pink = pytest.importorskip('pink')

from pink.rubric import SCORE_COLUMNS  # noqa: E402

from vlmeval.dataset import DATASET_MODALITY, DATASET_TYPE, build_dataset  # noqa: E402
from vlmeval.dataset.fermat_pink import FERMAT_PINK, OCR_SYSTEM_PROMPT  # noqa: E402
from vlmeval.smp import dump  # noqa: E402


def _png(size=(1200, 1600)):
    stream = BytesIO()
    PILImage.new('RGB', size, 'white').save(stream, format='PNG')
    return stream.getvalue()


def _source(n=3):
    records = [{
        'new_custom_id': f'img_{i}_pert_1.1',
        'image': {
            'bytes': _png(),
            'path': None
        },
        'orig_q': f'Question {i}\r\n',
        'pert_a': f'2 \\times 3 = \\textcolor{{red}}{{\\frac{{1{i}}}{{2}}}}',
        'domain_code': 'alg',
    } for i in range(n)]
    return Dataset.from_list(records).cast_column('image', Image(decode=False))


def test_load_keeps_original_images_and_builds_oracle(tmp_path, monkeypatch):
    monkeypatch.setenv('LMUData', str(tmp_path))
    with mock.patch('datasets.load_dataset', return_value=_source()) as loader:
        dataset = build_dataset('FERMAT_PINK')
    loader.assert_called_once_with('ai4bharat/FERMAT', split='train')
    assert DATASET_TYPE('FERMAT_PINK') == 'VQA'
    assert DATASET_MODALITY('FERMAT_PINK') == 'IMAGE'
    row = dataset.data.iloc[0]
    assert row['sample_id'] == 'img_0_pert_1.1'
    assert row['answer'] == '2 \\times 3 = \\frac{10}{2}'  # red error marker removed, braces kept
    assert row['problem'] == 'Question 0\n'
    with PILImage.open(row['image_path']) as image:
        assert image.size == (1200, 1600)
    prompt = dataset.build_prompt(0)
    assert prompt[0]['type'] == 'image' and prompt[1]['value'].startswith(OCR_SYSTEM_PROMPT)


def test_judge_config_maps_vlmevalkit_arguments(monkeypatch):
    monkeypatch.delenv('PINK_ORACLE_GRADES', raising=False)
    name, cfg, oracle = FERMAT_PINK.judge_config({})
    assert (name, cfg.model, oracle) == ('gpt-5', 'gpt-5-2025-08-07', 'fermat:gpt-5')
    name, cfg, oracle = FERMAT_PINK.judge_config({
        'model':
        'qwen3-next-80b',
        'api_base':
        'http://localhost:8000/v1/chat/completions'
    })
    assert cfg.base_url == 'http://localhost:8000/v1' and oracle == 'fermat:qwen3-next-80b'
    name, cfg, oracle = FERMAT_PINK.judge_config({'model': 'my-judge', 'key': 'k'})
    assert cfg.model == 'my-judge' and cfg.api_key == 'k' and oracle is None


def test_evaluate_penalizes_over_correction_and_caches(tmp_path, monkeypatch):
    oracle = tmp_path / 'oracle.csv'
    pd.DataFrame([
        {
            'sample_id': 's1',
            'success': True,
            **dict(zip(SCORE_COLUMNS, [20, 20, 20, 5, 5]))
        },
        {
            'sample_id': 's2',
            'success': True,
            **dict(zip(SCORE_COLUMNS, [10, 10, 10, 10, 10]))
        },
        {
            'sample_id': 's3',
            'success': True,
            **dict(zip(SCORE_COLUMNS, [10, 10, 10, 10, 10]))
        },
    ]).to_csv(
        oracle, index=False)
    monkeypatch.setenv('PINK_ORACLE_GRADES', str(oracle))
    calls = []

    def fake_grade(client, cfg, problem, solution, sample_id=''):
        calls.append(sample_id)
        scores = [20] * 5 if sample_id == 's1' else [10] * 5
        return {'sample_id': sample_id, 'success': True, **dict(zip(SCORE_COLUMNS, scores))}

    monkeypatch.setattr(pink.judge, 'grade', fake_grade)
    monkeypatch.setattr(pink.judge, 'make_client', lambda cfg: object())
    eval_file = tmp_path / 'model_FERMAT_PINK.xlsx'
    dump(
        pd.DataFrame({
            'index': [0, 1, 2],
            'sample_id': ['s1', 's2', 's3'],
            'problem': ['p'] * 3,
            'answer': ['a'] * 3,
            'prediction': ['**Answer:** x', '**Answer:** y', None],
        }), str(eval_file))
    dataset = object.__new__(FERMAT_PINK)
    table = dataset.evaluate(str(eval_file), model='qwen3-next-80b', nproc=2)
    # s1: two components +15 -> nullified (60 vs oracle 70); s2 faithful (50); s3 empty -> 0.
    assert table.loc[0, 'PINK'] == pytest.approx((60 + 50 + 0) / (70 + 50 + 50))
    assert table.loc[0, 'over_correction_rate'] == pytest.approx(1 / 3)
    assert sorted(calls) == ['s1', 's2']
    dataset.evaluate(str(eval_file), model='qwen3-next-80b', nproc=2)
    assert sorted(calls) == ['s1', 's2']  # second run reuses the cached judge results
    assert Path(str(eval_file).replace('.xlsx', '_score.csv')).exists()
    assert all(pd.api.types.is_numeric_dtype(table[c]) for c in table.columns)
    assert FERMAT_PINK.report_primary_metric({
        'raw_score_ratio': 0.9,
        'PINK': 0.5
    }) == {
        'PINK': 0.5
    }
