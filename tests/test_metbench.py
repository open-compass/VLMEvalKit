"""MET-Bench's native dataset, prompt ordering, and result-file contracts."""

import io

import pandas as pd
import pytest
from datasets import Dataset
from PIL import Image

from vlmeval.dataset import SUPPORTED_DATASETS, build_dataset
from vlmeval.dataset.metbench import EVALUATION_RELEASES, METBenchImage, METBenchText
from vlmeval.smp import dump
from vlmeval.smp.file import INFER_FAIL_MSG

FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


def image(index):
    """Encode distinguishable frames to detect reordering and action-frame leakage."""
    buffer = io.BytesIO()
    Image.new('RGB', (2, 2), (index, 0, 0)).save(buffer, format='PNG')
    return {'bytes': buffer.getvalue(), 'path': None}


def rows(domain, modality):
    """Make a full-size synthetic split with sparse source indices."""
    if domain == 'minecraft':
        row = dict(initial_state='{"x":0}', action='Walk forward 1 block.',
                   candidate_states=['{"x":1}', '{"x":2}', '{"x":3}', '{"x":4}'], correct_choice=2)
        if modality == 'image':
            row.update(image_initial_state=image(0), image_action=image(99),
                       image_candidate_states=[image(i) for i in range(1, 5)])
    else:
        row = dict(initial_state=FEN if domain == 'chess' else 1,
                   final_state=FEN if domain == 'chess' else 2,
                   actions=['g1f3' if domain == 'chess' else '1 swap 2'] * 10)
        if modality == 'image':
            row['image_actions'] = [image(i) for i in range(10)]
    return Dataset.from_list([dict(row, example_id=f'{domain}-test-{3 * i}') for i in range(500)])


@pytest.mark.parametrize('domain', ['minecraft', 'chess', 'shell'])
@pytest.mark.parametrize('modality', ['text', 'image'])
def test_native_dataset_contract(monkeypatch, tmp_path, domain, modality):
    """Check discovery, pinned downloads, prompts, and scoring through native APIs."""
    calls = []

    def load(repo, config, **kwargs):
        calls.append((repo, config, kwargs))
        return rows(domain, modality)

    monkeypatch.setattr('vlmeval.dataset.metbench.load_dataset', load)
    monkeypatch.setattr('vlmeval.dataset.metbench.LMUDataRoot', lambda: str(tmp_path))
    name = f'METBench_{domain}_{modality}'
    assert name in SUPPORTED_DATASETS
    registered = build_dataset(name)
    assert len(registered.data) == 500
    cls = METBenchText if modality == 'text' else METBenchImage
    task = cls(name, limit=2)
    repo, revision = EVALUATION_RELEASES[domain]
    assert calls[-1] == (repo, 'evaluation_text_only' if modality == 'text' else 'evaluation',
                         {'split': 'test', 'revision': revision})
    assert task.data['index'].tolist() == [0, 3]
    prompt = task.build_prompt(task.data.iloc[1])
    images = [part['value'] for part in prompt if part['type'] == 'image']
    expected = list(range(5 if domain == 'minecraft' else 10)) if modality == 'image' else []
    assert [Image.open(path).getpixel((0, 0))[0] for path in images] == expected
    assert task.dump_image(task.data.iloc[1]) == images
    if modality == 'text':
        assert len(prompt) == 1
    original_text = [part['value'] for part in prompt if part['type'] == 'text']
    task.by_index['3']['target'] = 'PRIVATE_TARGET'
    assert [p['value'] for p in task.build_prompt(task.data.iloc[1]) if p['type'] == 'text'] == original_text
    task.by_index['3']['target'] = task.examples[0]['target']
    path = str(tmp_path / 'predictions.xlsx')
    predictions = task.data.copy()
    predictions['prediction'] = [f'FINAL ANSWER: {task.examples[0]["target"]}', 'unparseable']
    dump(predictions.iloc[::-1], path)
    result = task.evaluate(path).iloc[0]
    assert result['Overall'] == 50.0
    assert result['ci_lower'] < 50 < result['ci_upper']
    assert result['examples'] == 2
    predictions['prediction'] = [None, 'unparseable']
    dump(predictions, path)
    assert task.evaluate(path).iloc[0]['Overall'] == 0
    predictions['prediction'] = INFER_FAIL_MSG
    dump(predictions, path)
    with pytest.raises(ValueError, match='failed inference'):
        task.evaluate(path)
    dump(pd.concat([predictions.iloc[:1]] * 2), path)
    with pytest.raises(ValueError, match='duplicate'):
        task.evaluate(path)
    dump(predictions.iloc[:1], path)
    with pytest.raises(ValueError, match='selected examples'):
        task.evaluate(path)


def test_chess_confidence_uses_whole_trials(monkeypatch, tmp_path):
    """Correlated squares cannot inflate the effective example count."""
    import math

    monkeypatch.setattr('vlmeval.dataset.metbench.load_dataset', lambda *a, **k: rows('chess', 'text'))
    monkeypatch.setattr('vlmeval.dataset.metbench.LMUDataRoot', lambda: str(tmp_path))
    task = METBenchText('METBench_chess_text', limit=100)
    predictions = task.data.copy()
    predictions['prediction'] = [f'FINAL ANSWER: {FEN}', 'unparseable'] * 50
    path = str(tmp_path / 'chess.xlsx')
    dump(predictions, path)
    result = task.evaluate(path).iloc[0]
    margin = 100 * 1.959963984540054 * math.sqrt(0.25 / 99)
    assert result['Overall'] == 50.0
    assert result['ci_lower'] == pytest.approx(50 - margin)
    assert result['ci_upper'] == pytest.approx(50 + margin)
    task = METBenchText('METBench_chess_text', limit=1)
    dump(predictions.iloc[:1], path)
    result = task.evaluate(path).iloc[0]
    assert pd.isna(result['ci_lower']) and pd.isna(result['ci_upper'])
