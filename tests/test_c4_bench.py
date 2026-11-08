import pandas as pd

from vlmeval.dataset import c4_bench
from vlmeval.dataset.c4_bench import C4Bench, normalize_answer, parse_task_answer


def _intermediate_file(path, suffix):
    return f'{path}{suffix}'


def test_c4_direct_answer_parsing():
    assert normalize_answer(' “一叶障目”。 ') == '一叶障目'
    assert parse_task_answer('H0', '一叶障目') == ('一叶障目', None)


def test_c4_explanation_json_and_recovery():
    valid = '```json\n{"answer": "一叶障目", "explanation": "..."}\n```'
    assert parse_task_answer('E0', valid) == ('一叶障目', True)
    assert parse_task_answer('E0', '推理很长，最终答案：一叶障目') == ('一叶障目', False)


def test_c4_primary_and_task_metrics(monkeypatch):
    tasks = ['H0', 'H0', 'H1', 'H1', 'H4', 'H4', 'E0', 'E0']
    indices = [str(index) for index in range(8)]
    metadata = pd.DataFrame({
        'index': indices,
        'task': tasks,
        'answer': ['一叶障目'] * 8,
        'answer_aliases': [[] for _ in range(8)],
    })
    predictions = pd.DataFrame({
        'prediction': [
            '一叶障目',
            '杯弓蛇影',
            '一叶障目',
            '一叶障目',
            '杯弓蛇影',
            '杯弓蛇影',
            '{"answer": "一叶障目"}',
            '最终答案：杯弓蛇影',
        ],
    })
    predictions['index'] = indices
    monkeypatch.setattr(c4_bench, 'load', lambda _path: predictions)
    monkeypatch.setattr(c4_bench, 'dump', lambda _data, _path: None)
    monkeypatch.setattr(c4_bench, 'get_intermediate_file_path', _intermediate_file)

    dataset = C4Bench.__new__(C4Bench)
    dataset.data = metadata
    dataset.dataset_name = 'C4Bench'
    metrics = dataset.evaluate('predictions.xlsx')

    assert metrics == {
        'Overall': 50.0,
        'H0 Exact Match': 50.0,
        'H1 Exact Match': 100.0,
        'H4 Exact Match': 0.0,
        'E0 Exact Match': 50.0,
        'E0 JSON Valid': 50.0,
    }
