"""Offline regressions for CountBenchQA count matching (no model imports)."""
import ast
import re
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope='module')
def evaluate():
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/image_vqa.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CountBenchQA')
    func = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    namespace = {
        're': re,
        'load': lambda data: data,
        'dump': lambda data, path: None,
        'd2df': lambda data: data,
        'get_intermediate_file_path': lambda path, suffix: path,
    }
    exec(compile(ast.Module(body=[func], type_ignores=[]), 'image_vqa.py', 'exec'), namespace)
    return namespace['evaluate']


def accuracy(evaluate, predictions, answers):
    data = pd.DataFrame({'index': range(len(predictions)), 'prediction': predictions, 'answer': answers})
    return evaluate(None, data)['accuracy']


@pytest.mark.parametrize('prediction, answer', [
    # The answer only appears inside a larger number.
    ('12', 2),
    ('20', 2),
    ('100', 10),
    ('There are 12 dogs.', 2),
])
def test_count_inside_a_larger_number_is_wrong(evaluate, prediction, answer):
    assert accuracy(evaluate, [prediction], [answer]) == 0


@pytest.mark.parametrize('prediction, answer', [
    ('2', 2),
    ('2.', 2),
    ('There are 2 dogs.', 2),
    ('10', 10),
])
def test_count_as_a_whole_number_is_correct(evaluate, prediction, answer):
    assert accuracy(evaluate, [prediction], [answer]) == 100


def test_accuracy_over_several_samples(evaluate):
    assert accuracy(evaluate, ['2', '12', '10', '100'], [2, 2, 10, 10]) == 50
