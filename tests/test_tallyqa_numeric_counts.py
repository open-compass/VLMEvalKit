"""Offline regressions for TallyQA's single-number count contract."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def extract_count():
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/utils/tallyqa.py'
    spec = importlib.util.spec_from_file_location('tallyqa_counts_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_count_from_prediction


@pytest.mark.parametrize('prediction', ['-2', 'There are -2 objects.', '20.5', '-12', 'Count: -20'])
def test_incorrect_numeric_values_do_not_become_correct_counts(extract_count, prediction):
    assert extract_count(prediction) is None


@pytest.mark.parametrize('prediction, expected', [
    (0, 0),
    (20, 20),
    (2.0, 2),
    (2.9, 2),
    ('2.9', 2),
    ('2', 2),
    ('2.0', 2),
    ('There are 12 objects.', 12),
    ('In 2024, the answer is 2', 2),
    ('2.', 2),
    ('2. There are two objects.', 2),
    ('two', 2),
    ('twenty', 20),
    ('No count available', None),
])
def test_valid_counts_and_existing_prose_extraction(extract_count, prediction, expected):
    assert extract_count(prediction) == expected


def test_native_tallyqa_evaluator_does_not_credit_negative_counts(tmp_path, monkeypatch):
    """Run the shipping evaluator with local CSV/Pandas I/O, without model imports."""
    import ast
    import sys

    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    helper_spec = importlib.util.spec_from_file_location('vlmeval.dataset.utils.tallyqa',
                                                         root / 'vlmeval/dataset/utils/tallyqa.py')
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    monkeypatch.setitem(sys.modules, helper_spec.name, helper)
    tree = ast.parse((root / 'vlmeval/dataset/image_vqa.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TallyQA')
    evaluate = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    namespace = {
        '__package__': 'vlmeval.dataset',
        'load': pd.read_csv,
        'dump': lambda frame, path: frame.to_csv(path, index=False),
    }
    exec(compile(ast.Module(body=[evaluate], type_ignores=[]), 'image_vqa.py', 'exec'), namespace)
    path = tmp_path / 'predictions.csv'
    pd.DataFrame({
        'prediction': ['Count: -2', 'Count: -12', 'Count: 20.5', 'Count: 2'],
        'answer': [2, 12, 20, 2]
    }).to_csv(path, index=False)
    result = namespace['evaluate'](None, str(path))
    assert result.iloc[0]['TallyQA'] == 0.25
    assert pd.read_csv(tmp_path / 'predictions_acc.csv').iloc[0]['TallyQA'] == 0.25
