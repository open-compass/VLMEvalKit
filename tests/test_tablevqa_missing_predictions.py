"""Physical XLSX regressions for supported TableVQA evaluation."""
import ast
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def evaluate(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    name = 'vlmeval.dataset.utils.tablevqabench'
    spec = importlib.util.spec_from_file_location(name, root / 'vlmeval/dataset/utils/tablevqabench.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setitem(sys.modules, name, helper)
    tree = ast.parse((root / 'vlmeval/dataset/image_vqa.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TableVQABench')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'evaluate')
    method.decorator_list = []
    namespace = {
        '__package__': 'vlmeval.dataset', 'load': pd.read_excel,
        'dump': lambda frame, path: frame.to_csv(path, index=False),
        'get_intermediate_file_path': lambda path, suffix, ext: str(Path(path).with_suffix('')) + suffix + '.' + ext,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'image_vqa.py', 'exec'), namespace)
    return namespace['evaluate']


def _rows():
    return [
        {'split': 'fintabnetqa', 'answer': '10 million', 'prediction': 'Answer: 10 million'},
        {'split': 'vtabfact', 'answer': '1', 'prediction': 'Answer: True'},
        {'split': 'vwtq', 'answer': 'Alpha', 'prediction': 'Answer: Alpha'},
        {'split': 'vwtq_syn', 'answer': 'Beta', 'prediction': 'Answer: Beta'},
    ]


@pytest.mark.parametrize('missing_split', ['fintabnetqa', 'vtabfact', 'vwtq', 'vwtq_syn', 'all'])
def test_blank_xlsx_predictions_score_zero_without_aborting(evaluate, tmp_path, missing_split):
    rows = _rows()
    for row in rows:
        if missing_split == 'all' or row['split'] == missing_split:
            row['prediction'] = ''
    path = tmp_path / 'predictions.xlsx'
    pd.DataFrame(rows).to_excel(path, index=False)
    loaded = pd.read_excel(path)
    assert loaded['prediction'].isna().sum() == (4 if missing_split == 'all' else 1)
    result = evaluate(None, str(path))
    for _, row in result.iterrows():
        expected = 0 if missing_split == 'all' or row['split'] == missing_split else 100
        assert row['average_scores'] == pytest.approx([expected] * len(row['average_scores']), abs=1e-5)
    assert (tmp_path / 'predictions_acc.csv').is_file()


def test_valid_prefixed_predictions_keep_full_credit(evaluate, tmp_path):
    path = tmp_path / 'controls.xlsx'
    pd.DataFrame(_rows()).to_excel(path, index=False)
    result = evaluate(None, str(path))
    for scores in result['average_scores']:
        assert scores == pytest.approx([100] * len(scores))
