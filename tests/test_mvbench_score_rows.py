import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.mark.parametrize('class_name', ['MVBench', 'MVBench_MP4'])
@pytest.mark.parametrize('indices', [[10, 20], [1, 0], [0, 1]])
def test_scores_follow_dataset_ids_without_creating_or_swapping_rows(class_name, indices):
    source = Path('vlmeval/dataset/mvbench.py').read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    module = ast.Module(body=[method], type_ignores=[])
    data = pd.DataFrame({
        'index': indices,
        'answer': ['yes', 'yes'],
        'prediction': ['correct', 'FAIL'],
        'candidates': ["['yes', 'no']"] * 2,
    })
    artifacts = {}

    def dump(value, filename):
        artifacts[filename] = value.copy() if hasattr(value, 'copy') else value

    def rating(filename):
        scored = artifacts[filename]
        assert scored['index'].tolist() == indices
        assert scored['score'].tolist() == [1, -1]
        assert scored['answer'].tolist() == ['yes', 'yes']
        return {'row_count': len(scored)}

    namespace = {
        'pd': pd,
        'osp': SimpleNamespace(exists=lambda _: False),
        'get_file_extension': lambda _: 'json',
        'get_intermediate_file_path': lambda name, suffix, *args: name + suffix,
        'load': lambda _: data.copy(),
        'dump': dump,
        'get_dimension_rating': rating,
        'FAIL_MSG': 'FAIL',
        'check_ans_with_model': lambda pred, *args: pred == 'correct',
    }
    exec(compile(module, 'mvbench.py', 'exec'), namespace)
    result = namespace['evaluate'].__func__(
        SimpleNamespace(DEFAULT_JUDGE_MODEL='exact_matching'), 'predictions.json')
    assert result == {'row_count': 2}
