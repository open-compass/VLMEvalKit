import ast
import re
from pathlib import Path

import pandas as pd
import pytest


def load_rule_scorer():
    root = Path(__file__).resolve().parents[1]
    matcher = root / 'vlmeval/dataset/utils/spatial_bench/matching_func.py'
    nodes = []
    for node in ast.parse(matcher.read_text()).body:
        if isinstance(node, ast.Assign):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == 'can_match_option':
            nodes.append(node)
            break
    namespace = {'re': re, 'ast': ast}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(matcher), 'exec'), namespace)
    scores = root / 'vlmeval/dataset/utils/spatial_bench/cal_scores.py'
    nodes = [node for node in ast.parse(scores.read_text()).body
             if isinstance(node, ast.FunctionDef)
             and node.name in ('exact_match', 'compute_mcq_score')]
    namespace['pd'] = pd
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(scores), 'exec'), namespace)
    return namespace['compute_mcq_score']


@pytest.mark.parametrize('prediction,gold', [
    ('<answer>cat</answer>', 'C'),
    ('<answer>banana</answer>', 'B'),
    ('<answer>apple</answer>', 'A'),
])
def test_tagged_words_do_not_score_as_their_first_letter(prediction, gold):
    result = load_rule_scorer()(pd.DataFrame([{'prediction': prediction, 'answer': gold}]))
    assert result.iloc[0]['hit'] == 0.0


@pytest.mark.parametrize('prediction,gold,expected', [
    ('<answer>C</answer>', 'C', 1.0),
    ('<answer>C. cat</answer>', 'C', 1.0),
    ('<answer>c</answer>', 'A', 0.0),
    ('<think>A might work</think><answer>D</answer>', 'D', 1.0),
])
def test_tagged_labels_and_explanations_preserve_scoring(prediction, gold, expected):
    result = load_rule_scorer()(pd.DataFrame([{'prediction': prediction, 'answer': gold}]))
    assert result.iloc[0]['hit'] == expected
