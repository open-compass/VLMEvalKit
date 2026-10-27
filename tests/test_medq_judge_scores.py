"""MedQ's documented 0–2 judge scale applies to every response encoding."""
import ast
import re
import time
from pathlib import Path

import pytest


@pytest.fixture(params=[
    ('medqbench_caption.py', 'MedQBench_Caption_Scorer'),
    ('medqbench_paired_description.py', 'MedQBench_PairedDescription_Scorer'),
])
def scorer(request):
    filename, name = request.param
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset' / filename
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in ('parse_score_from_response', 'ask_judge')]
    selected = ast.ClassDef(name=cls.name, bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {'re': re, 'time': time}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('response', ['Score: 20', 'Score: 2.8', '2.8', 'Score: 2e3', 'Score: 1.2.3'])
def test_invalid_text_scores_cannot_be_truncated_or_exceed_the_scale(scorer, response):
    assert scorer.parse_score_from_response(response) is None


@pytest.mark.parametrize('response,expected', [
    ('Score: 0', 0.0), ('Score: 2', 2.0), ('Score: 1.5', 1.5),
    ('1.5', 1.5), ('{"score": 2}', 2.0), ('{"Score": 0}', 0.0),
])
def test_valid_encodings_keep_the_same_score(scorer, response, expected):
    assert scorer.parse_score_from_response(response) == expected


def test_invalid_score_is_retried_instead_of_entering_the_metric(scorer):
    calls = []
    replies = iter(['Score: 20', 'Score: 1'])

    class Judge:
        def generate(self, prompt):
            calls.append(prompt)
            return next(replies)

    instance = scorer.__new__(scorer)
    instance.judge_model = Judge()
    instance.sleep = 0
    instance._safe_print_prompt = lambda *_: None
    assert instance.ask_judge('synthetic prompt') == 1.0
    assert len(calls) == 2
