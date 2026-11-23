import ast
from pathlib import Path

import pandas as pd
import pytest


def dataset_fixture(meta_only):
    root = Path(__file__).resolve().parents[1]
    base_tree = ast.parse((root / 'vlmeval/dataset/image_base.py').read_text())
    base_class = next(node for node in base_tree.body
                      if isinstance(node, ast.ClassDef) and node.name == 'ImageBaseDataset')
    method = next(node for node in base_class.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'build_prompt')
    namespace = {'toliststr': lambda value: value}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'image_base.py', 'exec'), namespace)
    base = type('ImageBaseDataset', (), {'build_prompt': namespace['build_prompt']})
    tree = ast.parse((root / 'vlmeval/dataset/image_vqa.py').read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'VizWiz')
    namespace = {'ImageBaseDataset': base}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'image_vqa.py', 'exec'), namespace)
    dataset = namespace['VizWiz']()
    dataset.meta_only = meta_only
    dataset.data = pd.DataFrame([
        {'question': 'What color is this?', 'image_path': ['first.jpg', 'second.jpg']}])
    dataset.dump_image = lambda line: ['first.jpg', 'second.jpg']
    return dataset


@pytest.mark.parametrize('indexed', [False, True])
@pytest.mark.parametrize('meta_only', [False, True])
def test_vizwiz_prompt_requests_short_answers_and_unanswerable(indexed, meta_only):
    dataset = dataset_fixture(meta_only)
    line = 0 if indexed else dataset.data.iloc[0]
    messages = dataset.build_prompt(line)
    assert messages[:2] == [{'type': 'image', 'value': 'first.jpg'},
                            {'type': 'image', 'value': 'second.jpg'}]
    assert messages[-1]['value'] == (
        'What color is this?'
        "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
        '\nAnswer the question using a single word or phrase.')
    assert dataset.data.iloc[0]['question'] == 'What color is this?'
