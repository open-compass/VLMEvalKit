import ast
import base64
import importlib.util
import io
import json
import os
import pickle
import string
import sys
import types
import warnings
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from IPython.display import display
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _definitions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / relative), 'exec'), namespace)


@pytest.fixture
def consumers(monkeypatch, tmp_path):
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    namespace = dict(np=np, pd=pd, os=os, osp=os.path, warnings=warnings,
                     abstractmethod=abstractmethod, string=string, json=json, pickle=pickle,
                     base64=base64, io=io, display=display)
    # Execute exact producer/consumer definitions without the unrelated model
    # registry or the browser script's module-level paid API translator.
    _definitions('vlmeval/smp/file.py', ['LMUDataRoot', 'load', 'file_size'], namespace)
    _definitions('vlmeval/smp/misc.py', ['istype'], namespace)
    for name in ('vlmeval', 'vlmeval.smp'):
        package = types.ModuleType(name)
        package.__path__ = [str(ROOT / name.replace('.', '/'))]
        monkeypatch.setitem(sys.modules, name, package)
    for name in ('log', 'vlm'):
        full_name = f'vlmeval.smp.{name}'
        spec = importlib.util.spec_from_file_location(full_name, ROOT / f'vlmeval/smp/{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full_name, module)
        spec.loader.exec_module(module)
    module.display = display  # Same real IPython display supplied by notebooks.
    namespace.update(mmqa_display=module.mmqa_display,
                     encode_image_file_to_base64=module.encode_image_file_to_base64)
    _definitions('vlmeval/dataset/text_base.py', ['TextBaseDataset'], namespace)
    _definitions('vlmeval/dataset/text_mcq.py', ['TextMCQDataset', 'CustomTextMCQDataset'], namespace)
    browser = dict(namespace)
    _definitions('scripts/data_browser.py', ['image_to_mdstring', 'images_to_md', 'mmqa_display'], browser)
    return namespace, browser


def _text_dataset(namespace, tmp_path):
    pd.DataFrame([dict(index=1, question='Which is a planet?', A='Mars', B='Granite', answer='A')]).to_csv(
        tmp_path / 'local_text.tsv', sep='\t', index=False)
    dataset = namespace['CustomTextMCQDataset']('local_text')
    assert dataset.MODALITY == 'TEXT'
    assert dataset.build_prompt(0)[0]['type'] == 'text'
    assert 'image' not in dataset[0] and 'image_path' not in dataset[0]
    return dataset


def test_native_custom_text_dataset_console_display(consumers, tmp_path, capsys):
    namespace, _ = consumers
    dataset = _text_dataset(namespace, tmp_path)
    dataset.display(0)
    output = capsys.readouterr().out
    assert 'INDEX: 1' in output and 'Which is a planet?' in output
    assert 'Mars' in output and 'Granite' in output


def test_browser_renderer_accepts_native_text_dataset_row(consumers, tmp_path):
    namespace, browser = consumers
    dataset = _text_dataset(namespace, tmp_path)
    text, images = browser['mmqa_display'](dataset[0])
    assert 'INDEX: 1' in text and 'Which is a planet?' in text
    assert '-A: Mars' in text and '- ANSWER: A' in text
    assert images == ''


def test_browser_base64_image_control(consumers):
    _, browser = consumers
    buffer = io.BytesIO()
    Image.new('RGB', (32, 24), 'red').save(buffer, 'JPEG')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    text, images = browser['mmqa_display'](dict(index=2, question='What color?', image=encoded))
    assert 'What color?' in text
    decoded = Image.open(io.BytesIO(base64.b64decode(images.split('base64,')[1][:-1])))
    assert decoded.size == (32, 24) and decoded.getpixel((0, 0))[0] > 240


def test_browser_image_file_control(consumers, tmp_path):
    _, browser = consumers
    image_path = tmp_path / 'red.png'
    Image.new('RGB', (32, 24), 'red').save(image_path)
    text, images = browser['mmqa_display'](dict(index=3, question='What color?', image_path=str(image_path)))
    assert 'What color?' in text
    decoded = Image.open(io.BytesIO(base64.b64decode(images.split('base64,')[1][:-1])))
    assert decoded.size == (32, 24) and decoded.getpixel((0, 0))[0] > 240


def test_console_notebook_image_control(consumers, capsys):
    namespace, _ = consumers
    buffer = io.BytesIO()
    Image.new('RGB', (32, 24), 'red').save(buffer, 'JPEG')
    namespace['mmqa_display'](dict(index=4, question='What color?', image=base64.b64encode(buffer.getvalue()).decode()))
    assert 'What color?' in capsys.readouterr().out
