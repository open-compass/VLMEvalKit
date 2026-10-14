import ast
import base64
import csv
import importlib.util
import io
import json
import logging
import os
import pickle
import random
import re
import string
import sys
import types
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _definitions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text(encoding='utf-8'))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / relative), 'exec'), namespace)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    original = {name: module for name, module in sys.modules.items() if name.startswith('vlmeval')}
    namespace = dict(np=np, pd=pd, os=os, osp=os.path, warnings=warnings, abstractmethod=abstractmethod,
                     string=string, json=json, pickle=pickle, base64=base64, io=io, csv=csv,
                     re=re, rd=random, defaultdict=defaultdict, ABCMeta=ABCMeta, Any=Any, Path=Path,
                     PANDAS_DEFAULT_NA_VALUES=STR_NA_VALUES, INFER_FAIL_MSG='Failed to obtain answer',
                     logger=logging.getLogger('mcq-table-tests'), __package__='vlmeval.dataset')
    try:
        # Only namespace isolation: all loaders, dataset classes, matching,
        # scoring and reporting below are exact shipping definitions.
        for name in ('vlmeval', 'vlmeval.smp', 'vlmeval.dataset', 'vlmeval.dataset.utils'):
            package = types.ModuleType(name)
            package.__path__ = [str(ROOT / name.replace('.', '/'))]
            sys.modules[name] = package
        for name in ('log', 'vlm'):
            full_name = f'vlmeval.smp.{name}'
            spec = importlib.util.spec_from_file_location(full_name, ROOT / f'vlmeval/smp/{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        namespace.update(read_ok=module.read_ok, decode_base64_to_image_file=module.decode_base64_to_image_file,
                         get_logger=sys.modules['vlmeval.smp.log'].get_logger)
        file_definitions = [
            'LMUDataRoot', 'load', 'file_size', 'dump', 'NumpyEncoder', 'get_pred_file_format',
            'get_eval_file_format', 'get_file_extension', 'get_intermediate_file_path',
            '_prediction_table', '_should_convert_to_dataframe', 'load_prediction_file', 'load_table_file',
        ]
        _definitions('vlmeval/smp/file.py', file_definitions, namespace)
        misc_definitions = ['istype', 'cn_string', 'd2df', 'timestr', 'toliststr', 'listinstr']
        _definitions('vlmeval/smp/misc.py', misc_definitions, namespace)
        _definitions('vlmeval/smp/vlm.py', ['build_option_str'], namespace)
        _definitions('vlmeval/dataset/text_base.py', ['TextBaseDataset'], namespace)
        _definitions('vlmeval/dataset/text_mcq.py', ['TextMCQDataset', 'CustomTextMCQDataset'], namespace)
        image_definitions = ['img_root_map', 'restore_image_default_na', 'ImageBaseDataset']
        _definitions('vlmeval/dataset/image_base.py', image_definitions, namespace)
        tree = ast.parse((ROOT / 'vlmeval/dataset/image_mcq.py').read_text())
        constants = [node for node in tree.body if isinstance(node, ast.Assign)]
        exec(compile(ast.Module(body=constants, type_ignores=[]), 'image_constants', 'exec'), namespace)
        _definitions('vlmeval/dataset/image_mcq.py', ['ImageMCQDataset', 'CustomMCQDataset', 'CVBench'], namespace)
        spec = importlib.util.spec_from_file_location('matching_table_tests', ROOT / 'vlmeval/utils/matching_util.py')
        matching = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(matching)
        namespace.update(can_infer=matching.can_infer, can_infer_lego=matching.can_infer_lego)
        full_name = 'vlmeval.dataset.utils.multiple_choice'
        scoring = types.ModuleType(full_name)
        scoring.__dict__.update(namespace)
        sys.modules[full_name] = scoring
        tree = ast.parse((ROOT / 'vlmeval/dataset/utils/multiple_choice.py').read_text())
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), 'actual_mcq_scoring', 'exec'), scoring.__dict__)
        yield namespace, scoring
    finally:
        for name in list(sys.modules):
            if name.startswith('vlmeval'):
                del sys.modules[name]
        sys.modules.update(original)


@pytest.mark.parametrize('kind', ('text', 'image', 'cvbench'))
@pytest.mark.parametrize('input_format, intermediate_format', (
    ('xlsx', 'xlsx'), ('tsv', 'tsv'), ('json', 'json'), ('xlsx', 'json'),
))
def test_native_export_and_cached_exact_matching_evaluation(
    runtime, monkeypatch, tmp_path, kind, input_format, intermediate_format
):
    namespace, scoring = runtime
    monkeypatch.setenv('PRED_FORMAT', intermediate_format)
    rows = pd.DataFrame([
        dict(index=0, question='Choose a planet.', A='Mars', B='Granite', answer='A'),
        dict(index=1, question='Choose a rock.', A='Mars', B='Granite', answer='B'),
    ])
    if kind != 'text':
        buffer = io.BytesIO()
        Image.new('RGB', (32, 24), 'red').save(buffer, 'JPEG')
        rows['image'] = base64.b64encode(buffer.getvalue()).decode()
    if kind == 'cvbench':
        rows['split'] = '3D'
        rows['prompt'] = rows['question'] + '\n(A) Mars\n(B) Granite'
    name = f'local_{kind}'
    namespace['dump'](rows, str(tmp_path / f'{name}.tsv'))
    if kind == 'cvbench':
        # Native custom TSV loading with the shipping CVBench evaluator, using
        # the supported dataset subclass extension point rather than mocks.
        klass = type('LocalCVBench', (namespace['CustomMCQDataset'], namespace['CVBench']), {})
    else:
        klass = namespace['CustomMCQDataset' if kind == 'image' else 'CustomTextMCQDataset']
    dataset = klass(name)
    prompt = dataset.build_prompt(0)
    assert prompt[-1]['type'] == 'text'
    if kind != 'text':
        assert Image.open(prompt[0]['value']).size == (32, 24)
    data = dataset.data.copy()
    data['prediction'] = ['A', 'A']
    target = str(tmp_path / f'predictions.{input_format}')
    namespace['dump'](data, target)
    # Real scorer outputs populate the native resume cache; no fake matcher,
    # fabricated hit values, optional judge or model generation is used.
    cache = {}
    for _, row in data.iterrows():
        item = row.copy()
        item['GT'] = item['answer']
        cache[row['index']] = scoring.eval_vanilla(None, item, dataset_name=name)
    assert [cache[index]['hit'] for index in (0, 1)] == [1, 0]
    cache_path = namespace['get_intermediate_file_path'](target, '_exact_matching_result', 'pkl')
    namespace['dump'](cache, cache_path)
    result = dataset.evaluate(target, model='exact_matching', nproc=1)
    assert result['Overall'].tolist() == [0.5]
    scored_path = namespace['get_intermediate_file_path'](target, '_exact_matching_result')
    scored = namespace['load_table_file'](scored_path)
    assert scored['index'].tolist() == [0, 1]
    assert scored['hit'].tolist() == [1, 0]


@pytest.mark.parametrize('value', ({'model': 'local fixture'}, [{'arbitrary': 'record'}]))
def test_ordinary_json_loading_keeps_its_return_type(runtime, tmp_path, value):
    namespace, _ = runtime
    target = tmp_path / 'metadata.json'
    target.write_text(json.dumps(value), encoding='utf-8')
    assert namespace['load'](str(target)) == value
