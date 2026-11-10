import ast
import copy
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest


@contextmanager
def isolated_model_builder(supported_vlm):
    run_path = Path(__file__).resolve().parents[1] / 'run.py'
    run_tree = ast.parse(run_path.read_text(encoding='utf-8'))
    function = next(
        node for node in run_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'build_model_from_config'
    )
    namespace = {'os': os, 'cp': copy, 'supported_VLM': supported_vlm}
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = []
    api = types.ModuleType('vlmeval.api')
    vlm = types.ModuleType('vlmeval.vlm')
    vlmeval.api = api
    vlmeval.vlm = vlm
    with mock.patch.dict(sys.modules, {
        'vlmeval': vlmeval,
        'vlmeval.api': api,
        'vlmeval.vlm': vlm,
    }):
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(run_path), 'exec'), namespace)
        yield namespace['build_model_from_config']


def test_shortcut_model_config_restores_world_size():
    model = object()

    def build_model(**kwargs):
        assert kwargs == {}
        assert 'WORLD_SIZE' not in os.environ
        return model

    with mock.patch.dict(os.environ, {'WORLD_SIZE': '2'}):
        with isolated_model_builder({'WorldSizeTestModel': build_model}) as builder:
            assert builder({'WorldSizeTestModel': {}}, 'WorldSizeTestModel') is model
        assert os.environ['WORLD_SIZE'] == '2'


def test_failed_model_config_restores_world_size():
    def build_model(**kwargs):
        assert 'WORLD_SIZE' not in os.environ
        raise RuntimeError('model construction failed')

    with mock.patch.dict(os.environ, {'WORLD_SIZE': '2'}):
        with isolated_model_builder({'WorldSizeTestModel': build_model}) as builder:
            with pytest.raises(RuntimeError, match='model construction failed'):
                builder({'WorldSizeTestModel': {}}, 'WorldSizeTestModel')
        assert os.environ['WORLD_SIZE'] == '2'
