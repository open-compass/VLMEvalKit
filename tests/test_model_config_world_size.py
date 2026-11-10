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


class _FakeSeries(list):
    def isin(self, values):
        return _FakeSeries(value in values for value in self)

    def __invert__(self):
        return _FakeSeries(not value for value in self)


class _FakeILoc:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, index):
        if isinstance(index, int):
            return {'index': self.frame.indices[index]}
        return _FakeFrame(self.frame.indices[item] for item in index)


class _FakeFrame:
    def __init__(self, indices):
        self.indices = list(indices)
        self.iloc = _FakeILoc(self)

    def __getitem__(self, key):
        if key == 'index':
            return _FakeSeries(self.indices)
        return _FakeFrame(
            value for value, keep in zip(self.indices, key) if keep
        )

    def __len__(self):
        return len(self.indices)


class _FakeDataset:
    dataset_name = 'WorldSizeTestDataset'
    pack = False

    def __init__(self):
        self.data = _FakeFrame([0])

    def __len__(self):
        return len(self.data)


def isolated_inference_function(module_name, supported_vlm):
    inference_path = Path(__file__).resolve().parents[1] / 'vlmeval' / f'{module_name}.py'
    inference_tree = ast.parse(inference_path.read_text(encoding='utf-8'))
    function = next(
        node for node in inference_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == 'infer_data'
    )
    namespace = {
        'os': os,
        'osp': os.path,
        'np': types.SimpleNamespace(all=all),
        'load': lambda _: {},
        'get_rank_and_world_size': lambda: (0, 2),
        'resolve_dataset_alias_name': lambda name, alias=None: alias or name,
        'supported_VLM': supported_vlm,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(inference_path), 'exec'), namespace)
    return namespace['infer_data']


@pytest.mark.parametrize('module_name', ['inference', 'inference_mt', 'inference_video'])
def test_failed_inference_model_construction_restores_world_size(module_name, tmp_path):
    def build_model(**kwargs):
        assert 'WORLD_SIZE' not in os.environ
        raise RuntimeError('model construction failed')

    infer_data = isolated_inference_function(module_name, {'WorldSizeTestModel': build_model})
    with mock.patch.dict(os.environ, {'WORLD_SIZE': '2'}):
        with pytest.raises(RuntimeError, match='model construction failed'):
            infer_data(
                model='WorldSizeTestModel',
                model_name='WorldSizeTestModel',
                work_dir=str(tmp_path),
                dataset=_FakeDataset(),
                out_file=str(tmp_path / 'rank.pkl'),
            )
        assert os.environ['WORLD_SIZE'] == '2'
