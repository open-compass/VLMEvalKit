import functools
import importlib.util
import json
import logging
import sys
import types
from unittest import mock

import pytest
from PIL import Image


@functools.lru_cache(maxsize=1)
def _load_file_module():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']

    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']

    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = logging.getLogger

    misc = types.ModuleType('vlmeval.smp.misc')
    misc.toliststr = lambda value: value if isinstance(value, list) else [value]

    vlm = types.ModuleType('vlmeval.smp.vlm')
    vlm.decode_base64_to_image_file = mock.MagicMock()

    validators = types.ModuleType('validators')
    validators.url = lambda value: False

    modules = {
        'validators': validators,
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.log': log,
        'vlmeval.smp.misc': misc,
        'vlmeval.smp.vlm': vlm,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.smp.file',
            'vlmeval/smp/file.py',
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['vlmeval.smp.file'] = module
        spec.loader.exec_module(module)
        sys.modules.pop('vlmeval.smp.file', None)
        return module


def test_parse_file_detects_extensionless_image(tmp_path):
    module = _load_file_module()
    image_path = tmp_path / 'sitebench-image'
    Image.new('RGB', (4, 4), color='red').save(image_path, format='PNG')

    assert module.parse_file(str(image_path)) == ('image/png', str(image_path))


def test_parse_file_keeps_unknown_for_non_image(tmp_path):
    module = _load_file_module()
    file_path = tmp_path / 'extensionless-text'
    file_path.write_text('not an image')

    assert module.parse_file(str(file_path)) == ('unknown', str(file_path))


def test_parse_file_keeps_unknown_when_pillow_rejects_image(tmp_path):
    module = _load_file_module()
    image_path = tmp_path / 'extensionless-image'
    image_path.write_bytes(b'image')

    error = Image.DecompressionBombError('image exceeds Pillow safety limit')
    with mock.patch.object(Image, 'open', side_effect=error):
        assert module.parse_file(str(image_path)) == ('unknown', str(image_path))


@pytest.mark.parametrize('payload', [
    {
        'columns': ['index', 'prediction'],
        'data': [[1, 'yes', 'unexpected']]
    },
    {
        'columns': 7,
        'data': [[1, 'yes']]
    },
    [{
        'index': 1,
        'prediction': 'yes'
    }, 'invalid row'],
])
def test_prediction_reuse_rejects_malformed_json_tables(tmp_path, monkeypatch, payload):
    module = _load_file_module()
    validators = types.ModuleType('validators')
    validators.url = lambda value: False
    monkeypatch.setitem(sys.modules, 'validators', validators)
    path = tmp_path / 'predictions.json'
    path.write_text(json.dumps(payload), encoding='utf-8')

    assert module.is_prediction_complete(str(path), [1]) is False
    with pytest.raises(ValueError, match='Invalid prediction table:'):
        module.load_prediction_file(str(path))


@pytest.mark.parametrize('payload', [
    {
        'columns': ['index', 'prediction'],
        'data': [[1, 'yes']]
    },
    [{
        'index': 1,
        'prediction': 'yes'
    }],
    {
        'index': [1],
        'prediction': ['yes']
    },
])
def test_prediction_reuse_preserves_valid_json_tables(tmp_path, monkeypatch, payload):
    module = _load_file_module()
    validators = types.ModuleType('validators')
    validators.url = lambda value: False
    monkeypatch.setitem(sys.modules, 'validators', validators)
    path = tmp_path / 'predictions.json'
    path.write_text(json.dumps(payload), encoding='utf-8')

    assert module.is_prediction_complete(str(path), [1]) is True
    frame = module.load_prediction_file(str(path))
    assert frame['prediction'].tolist() == ['yes']
