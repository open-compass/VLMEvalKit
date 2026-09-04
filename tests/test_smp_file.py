import functools
import importlib.util
import logging
import sys
import types
from unittest import mock

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
