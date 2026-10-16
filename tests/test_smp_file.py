import functools
import importlib.util
import logging
import sys
import types
from unittest import mock

from PIL import Image


@functools.lru_cache(maxsize=1)
def _load_file_module():
    importlib.import_module('pandas')
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


def test_decode_multiple_images_with_one_listed_path(tmp_path, monkeypatch):
    import base64
    import io

    module = _load_file_module()
    spec = importlib.util.spec_from_file_location('image_codec', 'vlmeval/smp/vlm.py')
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    monkeypatch.setattr(module, 'decode_base64_to_image_file', codec.decode_base64_to_image_file)
    encoded = []
    for color in ('red', 'blue'):
        buffer = io.BytesIO()
        Image.new('RGB', (4, 4), color=color).save(buffer, format='PNG')
        encoded.append(base64.b64encode(buffer.getvalue()).decode('ascii'))
    paths = module.decode_img_omni((str(tmp_path), encoded, ['scene.png']))
    assert paths == [str(tmp_path / 'scene_0.png'), str(tmp_path / 'scene_1.png')]
    with Image.open(paths[0]) as red, Image.open(paths[1]) as blue:
        assert red.getpixel((0, 0)) == (255, 0, 0)
        assert blue.getpixel((0, 0)) == (0, 0, 255)
