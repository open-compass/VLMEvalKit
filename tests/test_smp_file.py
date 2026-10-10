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


def test_localization_returns_resolved_paths_for_existing_image_path_column(tmp_path, monkeypatch):
    import base64
    import io

    import pandas as pd

    module = _load_file_module()
    spec = importlib.util.spec_from_file_location('image_codec', 'vlmeval/smp/vlm.py')
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    monkeypatch.setattr(module, 'decode_base64_to_image_file', codec.decode_base64_to_image_file)
    monkeypatch.setattr(module, 'LMUDataRoot', lambda: str(tmp_path))
    pool = mock.MagicMock()
    pool.__enter__.return_value = pool
    pool.map.side_effect = lambda function, items: list(map(function, items))
    monkeypatch.setattr(module.mp, 'Pool', lambda workers: pool)
    buffer = io.BytesIO()
    Image.new('RGB', (4, 4), color='red').save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    absolute = tmp_path / 'absolute.png'
    result = module.localize_df(pd.DataFrame({
        'index': [0, 1], 'image': [encoded, encoded],
        'image_path': ['provided.png', str(absolute)],
    }), 'Example')
    expected = [str(tmp_path / 'images' / 'Example' / 'provided.png'), str(absolute)]
    assert result['image_path'].tolist() == expected
    assert 'image' not in result
    for path in result['image_path']:
        with Image.open(path) as image:
            assert image.getpixel((0, 0)) == (255, 0, 0)
