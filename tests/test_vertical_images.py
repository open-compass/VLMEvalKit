import importlib.util
import sys
import types
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _restore_modules():
    original = {name: module for name, module in sys.modules.items() if name.startswith(('vlmeval', 'torch'))}
    yield
    for name in list(sys.modules):
        if name.startswith(('vlmeval', 'torch')):
            del sys.modules[name]
    sys.modules.update(original)


def _load_modules(*names):
    root = Path(__file__).resolve().parents[1]
    modules = {}
    for name in ('vlmeval', 'vlmeval.smp', 'vlmeval.api'):
        package = types.ModuleType(name)
        package.__path__ = [str(root / name.replace('.', '/'))]
        modules[name] = package
    misc = types.ModuleType('vlmeval.smp.misc')
    misc.toliststr = lambda value: value if isinstance(value, list) else [value]
    modules[misc.__name__] = misc
    sys.modules.update(modules)
    for name in ('smp.log', 'smp.vlm', 'smp.file', 'api.base', *names):
        full_name = 'vlmeval.' + name
        spec = importlib.util.spec_from_file_location(full_name, root / (full_name.replace('.', '/') + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        modules[full_name] = module
        spec.loader.exec_module(module)
        if name.startswith('smp.'):
            for key, value in vars(module).items():
                if not key.startswith('_'):
                    setattr(modules['vlmeval.smp'], key, value)
    return modules


@pytest.mark.parametrize('target_size', [-1, 12])
def test_vertical_concat_stacks_images_using_heights(tmp_path, target_size):
    module = _load_modules()['vlmeval.smp.vlm']
    paths = []
    for name, size, color in [('top', (10, 20), 'red'), ('bottom', (20, 10), 'blue')]:
        path = tmp_path / (name + '.png')
        Image.new('RGB', size, color).save(path)
        paths.append(str(path))
    result = module.concat_images_vlmeval(paths, target_size=target_size, mode='v', return_image=True)
    top_height, bottom_height, width = (20, 10, 20) if target_size == -1 else (24, 6, 12)
    assert result.size == (width, top_height + bottom_height)
    assert result.getpixel((0, top_height - 1)) == (255, 0, 0)
    assert result.getpixel((0, top_height)) == (0, 0, 255)
    assert result.getpixel((0, top_height + bottom_height - 1)) == (0, 0, 255)
    path = module.concat_images_vlmeval(paths, mode='v')
    with Image.open(path) as saved:
        assert saved.size == (20, 30)


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
@pytest.mark.parametrize('operation', ['encode', 'decode'])
def test_transparent_diagrams_keep_white_background(mode, operation, monkeypatch):
    import base64
    import io

    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    modules = _load_modules()
    image = Image.new('RGBA', (4, 4), (0, 0, 0, 0))
    image.putpixel((2, 2), (0, 0, 0, 255))
    if mode == 'LA':
        image = image.convert('LA')
    elif mode == 'P':
        image = Image.new('P', (4, 4), 0)
        image.putpalette([0] * 768)
        image.putpixel((2, 2), 1)
        image.info['transparency'] = 0
    if operation == 'encode':
        encoded = modules['vlmeval.smp.vlm'].encode_image_to_base64(image, fmt='PNG')
        actual = Image.open(io.BytesIO(base64.b64decode(encoded)))
    else:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        actual = modules['vlmeval.smp.vlm'].decode_base64_to_image(
            base64.b64encode(buffer.getvalue()))
    assert actual.getpixel((0, 0)) == (255, 255, 255)
    assert actual.getpixel((2, 2)) == (0, 0, 0)


@pytest.mark.parametrize('mode', ['RGB', 'L', 'I;16'])
def test_opaque_image_decode_keeps_mode_and_pixels(mode):
    import base64
    import io

    modules = _load_modules()
    value = (12, 34, 56) if mode == 'RGB' else 127
    image = Image.new(mode, (4, 4), value)
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    actual = modules['vlmeval.smp.vlm'].decode_base64_to_image(base64.b64encode(buffer.getvalue()))
    assert actual.mode == mode
    assert actual.getpixel((0, 0)) == value
