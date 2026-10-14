import base64
import importlib.util
import io
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


@pytest.mark.parametrize('img_size,total_img_size,count,expected', [(-1, 200, 4, 100), (300, 200, 4, 100),
                                                                    (100, 400, 4, 100), (-1, 1, 4, 1),
                                                                    (-1, -1, 1, 600)])
def test_gpt_encoded_images_respect_total_budget(tmp_path, monkeypatch, img_size, total_img_size, count, expected):
    module = _load_modules('api.openai_sdk', 'api.gpt')['vlmeval.api.gpt']
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    image_path = tmp_path / 'image.png'
    Image.new('RGB', (600, 300), 'red').save(image_path)
    wrapper = module.OpenAIWrapper(key='test-key', img_size=img_size, total_img_size=total_img_size)
    content = wrapper.prepare_itlist([{'type': 'image', 'value': str(image_path)}] * count)
    for item in content:
        encoded = item['image_url']['url'].split(',', 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            assert max(image.size) == expected
