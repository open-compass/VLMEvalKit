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
