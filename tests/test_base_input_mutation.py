import copy
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


def test_generate_preserves_roles_and_data_urls(tmp_path, monkeypatch):
    modules = _load_modules()
    base = modules['vlmeval.api.base']
    vlm = modules['vlmeval.smp.vlm']
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    monkeypatch.setattr(base.time, 'sleep', lambda delay: None)

    class Echo(base.BaseAPI):

        def generate_inner(self, inputs, **kwargs):
            return 0, 'answer', None

    inline_image = 'data:image/jpeg;base64,' + vlm.encode_image_to_base64(Image.new('RGB', (4, 4)))
    message = [{
        'role': 'system',
        'type': 'text',
        'value': 'Instruction'
    }, {
        'role': 'user',
        'type': 'image',
        'value': inline_image
    }, {
        'role': 'user',
        'type': 'text',
        'value': 'Question'
    }]
    original = copy.deepcopy(message)
    model = Echo(retry=1, wait=0, verbose=False)
    assert model.generate(message) == 'answer'
    assert message == original
    assert model.generate(message) == 'answer'
    assert message == original


def test_chat_preserves_original_content(tmp_path, monkeypatch):
    base = _load_modules()['vlmeval.api.base']
    monkeypatch.setattr(base.time, 'sleep', lambda delay: None)

    class Echo(base.BaseAPI):

        def generate_inner(self, inputs, **kwargs):
            return 0, 'answer', None

    message = [{'role': 'user', 'content': 'Question'}]
    assert Echo(retry=1, wait=0, verbose=False).chat(message) == 'answer'
    assert message == [{'role': 'user', 'content': 'Question'}]
