import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


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


@pytest.mark.parametrize('suffix', ['json', 'jsonl'])
def test_dump_numpy_values_remains_json(tmp_path, suffix):
    module = _load_modules()['vlmeval.smp.file']
    values = {
        'score': np.float32(0.75),
        'correct': np.bool_(True),
        'counts': np.array([1, 2]),
        'phase': np.complex64(1 + 2j),
        'index': np.int64(3),
        'void': np.void(b'')
    }
    target = tmp_path / ('metrics.' + suffix)
    module.dump(values if suffix == 'json' else [values], str(target))
    assert not target.with_suffix('.pkl').exists()
    expected = {
        'score': 0.75,
        'correct': True,
        'counts': [1, 2],
        'phase': {
            'real': 1.0,
            'imag': 2.0
        },
        'index': 3,
        'void': None
    }
    assert json.loads(target.read_text()) == expected
