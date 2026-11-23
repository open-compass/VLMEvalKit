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


@pytest.mark.parametrize('suffix', ['json', 'jsonl'])
@pytest.mark.parametrize('scalar_type', [
    np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.float16, np.float32, np.float64, np.complex64, np.complex128,
])
def test_dump_preserves_supported_scalar_types(tmp_path, suffix, scalar_type):
    module = _load_modules()['vlmeval.smp.file']
    value = scalar_type(3)
    data = {'value': value}
    target = tmp_path / ('scalar.' + suffix)
    module.dump(data if suffix == 'json' else [data], str(target))
    assert not target.with_suffix('.pkl').exists()
    expected = {'real': 3.0, 'imag': 0.0} if scalar_type in (np.complex64, np.complex128) else 3
    assert module.load(str(target)) == ({'value': expected} if suffix == 'json' else [{'value': expected}])


@pytest.mark.parametrize('suffix', ['json', 'jsonl'])
@pytest.mark.parametrize('value', [
    np.longdouble('1.000000000000000001'),
    np.clongdouble('1.000000000000000001+2j'),
    np.timedelta64(5, 'ns'),
    np.timedelta64(5, 'D'),
    np.longlong(3),
    np.ulonglong(3),
], ids=['longdouble', 'clongdouble', 'timedelta-ns', 'timedelta-D', 'longlong', 'ulonglong'])
def test_unsupported_scalar_types_keep_pickle_fallback(tmp_path, suffix, value):
    module = _load_modules()['vlmeval.smp.file']
    # Some platforms alias longlong to the already supported int64/uint64.
    if type(value) in (np.int64, np.uint64):
        pytest.skip('This platform already includes this integer type in the explicit tuple')
    with pytest.raises(TypeError):
        json.dumps({'value': value}, cls=module.NumpyEncoder)
    data = {'value': value}
    target = tmp_path / ('scalar.' + suffix)
    module.dump(data if suffix == 'json' else [data], str(target))
    pickle_target = target.with_suffix('.pkl')
    assert pickle_target.exists()
    recovered = module.load(str(pickle_target))
    actual = recovered['value'] if suffix == 'json' else recovered[0]['value']
    # NumPy pickles canonicalize scalar aliases; dtype and value remain exact.
    assert isinstance(actual, np.generic)
    assert actual == value
    assert actual.dtype == value.dtype
