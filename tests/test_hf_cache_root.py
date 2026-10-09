import importlib.util
import sys
import types
from pathlib import Path

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


@pytest.mark.parametrize('variable', ['HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME'])
def test_cache_environment_matches_hub_semantics(tmp_path, monkeypatch, variable):
    modules = _load_modules()
    module = modules['vlmeval.smp.file']
    for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'HF_HOME'):
        monkeypatch.delenv(name, raising=False)
    configured = tmp_path / 'custom-cache'
    configured.mkdir()
    cache = configured / 'hub' if variable == 'HF_HOME' else configured
    repo = cache / 'datasets--test--example'
    snapshot = repo / 'snapshots' / ('a' * 40)
    snapshot.mkdir(parents=True)
    (snapshot / 'dataset.tsv').write_text('index\tquestion\n0\tQuestion\n')
    (repo / 'refs').mkdir()
    (repo / 'refs' / 'main').write_text('a' * 40)
    monkeypatch.setenv(variable, str(configured))
    assert module.HFCacheRoot() == str(cache)
    misc = _load_modules('smp.misc')['vlmeval.smp.misc']
    assert misc.get_cache_path('test/example') == str(snapshot)
