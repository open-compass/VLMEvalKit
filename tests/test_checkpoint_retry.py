import importlib.util
import os
import sys
import types
from pathlib import Path

import pandas as pd
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


def _load_inference_module(name):
    modules = _load_modules('smp.dataset_alias')
    smp = modules['vlmeval.smp']
    smp.get_rank_and_world_size = lambda: (0, 1)
    smp.upsert_dataset_status = lambda *args, **kwargs: None
    utils = types.ModuleType('vlmeval.utils')
    utils.__path__ = [str(Path(__file__).resolve().parents[1] / 'vlmeval/utils')]

    def track_progress(func, tasks, save, keys, **kwargs):
        result = smp.load(save) if os.path.exists(save) else {}
        for key, task in zip(keys, tasks):
            result[key] = func(**task)
        smp.dump(result, save)
        return list(result.values())

    utils.track_progress_rich = track_progress
    sys.modules['vlmeval.utils'] = utils
    config = types.ModuleType('vlmeval.config')
    config.supported_VLM = {}
    sys.modules['vlmeval.config'] = config
    torch = types.ModuleType('torch')
    torch.distributed = types.ModuleType('torch.distributed')
    sys.modules['torch'] = torch
    sys.modules['torch.distributed'] = torch.distributed
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('vlmeval.' + name, root / ('vlmeval/' + name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, modules


@pytest.mark.parametrize('retry_failed', [True, False])
def test_api_checkpoint_retries_failed_turn_answers(tmp_path, monkeypatch, retry_failed):
    module, modules = _load_inference_module('inference_mt')
    base = modules['vlmeval.api.base']
    monkeypatch.setattr(base.time, 'sleep', lambda delay: None)
    failed = ['ok', module.FAIL_MSG + 'rate limited']
    success = ['cached']
    checkpoint = tmp_path / 'mock_Example_checkpoint.pkl'
    modules['vlmeval.smp.file'].dump({'0': success, '1': failed}, str(checkpoint))

    class Dataset:
        dataset_name = 'Example'
        data = pd.DataFrame({'index': [0, 1]})

        def build_prompt(self, row):
            return [{
                'role': 'user',
                'content': [{
                    'type': 'text',
                    'value': str(row['index'])
                }]
            }, {
                'role': 'assistant',
                'content': 'reference'
            }]

    class Model(base.BaseAPI):
        is_api = True
        calls = 0

        def generate_inner(self, inputs, **kwargs):
            self.calls += 1
            return 0, 'renewed', None

    model = Model(retry=1, wait=0, verbose=False)
    result = module.infer_data_api(model, str(tmp_path), 'mock', Dataset(), retry_failed=retry_failed)
    assert result[0] == success
    assert model.calls == int(retry_failed)
    assert result[1] == (['renewed'] if retry_failed else failed)
