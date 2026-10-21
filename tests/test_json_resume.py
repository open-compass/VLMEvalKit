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


@pytest.mark.parametrize('name', ['inference', 'inference_mt', 'inference_video'])
def test_public_inference_job_reuses_json_predictions(tmp_path, name):
    module, modules = _load_inference_module(name)
    file = modules['vlmeval.smp.file']
    target = tmp_path / 'mock_Example.json'
    file.dump(pd.DataFrame({'index': [0, 1], 'prediction': ['cached A', 'cached B']}), str(target))

    class Dataset:
        dataset_name = 'Example'
        data = pd.DataFrame({'index': [0, 1]})

        def __len__(self):
            return len(self.data)

    model = object()
    entry = getattr(module, {
        'inference': 'infer_data_job',
        'inference_mt': 'infer_data_job_mt',
        'inference_video': 'infer_data_job_video'
    }[name])
    result = entry(model, str(tmp_path), 'mock', Dataset(), result_file=str(target))
    assert result is model
    assert file.load(str(target)) == [{'index': 0, 'prediction': 'cached A'}, {'index': 1, 'prediction': 'cached B'}]


def test_api_pipeline_reuses_json_result_file(tmp_path):
    module, modules = _load_inference_module('inference_api')
    file = modules['vlmeval.smp.file']
    target = tmp_path / 'mock_Example.json'
    file.dump(pd.DataFrame({'index': [0, 1], 'prediction': ['cached', module.FAIL_MSG]}), str(target))
    dataset = types.SimpleNamespace(dataset_name='Example', data=pd.DataFrame({'index': [0, 1]}))
    cfg = module.DatasetConfig(dataset_name='Example',
                               dataset_obj=dataset,
                               model_name='mock',
                               model_obj=None,
                               work_dir=str(tmp_path),
                               result_file=str(target),
                               judge_kwargs={})
    pipeline = module.APIEvalPipeline([cfg], concurrency=1, run_eval=False)
    try:
        assert pipeline._load_checkpoint('Example') == {'0': 'cached'}
    finally:
        pipeline._shutdown_executors()
