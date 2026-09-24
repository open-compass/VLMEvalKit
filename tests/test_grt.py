import json
from dataclasses import dataclass
from unittest import mock

import pytest

from vlmeval.config import supported_VLM
from vlmeval.vlm import grt


@pytest.fixture
def reproduction_environment(monkeypatch):
    for name in (*grt._PROCESS_SIZE_ENV, *grt._PROCESS_RANK_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def test_effective_seed_and_single_device_contract(monkeypatch, reproduction_environment):
    import numpy as np
    import torch
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('SLURM_NTASKS', '1')
    monkeypatch.setenv('LOCAL_RANK', '-1')
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', False)
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', True)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', True)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', True)
    with (
        mock.patch.object(torch.cuda, 'device_count', return_value=1),
        mock.patch.object(torch.distributed, 'is_initialized', return_value=False),
        mock.patch.object(grt.random, 'seed') as python_seed,
        mock.patch.object(np.random, 'seed') as numpy_seed,
        mock.patch.object(torch, 'manual_seed') as torch_seed,
        mock.patch.object(torch.cuda, 'manual_seed_all') as cuda_seed,
        mock.patch.object(torch, 'use_deterministic_algorithms') as deterministic,
    ):
        grt._configure_reproduction()
        python_seed.assert_called_once_with(0)
        numpy_seed.assert_called_once_with(1234)
        torch_seed.assert_called_once_with(1234)
        cuda_seed.assert_called_once_with(1234)
        deterministic.assert_called_once_with(True)
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32


@pytest.mark.parametrize('name', grt._PROCESS_SIZE_ENV)
@pytest.mark.parametrize('value', ['0', '2', 'invalid'])
def test_distributed_size_metadata_fails_before_cuda(
    monkeypatch, reproduction_environment, name, value,
):
    import torch
    monkeypatch.setenv(name, value)
    with mock.patch.object(torch.cuda, 'device_count') as device_count:
        with pytest.raises(ValueError, match='single process'):
            grt._configure_reproduction()
    device_count.assert_not_called()


@pytest.mark.parametrize('name', grt._PROCESS_RANK_ENV)
def test_nonzero_rank_metadata_fails_before_cuda(
    monkeypatch, reproduction_environment, name,
):
    import torch
    monkeypatch.setenv(name, '1')
    with mock.patch.object(torch.cuda, 'device_count') as device_count:
        with pytest.raises(ValueError, match='single process'):
            grt._configure_reproduction()
    device_count.assert_not_called()


def test_initialized_multiprocess_group_fails_before_cuda(reproduction_environment):
    import torch
    with (
        mock.patch.object(torch.distributed, 'is_available', return_value=True),
        mock.patch.object(torch.distributed, 'is_initialized', return_value=True),
        mock.patch.object(torch.distributed, 'get_world_size', return_value=2),
        mock.patch.object(torch.cuda, 'device_count') as device_count,
    ):
        with pytest.raises(ValueError, match='single process'):
            grt._configure_reproduction()
    device_count.assert_not_called()


@dataclass
class Request:
    request_type: str
    arguments: tuple
    idx: int
    metadata: dict

    @property
    def args(self):
        return self.arguments


@pytest.fixture
def backend(monkeypatch):
    constructed = []

    class Backend:
        def __init__(self, **kwargs):
            self.task_dict = {'old': {}}
            self.kwargs = kwargs
            self.requests = []
            self.error = False
            constructed.append(self)

        def generate_until(self, requests):
            self.requests.extend(requests)
            request = requests[0]
            context, generation, visual, doc_id, task, split = request.args
            assert request.request_type == 'generate_until'
            assert request.metadata == {'task': task, 'doc_id': doc_id, 'repeats': 1}
            doc = self.task_dict[task][split][doc_id]
            assert set(doc) == {'video_path', 'question'}
            assert visual(doc) == ['clip.mp4']
            assert context == doc['question'] == 'Question'
            assert generation['temperature'] == 0
            assert generation['until'] == ['ASSISTANT:']
            generation.pop('until')  # Real lmms wrappers mutate their generation dictionary.
            if self.error:
                raise RuntimeError('inference failure')
            return ['answer']

    def runtime(profile, role):
        return ({'task': 'densevideo', 'max_new_tokens': 48 if profile == 'qwen7' else 128},
                {'method': f'{profile}-{role}', 'model_args': 'frozen'}, Backend, Request,
                lambda args: {'from_frozen_args': args})

    monkeypatch.setattr(grt, '_load_runtime', runtime)
    monkeypatch.setattr(grt, '_configure_reproduction', lambda: None)
    return constructed


@pytest.mark.parametrize('name,profile,cap', [
    ('GRT-LLaVA-OneVision-0.5B', 'route31', 128),
    ('GRT-Qwen2.5-VL-3B', 'qwen3', 128),
    ('GRT-Qwen2.5-VL-7B', 'qwen7', 48),
])
def test_registered_model_real_request_boundary(backend, name, profile, cap):
    model = supported_VLM[name]()
    assert model.profile == profile
    assert model.backend.kwargs == {'from_frozen_args': 'frozen', 'batch_size': 1}
    message = [dict(type='video', value='clip.mp4'), dict(type='text', value='Question')]
    for _ in range(2):
        assert model.generate_inner(message, dataset='dive_bench_educational_high_fps') == 'answer'
    assert model.backend.task_dict == {'old': {}}
    assert model.backend.requests[0].args[1]['max_new_tokens'] == cap
    assert [request.args[3] for request in model.backend.requests] == [0, 1]


def test_backend_error_restores_task_state(backend):
    model = grt.GRT()
    model.backend.error = True
    with pytest.raises(RuntimeError, match='inference failure'):
        model.generate_inner([dict(type='video', value='clip.mp4'), dict(type='text', value='Question')],
                             dataset='densevideo')
    assert model.backend.task_dict == {'old': {}}


@pytest.mark.parametrize('kwargs', [{'profile': 'unknown'}, {'role': 'unknown'}, {'use_vllm': True}])
def test_unsupported_constructor_options(backend, kwargs):
    with pytest.raises(ValueError):
        grt.GRT(**kwargs)


def test_input_and_frame_overrides_fail_closed(backend):
    model = grt.GRT()
    message = [dict(type='video', value='clip.mp4'), dict(type='text', value='Question')]
    with pytest.raises(ValueError, match='scoped'):
        model.generate_inner(message, dataset='Video-MME')
    with pytest.raises(ValueError, match='exactly one video'):
        model.generate_inner([*message, dict(type='video', value='other.mp4')], dataset='densevideo')
    model.nframe = 16
    with pytest.raises(ValueError, match='overridden'):
        model.generate_inner(message, dataset='densevideo')


def test_model_custom_prompt_uses_dataset_frame_contract(backend):
    model = grt.GRT()

    class Dataset:
        dataset_name = 'dive_bench_educational_high_fps'

        def build_grt_prompt(self, row):
            assert row == {'index': 13}
            return ['known aligned video prompt']

    assert model.use_custom_prompt('dive_bench_educational_high_fps')
    with pytest.raises(ValueError, match='Educational'):
        model.use_custom_prompt('Video-MME')
    assert model.build_prompt({'index': 13}, Dataset()) == ['known aligned video prompt']


UNSUPPORTED_GRT_TASKS = (
    'dive_bench_high_motion_high_fps',
    'dive_bench_high_motion_high_fps_preview1000',
    'densevideo_highmotion',
    'Video-MME',
    None,
    ['densevideo'],
)


@pytest.mark.parametrize('profile', ['route31', 'qwen3', 'qwen7'])
@pytest.mark.parametrize('role', ['base', 'all', 'candidate'])
@pytest.mark.parametrize('task', UNSUPPORTED_GRT_TASKS)
def test_unsupported_task_rejected_before_prompt_or_generation(backend, profile, role, task):
    model = grt.GRT(profile=profile, role=role)

    class UnreadMessage:
        def __iter__(self):
            pytest.fail('An unsupported task must be rejected before reading model inputs')

    class Dataset:
        dataset_name = task

        def build_grt_prompt(self, row):
            pytest.fail('An unsupported task must not construct or decode a prompt')

    with pytest.raises(ValueError, match='Educational'):
        model.use_custom_prompt(task)
    with pytest.raises(ValueError, match='Educational'):
        model.build_prompt(object(), Dataset())
    with pytest.raises(ValueError, match='Educational'):
        model.generate_inner(UnreadMessage(), dataset=task)
    assert model.backend.requests == []
    assert model.backend.task_dict == {'old': {}}
    assert model._request_id == 0


@pytest.mark.parametrize('profile', ['route31', 'qwen3', 'qwen7'])
@pytest.mark.parametrize('role', ['base', 'all', 'candidate'])
@pytest.mark.parametrize('task', ['densevideo', 'dive_bench_educational_high_fps'])
def test_educational_aliases_preserve_all_profile_roles(backend, profile, role, task):
    model = grt.GRT(profile=profile, role=role)
    assert model.use_custom_prompt(task)
    message = [dict(type='video', value='clip.mp4'), dict(type='text', value='Question')]
    assert model.generate_inner(message, dataset=task) == 'answer'
    assert model.profile == profile
    assert model.role == role
    assert model.backend.task_dict == {'old': {}}
    assert len(model.backend.requests) == 1


@pytest.mark.parametrize('task', UNSUPPORTED_GRT_TASKS)
def test_runtime_task_binding_rejected_before_cuda_or_backend(backend, monkeypatch, task):
    valid_runtime = grt._load_runtime

    def wrong_task_runtime(profile, role):
        selected, *rest = valid_runtime(profile, role)
        return {'task': task, 'max_new_tokens': selected['max_new_tokens']}, *rest

    configure = mock.Mock()
    monkeypatch.setattr(grt, '_load_runtime', wrong_task_runtime)
    monkeypatch.setattr(grt, '_configure_reproduction', configure)
    with pytest.raises(ValueError, match='Educational'):
        grt.GRT()
    configure.assert_not_called()
    assert backend == []


@pytest.mark.parametrize('task', UNSUPPORTED_GRT_TASKS)
def test_invalid_profile_task_rejected_before_optional_model_import(monkeypatch, task):
    class Resource:
        def joinpath(self, name):
            assert name == 'profiles.json'
            return self

        def read_text(self):
            return json.dumps({
                'schema': 1,
                'source_commit': '78284318c9c8664df5fc6785410a0a9c4e494436',
                'profiles': {'route31': {'task': task}},
            })

    monkeypatch.setattr(grt.importlib.metadata, 'version', lambda name: '0.1.0')
    monkeypatch.setattr(grt.importlib.resources, 'files', lambda name: Resource())
    with mock.patch.object(grt.importlib, 'import_module') as import_model:
        with pytest.raises(ValueError, match='Educational'):
            grt._load_runtime('route31', 'candidate')
    import_model.assert_not_called()
