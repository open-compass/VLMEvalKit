"""Policy LLM creation through the real run.py entry point, using local doubles."""

import json
import sys
from ast import literal_eval
from copy import deepcopy

import pandas as pd
import pytest
from test_interactive_mt import ChatModel, GenerateModel, InteractiveDataset, SingleTurnDataset

import run as runner
from vlmeval.smp import load_prediction_file


def parse_args(monkeypatch, *options):
    monkeypatch.setattr(sys, 'argv', ['run.py', '--data', 'Example', '--model', 'mock', *options])
    return runner.parse_args()


def run_cli(monkeypatch, dataset, model, tmp_path):
    monkeypatch.setitem(runner.supported_VLM, 'mock', lambda: model)
    monkeypatch.setattr(runner, 'build_dataset_from_spec', lambda *a, **kw: dataset)
    monkeypatch.setattr(runner, 'RANK', 0)
    monkeypatch.setattr(runner, 'WORLD_SIZE', 1)
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('PRED_FORMAT', 'tsv')
    for name in ('MMEVAL_ROOT', 'FWD_API', 'EVAL_PROXY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(runner, 'setup_logger', lambda **kw: None)
    monkeypatch.setattr(runner, 'log_run_benchmark_report', lambda *a: None)
    runner.main()
    files = list(tmp_path.glob('mock/*/mock_Example.tsv'))
    assert len(files) == 1
    return [literal_eval(value) for value in load_prediction_file(str(files[0]))['prediction']]


@pytest.mark.parametrize('options, expected', [
    ([], {'model': 'default-policy', 'retry': 6, 'timeout': 600, 'verbose': False}),
    (['--policy-args', '{"model":"json-policy","temperature":0,"retry":2}'],
     {'model': 'json-policy', 'temperature': 0, 'retry': 2, 'timeout': 600, 'verbose': False}),
    (['--policy', 'cli-policy', '--policy-base-url', 'https://policy.invalid/v1/',
      '--policy-retry', '3', '--policy-timeout', '42',
      '--policy-args', '{"model":"json-policy","api_base":"unused"}'],
     {'model': 'cli-policy', 'api_base': 'https://policy.invalid/v1/chat/completions',
      'retry': 3, 'timeout': 42, 'verbose': False}),
])
def test_policy_kwargs_precedence(monkeypatch, options, expected):
    dataset = InteractiveDataset()
    dataset.DEFAULT_POLICY_MODEL = 'default-policy'
    args = parse_args(monkeypatch, *options, '--judge', 'separate-judge')
    assert runner.get_policy_kwargs(args, dataset=dataset) == expected
    assert runner.get_judge_kwargs(args, dataset=dataset)['model'] == 'separate-judge'


@pytest.mark.parametrize('value', ['[]', 'null', '"model"', '{invalid'])
def test_policy_args_must_be_json_object(monkeypatch, value):
    with pytest.raises(ValueError):
        runner.get_policy_kwargs(parse_args(monkeypatch, '--policy-args', value), dataset=None)


def test_policy_llm_hook_is_optional_and_eval_does_not_build_it(monkeypatch):
    dataset = InteractiveDataset()
    args = parse_args(monkeypatch)
    assert runner.build_policy_llm(args, dataset=dataset) is None
    dataset.build_policy_llm = lambda kwargs: pytest.fail('Eval must not build a policy LLM.')
    args.mode = 'eval'
    assert runner.build_policy_llm(args, dataset=dataset) is None


@pytest.mark.parametrize('backend', [
    'local_image', 'local_video', 'api_image', 'api_video',
    'pipeline_image', 'pipeline_video', 'local_image_mt_type', 'pipeline_image_mt_type',
])
def test_run_cli_builds_policy_llm_once_and_shares_it(tmp_path, monkeypatch, backend):
    dataset, model = InteractiveDataset(), ChatModel()
    if backend.endswith('video'):
        dataset.MODALITY = 'VIDEO'
    if backend.endswith('mt_type'):
        dataset.TYPE = 'MT'
    model.is_api = backend.startswith(('api', 'pipeline'))
    llm = object()
    builds = []

    def build_policy_llm(kwargs):
        builds.append(deepcopy(kwargs))
        return llm

    dataset.build_policy_llm = build_policy_llm
    dataset.DEFAULT_POLICY_MODEL = 'default-policy'
    policy_args = {'temperature': 0, 'extra_body': {'seed': 1}}
    parse_args(
        monkeypatch, '--mode', 'infer', '--work-dir', str(tmp_path), '--policy', 'cli-policy',
        '--policy-base-url', 'https://policy.invalid/v1',
        '--policy-args', json.dumps(policy_args), '--api-nproc', '2',
        '--monitor-interval', '1', '--debug',
        *(['--api-mode'] if backend.startswith('pipeline') else []),
    )
    predictions = run_cli(monkeypatch, dataset, model, tmp_path)

    assert builds == [{
        **policy_args, 'model': 'cli-policy',
        'api_base': 'https://policy.invalid/v1/chat/completions',
        'retry': 6, 'timeout': 600, 'verbose': False,
    }]
    assert len(dataset.policy_llms) == 2 and all(item is llm for item in dataset.policy_llms)
    assert predictions == [
        {'responses': ['answer 10', 'END'], 'error': None},
        {'responses': ['answer 20', 'END'], 'error': None},
    ]


@pytest.mark.parametrize('backend', ['local_image', 'api_image', 'pipeline_image'])
def test_run_cli_mmdu_uses_image_orchestration(tmp_path, monkeypatch, backend):
    from vlmeval import inference_api
    from vlmeval.dataset.image_mt import MMDUDataset

    dataset = MMDUDataset.__new__(MMDUDataset)
    dataset.dataset_name = 'Example'
    dataset.meta_only = True
    dataset.data = pd.DataFrame([{
        'index': 1, 'question': repr(['first<ImageHere>', 'second']),
        'image_path': repr(['one.png']), 'answer': repr(['private 1', 'private 2']),
    }])

    class Model(ChatModel):
        def chat(self, history, dataset):
            self.calls.append((deepcopy(history), dataset))
            return f'answer {len(self.calls)}'

    model = Model()
    model.is_api = backend != 'local_image'
    routes = []
    real_job = runner.infer_data_job
    real_config = inference_api.DatasetConfig

    def image_job(*args, **kwargs):
        routes.append('image')
        return real_job(*args, **kwargs)

    def api_config(**kwargs):
        routes.append(kwargs['dataset_type'])
        return real_config(**kwargs)

    monkeypatch.setattr(runner, 'infer_data_job', image_job)
    monkeypatch.setattr(inference_api, 'DatasetConfig', api_config)
    parse_args(
        monkeypatch, '--mode', 'infer', '--work-dir', str(tmp_path),
        '--api-nproc', '1', '--monitor-interval', '1', '--debug',
        *(['--api-mode'] if backend.startswith('pipeline') else []),
    )
    assert run_cli(monkeypatch, dataset, model, tmp_path) == [['answer 1', 'answer 2']]
    assert routes == ['image']
    assert [name for _, name in model.calls] == ['Example', 'Example']
    assert 'private' not in str(model.calls)


@pytest.mark.parametrize('backend', [
    'local_image', 'local_video', 'api_image', 'api_video', 'pipeline_image', 'pipeline_video',
])
@pytest.mark.parametrize('force_dataset', [False, True])
def test_run_cli_uses_same_custom_prompt_priority(tmp_path, monkeypatch, backend, force_dataset):
    dataset = SingleTurnDataset()
    dataset.force_use_dataset_prompt = force_dataset
    video = backend.endswith('video')
    if video:
        dataset.MODALITY = 'VIDEO'

    class Model(GenerateModel):
        def generate(self, message, dataset):
            return repr(message[0]['value'])

    model = Model()
    model.is_api = not backend.startswith('local')
    parse_args(
        monkeypatch, '--mode', 'infer', '--work-dir', str(tmp_path),
        '--api-nproc', '1', '--monitor-interval', '1', '--debug',
        *(['--api-mode'] if backend.startswith('pipeline') else []),
    )
    predictions = run_cli(monkeypatch, dataset, model, tmp_path)
    if force_dataset:
        assert predictions == [f'dataset {i} / {True if video else None}' for i in (10, 20)]
    else:
        assert predictions == ['model prompt', 'model prompt']
