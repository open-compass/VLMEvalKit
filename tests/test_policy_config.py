"""Runtime MT policy configuration, including the real run.py inference paths."""

import json
import sys
from ast import literal_eval
from copy import deepcopy

import pandas as pd
import pytest

import run as runner
from vlmeval.dataset.image_mt import MMDUDataset
from vlmeval.dataset.interactive_mt import InteractiveMTMixin
from vlmeval.interactive_mt import StaticTurnPolicy
from vlmeval.smp import load_prediction_file


def parse_args(monkeypatch, *options):
    monkeypatch.setattr(sys, 'argv', ['run.py', '--data', 'Example', '--model', 'mock', *options])
    return runner.parse_args()


@pytest.mark.parametrize('options, expected', [
    ([], {'model': 'default-policy', 'retry': 6, 'timeout': 600, 'verbose': False}),
    (['--policy-args', '{"model":"json-policy","temperature":0,"retry":2}'],
     {'model': 'json-policy', 'temperature': 0, 'retry': 2, 'timeout': 600, 'verbose': False}),
    (['--policy', 'cli-policy', '--policy-base-url', 'https://policy.invalid/v1/',
      '--policy-key', 'dummy-policy-key', '--policy-retry', '3', '--policy-timeout', '42',
      '--policy-args', '{"model":"json-policy","api_base":"unused","key":"unused"}'],
     {'model': 'cli-policy', 'api_base': 'https://policy.invalid/v1/chat/completions',
      'key': 'dummy-policy-key', 'retry': 3, 'timeout': 42, 'verbose': False}),
])
def test_policy_kwargs_precedence(monkeypatch, options, expected):
    class Dataset(MMDUDataset):
        DEFAULT_POLICY_MODEL = 'default-policy'

        def __init__(self):
            pass

        def evaluate(self, eval_file, **judge_kwargs):
            raise NotImplementedError

    args = parse_args(monkeypatch, *options, '--judge', 'separate-judge')
    assert runner.get_policy_kwargs(args, dataset=Dataset()) == expected
    assert runner.get_judge_kwargs(args, dataset=Dataset())['model'] == 'separate-judge'


@pytest.mark.parametrize('value', ['[]', 'null', '"model"', '{invalid'])
def test_policy_args_must_be_json_object(monkeypatch, value):
    args = parse_args(monkeypatch, '--policy-args', value)
    with pytest.raises(ValueError):
        runner.get_policy_kwargs(args, dataset=None)


def test_static_mt_does_not_require_policy_model(monkeypatch):
    dataset = MMDUDataset.__new__(MMDUDataset)
    dataset.meta_only = True
    sample = {'question': '["first", "second"]', 'image_path': '[]'}
    kwargs = runner.get_policy_kwargs(parse_args(monkeypatch), dataset=dataset)
    assert 'model' not in kwargs
    assert runner.build_policy_llm(parse_args(monkeypatch), dataset=dataset) is None
    policy = dataset.build_policy(sample, dataset.build_prompt(sample))
    assert isinstance(policy, StaticTurnPolicy)
    assert policy.max_turns == 2


@pytest.mark.parametrize('mode', ['local', 'api', 'api_pipeline'])
def test_run_cli_passes_policy_options_to_each_case(tmp_path, monkeypatch, mode):
    class Dataset(InteractiveMTMixin):
        MODALITY = 'IMAGE'   # supplied by the concrete dataset base in real code
        dataset_name = 'Example'
        DEFAULT_POLICY_MODEL = 'default-policy'
        DEFAULT_JUDGE_MODEL = 'default-judge'

        def __init__(self):
            self.data = pd.DataFrame({'index': [0, 1]})
            self.evaluated = []
            self.llm = object()
            self.llm_builds = []
            self.llms = []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data.iloc[index]

        def dump_image(self, sample):
            return []

        def build_prompt(self, sample):
            return [{'role': 'user', 'content': str(sample['index'])}]

        def build_policy_llm(self, policy_kwargs):
            self.llm_builds.append(deepcopy(policy_kwargs))
            return self.llm

        def build_policy(self, sample, messages, *, policy_llm=None):
            self.llms.append(policy_llm)
            return StaticTurnPolicy(messages, [{'role': 'user', 'content': 'next'}])

        def evaluate(self, result_file, **judge_kwargs):
            assert judge_kwargs['model'] == 'separate-judge'
            assert 'extra_body' not in judge_kwargs
            frame = load_prediction_file(result_file)
            self.evaluated = [literal_eval(value) for value in frame['prediction']]
            return {'cases': len(self.evaluated)}

    class Model:
        is_api = mode != 'local'

        def set_dump_image(self, dump_image):
            pass

        def chat_inner(self, *args, **kwargs):
            raise AssertionError('The MT runner must use chat().')

        def chat(self, history, dataset):
            assert dataset == 'Example'
            return 'answer'

    policy_args = {'model': 'json-policy', 'temperature': 0, 'extra_body': {'seed': 1}}
    parse_args(
        monkeypatch, '--work-dir', str(tmp_path), '--policy', 'cli-policy',
        '--policy-base-url', 'https://policy.invalid/v1', '--policy-key', 'dummy-policy-key',
        '--policy-args', json.dumps(policy_args), '--judge', 'separate-judge',
        '--api-nproc', '1', '--monitor-interval', '1', '--debug',
        *(['--api-mode'] if mode == 'api_pipeline' else []),
    )
    dataset = Dataset()
    monkeypatch.setitem(runner.supported_VLM, 'mock', Model)
    monkeypatch.setattr(runner, 'build_dataset_from_spec', lambda *a, **kw: dataset)
    monkeypatch.setattr(runner, 'RANK', 0)
    monkeypatch.setattr(runner, 'WORLD_SIZE', 1)
    monkeypatch.setenv('WORLD_SIZE', '1')
    monkeypatch.setenv('PRED_FORMAT', 'tsv')
    monkeypatch.delenv('MMEVAL_ROOT', raising=False)
    monkeypatch.delenv('FWD_API', raising=False)
    monkeypatch.delenv('EVAL_PROXY', raising=False)
    monkeypatch.setattr(runner, 'setup_logger', lambda **kw: None)
    monkeypatch.setattr(runner, 'log_run_benchmark_report', lambda *a: None)

    # Keep the production entry point, inference runners, checkpointing and evaluator.
    runner.main()

    expected = {
        **policy_args, 'model': 'cli-policy',
        'api_base': 'https://policy.invalid/v1/chat/completions', 'key': 'dummy-policy-key',
        'retry': 6, 'timeout': 600, 'verbose': False,
    }
    # The policy LLM is created once from the --policy* options and shared.
    assert dataset.llm_builds == [expected]
    assert len(dataset.llms) == 2 and all(llm is dataset.llm for llm in dataset.llms)
    assert dataset.evaluated == [['answer', 'answer'], ['answer', 'answer']]
    result_files = list(tmp_path.glob('mock/*/mock_Example.tsv'))
    assert len(result_files) == 1
    assert len(load_prediction_file(str(result_files[0]))) == 2
