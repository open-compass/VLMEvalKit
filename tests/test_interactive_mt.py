import asyncio
import importlib.util
import sys
from ast import literal_eval
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

# Warm up at module scope: the autouse fixture snapshots sys.modules and restores
# it on teardown, so an import first made inside a test is rolled back afterwards.
# Re-importing torch after that fails (its C extensions stay initialized), so any
# later import of vlmeval -- which imports torch -- would blow up.
import vlmeval.dataset.interactive_mt  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _restore_modules():
    with patch.dict(sys.modules):
        yield


def _load_interactive_mt():
    name = 'vlmeval.interactive_mt'
    spec = importlib.util.spec_from_file_location(name, ROOT / 'vlmeval/interactive_mt.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_inference_module(name):
    test_module = ROOT / 'tests/test_checkpoint_retry.py'
    spec = importlib.util.spec_from_file_location('test_checkpoint_retry_helpers', test_module)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    return helpers._load_inference_module(name)


def test_static_policy_preserves_initial_context_and_replays_all_user_turns():
    import vlmeval.interactive_mt as mt
    from vlmeval.dataset.interactive_mt import InteractiveMTMixin
    initial = [
        {'role': 'system', 'content': 'system context'},
        {'role': 'assistant', 'content': 'dataset context'},
        {'role': 'user', 'content': 'first question'},
    ]
    follow_up = {'role': 'user', 'content': 'second question'}

    class Dataset(InteractiveMTMixin):
        def build_cases(self, sample):
            return [sample]

        def build_prompt(self, sample):
            return deepcopy(initial)

        def build_policy(self, sample, messages):
            return mt.StaticTurnPolicy(messages, [follow_up])

    class Model:
        def __init__(self):
            self.histories = []
            self.responses = iter(['answer one', 'answer two'])

        def chat(self, history, dataset):
            self.histories.append(deepcopy(history))
            return next(self.responses)

    model = Model()
    prediction = mt.chat_interactive_mt(model, Dataset(), {'index': 1}, 'Example')

    assert prediction == ['answer one', 'answer two']
    assert model.histories == [
        initial,
        initial + [
            {'role': 'assistant', 'content': 'answer one'},
            follow_up,
        ],
    ]


@pytest.mark.parametrize('raises', [True, False])
def test_failure_is_finalized_and_stops_remaining_turns(raises):
    import vlmeval.interactive_mt as mt
    from vlmeval.dataset.interactive_mt import InteractiveMTMixin
    initial = [{'role': 'user', 'content': 'first question'}]
    follow_up = {'role': 'user', 'content': 'second question'}

    class Dataset(InteractiveMTMixin):
        def build_cases(self, sample):
            return [sample]

        def build_prompt(self, sample):
            return initial

        def build_policy(self, sample, messages):
            return mt.StaticTurnPolicy(messages, [follow_up])

    class Model:
        def __init__(self):
            self.calls = 0

        def chat(self, history, dataset):
            self.calls += 1
            if raises:
                raise RuntimeError('rate limited')
            return mt.FAIL_MSG + ' rate limited'

    model = Model()
    prediction = mt.chat_interactive_mt(model, Dataset(), {'index': 1}, 'Example')

    assert len(prediction) == 1
    assert prediction[0].startswith(mt.FAIL_MSG)
    assert 'rate limited' in prediction[0]
    assert model.calls == 1


def test_mmdu_static_policy_preserves_text_images_and_prediction_format():
    from vlmeval.dataset.image_mt import MMDUDataset
    from vlmeval.interactive_mt import chat_interactive_mt

    dataset = MMDUDataset.__new__(MMDUDataset)
    dataset.meta_only = True
    sample = {
        'question': repr(['first<ImageHere>between<ImageHere>end', 'second', 'third<ImageHere>']),
        'image_path': repr(['one.png', 'two.png', 'three.png']),
        'answer': repr(['private reference 1', 'private reference 2', 'private reference 3']),
    }
    first = {'role': 'user', 'content': [
        {'type': 'text', 'value': 'first'},
        {'type': 'image', 'value': 'one.png'},
        {'type': 'text', 'value': 'between'},
        {'type': 'image', 'value': 'two.png'},
        {'type': 'text', 'value': 'end'},
    ]}
    second = {'role': 'user', 'content': [{'type': 'text', 'value': 'second'}]}
    third = {'role': 'user', 'content': [
        {'type': 'text', 'value': 'third'}, {'type': 'image', 'value': 'three.png'},
    ]}

    class Model:
        def __init__(self):
            self.calls = []

        def chat(self, history, dataset):
            self.calls.append(deepcopy(history))
            return f'answer {len(self.calls)}'

    assert dataset.build_prompt(sample) == [first]
    model = Model()
    prediction = chat_interactive_mt(model, dataset, sample, 'MMDU')
    assert prediction == ['answer 1', 'answer 2', 'answer 3']
    assert model.calls == [
        [first],
        [first, {'role': 'assistant', 'content': 'answer 1'}, second],
        [first, {'role': 'assistant', 'content': 'answer 1'}, second,
         {'role': 'assistant', 'content': 'answer 2'}, third],
    ]
    assert 'private reference' not in str(model.calls)


def test_failed_api_checkpoint_retries_case_from_first_turn(tmp_path):
    from vlmeval.dataset.interactive_mt import InteractiveMTMixin
    module, modules = _load_inference_module('inference')
    from vlmeval.interactive_mt import StaticTurnPolicy

    smp_file = modules['vlmeval.smp.file']

    class Dataset(InteractiveMTMixin):
        dataset_name = 'Example'
        data = pd.DataFrame({'index': [0, 1]})

        def build_cases(self, row):
            return [row]

        def build_prompt(self, row):
            return [{'role': 'user', 'content': str(row['index'])}]

        def build_policy(self, row, messages):
            follow_up = [{'role': 'user', 'content': 'follow-up'}]
            return StaticTurnPolicy(messages, follow_up)

    class Model:
        is_api = True

        def __init__(self):
            self.calls = []
            self.responses = iter(['retried first turn', 'retried second turn'])

        def chat_inner(self, inputs, **kwargs):
            raise AssertionError('chat_inner is only an API-model capability check')

        def chat(self, history, dataset):
            self.calls.append(deepcopy(history))
            return next(self.responses)

    checkpoint = tmp_path / 'mock_Example_checkpoint.pkl'
    smp_file.dump({
        '0': ['completed case'],
        '1': ['earlier answer', module.FAIL_MSG + ' rate limited'],
    }, str(checkpoint))
    model = Model()

    results = module.infer_data_api(model, str(tmp_path), 'mock', Dataset())

    assert results == {
        0: ['completed case'],
        1: ['retried first turn', 'retried second turn'],
    }
    assert len(model.calls) == 2
    assert model.calls[0] == [{'role': 'user', 'content': '1'}]
    assert model.calls[1] == [
        {'role': 'user', 'content': '1'},
        {'role': 'assistant', 'content': 'retried first turn'},
        {'role': 'user', 'content': 'follow-up'},
    ]


@pytest.mark.parametrize('runner', ['local', 'api_pipeline'])
@pytest.mark.parametrize('retry_failed', [True, False])
def test_interactive_case_retry_and_finalized_prediction(
    tmp_path, monkeypatch, runner, retry_failed,
):
    from vlmeval.dataset.interactive_mt import InteractiveMTMixin
    name = 'inference' if runner == 'local' else 'inference_api'
    module, modules = _load_inference_module(name)
    from vlmeval.interactive_mt import Step, TurnPolicy

    smp_file = modules['vlmeval.smp.file']
    target = tmp_path / 'mock_Example.tsv'

    class Policy(TurnPolicy):
        max_turns = 5

        def __init__(self):
            super().__init__()
            self.state['responses'] = []

        def step(self, response, history):
            self.state['responses'].append(response)
            if response == 'END':
                return Step(done=True)
            return Step(messages=[{'role': 'user', 'content': 'follow-up for ' + response}])

        def finalize(self):
            return {
                'responses': self.state['responses'],
                'error': self.state.get('failure', {}).get('message'),
            }

    class Dataset(InteractiveMTMixin):
        dataset_name = 'Example'

        def __init__(self):
            self.data = pd.DataFrame({'index': [0, 1]})
            self.policies = []
            self.evaluated = []

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data.iloc[index]

        def dump_image(self, sample):
            return []

        def build_prompt(self, sample):
            # An interactive_mt MT dataset always supplies its own prompt; the
            # model's single-turn custom prompt must not replace it.
            return [
                {'role': 'system', 'content': 'initial context'},
                {'role': 'assistant', 'content': 'dataset example'},
                {'role': 'user', 'content': f'case {sample["index"]} first'},
            ]

        def build_policy(self, sample, messages):
            assert messages[-1]['content'] == f'case {sample["index"]} first'
            policy = Policy()
            self.policies.append(policy)
            return policy

        def evaluate(self, result_file, **kwargs):
            frame = smp_file.load_prediction_file(result_file)
            self.evaluated = [literal_eval(value) for value in frame['prediction']]
            return {'cases': len(self.evaluated)}

    class Model:
        is_api = False

        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        def set_dump_image(self, dump_image):
            self.dump_image = dump_image

        def use_custom_prompt(self, dataset):
            raise AssertionError('Interactive MT must not consult the model prompt.')

        def build_prompt(self, sample, dataset):
            raise AssertionError('Interactive MT must not use the model prompt.')

        def chat_inner(self, inputs, **kwargs):
            raise AssertionError('Only used for the MT capability check.')

        def chat(self, history, dataset):
            self.calls.append(deepcopy(history))
            return next(self.responses)

    if runner == 'local':
        monkeypatch.setattr(
            module.torch, 'cuda', SimpleNamespace(empty_cache=lambda: None), raising=False,
        )

    saved_checkpoints = []

    def run(model, dataset):
        if runner == 'local':
            module.infer_data_job(
                model, str(tmp_path), 'mock', dataset,
                result_file=str(target), retry_failed=retry_failed,
            )
            dataset.evaluate(str(target))
        else:
            cfg = module.DatasetConfig(
                dataset_name='Example', dataset_obj=dataset, model_name='mock',
                model_obj=model, work_dir=str(tmp_path), result_file=str(target),
                judge_kwargs={}, dataset_type='image', interactive_mt=True,
            )
            pipeline = module.APIEvalPipeline(
                [cfg], concurrency=1, debug=True, retry_failed=retry_failed,
            )
            monkeypatch.setattr(pipeline, '_create_symlinks', lambda _: None)
            save_checkpoint = pipeline._save_checkpoint

            def capture_checkpoint(alias, result):
                save_checkpoint(alias, result)
                saved_checkpoints.append(smp_file.load(str(pipeline._get_checkpoint_file(alias))))

            monkeypatch.setattr(pipeline, '_save_checkpoint', capture_checkpoint)

            async def run_pipeline():
                await asyncio.wait_for(pipeline.run(), timeout=10)

            asyncio.run(run_pipeline())
            assert cfg.eval_status == module.EvalStatus.Done

    error = module.FAIL_MSG + ' rate limited'
    first_model = Model(['case 0 answer', 'END', 'old case 1 answer', error])
    first_dataset = Dataset()
    run(first_model, first_dataset)

    completed = {'responses': ['case 0 answer', 'END'], 'error': None}
    failed = {'responses': ['old case 1 answer'], 'error': error}
    assert first_dataset.evaluated == [completed, failed]
    assert len(first_model.calls) == 4
    assert first_model.calls[3][-1] == {
        'role': 'user', 'content': 'follow-up for old case 1 answer',
    }
    # Failure reaches finalize without being appended to the model's history.
    failure_history = first_dataset.policies[1].state['failure']['history']
    assert error not in str(failure_history)
    if runner == 'api_pipeline':
        assert saved_checkpoints[-1] == {'0': completed, '1': failed}

    second_model = Model(['new case 1 answer', 'END'])
    second_dataset = Dataset()
    run(second_model, second_dataset)

    if retry_failed:
        assert len(second_model.calls) == 2
        assert second_model.calls[0] == [
            {'role': 'system', 'content': 'initial context'},
            {'role': 'assistant', 'content': 'dataset example'},
            {'role': 'user', 'content': 'case 1 first'},
        ]
        assert second_model.calls[1][-1] == {
            'role': 'user', 'content': 'follow-up for new case 1 answer',
        }
        assert len(second_dataset.policies) == 1
        assert second_dataset.evaluated == [
            completed, {'responses': ['new case 1 answer', 'END'], 'error': None},
        ]
    else:
        assert second_model.calls == []
        assert second_dataset.policies == []
        assert second_dataset.evaluated == [completed, failed]
