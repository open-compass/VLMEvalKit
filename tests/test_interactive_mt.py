"""Per-sample runner contracts and backend integration without external calls."""

import asyncio
from ast import literal_eval
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from vlmeval import inference, inference_api, inference_video
from vlmeval.inference_runner import (InferenceRunner, InteractiveMTRunner, SingleTurnRunner,
                                      dispatch_sample, get_inference_runner)
from vlmeval.interactive_mt import FAIL_MSG, StaticTurnPolicy, Step, TurnPolicy
from vlmeval.smp import load, load_prediction_file


class AdaptivePolicy(TurnPolicy):
    max_turns = 3

    def __init__(self):
        super().__init__()
        self.state['responses'] = []

    def step(self, response, history):
        self.state['responses'].append(response)
        if response == 'END':
            return Step(done=True)
        return Step(messages=[{'role': 'user', 'content': 'follow-up for ' + response}])

    def finalize(self):
        return {'responses': self.state['responses'],
                'error': self.state.get('failure', {}).get('message')}


class InteractiveDataset:
    INFERENCE_RUNNER = InteractiveMTRunner
    dataset_name = 'Example'
    MODALITY = 'IMAGE'
    TYPE = 'VQA'
    nframe = 8
    fps = -1
    pack = False

    def __init__(self):
        self.data = pd.DataFrame({'index': [10, 20], 'question': ['first', 'second']})
        self.policies = []
        self.policy_llms = []
        self.prompt_calls = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, position):
        return self.data.iloc[position]

    def dump_image(self, sample):
        return []

    def get_default_judge_model(self, judge_kwargs=None):
        return None

    def build_prompt(self, sample):
        if isinstance(sample, int):
            sample = self.data.iloc[sample]
        self.prompt_calls.append(sample['index'])
        return [
            {'role': 'system', 'content': 'context'},
            {'role': 'user', 'content': str(sample['index'])},
        ]

    def build_policy(self, sample, messages, *, policy_llm=None):
        if isinstance(sample, int):
            sample = self.data.iloc[sample]
        assert messages[-1]['content'] == str(sample['index'])
        self.policy_llms.append(policy_llm)
        policy = AdaptivePolicy()
        self.policies.append(policy)
        return policy


class ChatModel:
    is_api = False
    VIDEO_LLM = True

    def __init__(self, fail_index=None):
        self.calls = []
        self.fail_index = fail_index

    def set_dump_image(self, dump_image):
        pass

    def use_custom_prompt(self, dataset):
        raise AssertionError('Interactive inputs must come from the dataset.')

    def chat(self, history, dataset):
        self.calls.append((deepcopy(history), dataset))
        if history[-1]['content'].startswith('follow-up'):
            return 'END'
        if history[-1]['content'] == self.fail_index:
            return FAIL_MSG + ' rate limited'
        return 'answer ' + history[-1]['content']


def prepared_dispatch(model, dataset, sample, **kwargs):
    runner = get_inference_runner(dataset)
    messages = runner.prepare_prompt(model, dataset, sample, 'Example')
    return dispatch_sample(
        model, dataset, sample, 'Example', message=messages, inference_runner=runner, **kwargs)


def test_run_consumes_prepared_prompt_and_defaults_to_one_case():
    dataset, model = InteractiveDataset(), ChatModel()
    sample = dataset[0]
    message = get_inference_runner(dataset).prepare_prompt(model, dataset, sample, 'Example')
    dataset.build_prompt = lambda sample: pytest.fail('run() must only consume prepared prompts.')
    prediction = dispatch_sample(model, dataset, sample, 'Example', message=message)
    assert prediction == {'responses': ['answer 10', 'END'], 'error': None}
    assert len(dataset.policies) == 1
    assert model.calls[1][0][-1] == {'role': 'user', 'content': 'follow-up for answer 10'}
    assert message[0].messages == [
        {'role': 'system', 'content': 'context'}, {'role': 'user', 'content': '10'}]


def test_cases_have_independent_policies_and_share_only_the_policy_llm():
    dataset, model = InteractiveDataset(), ChatModel()
    dataset.build_cases = lambda sample: [{'index': 1}, {'index': 2}]
    llm = object()
    result = prepared_dispatch(model, dataset, dataset[0], policy_llm=llm)
    assert result == [
        {'responses': ['answer 1', 'END'], 'error': None},
        {'responses': ['answer 2', 'END'], 'error': None},
    ]
    assert dataset.policy_llms == [llm, llm]
    assert dataset.policies[0].state is not dataset.policies[1].state
    assert 'answer 1' not in str(model.calls[2])
    assert vars(get_inference_runner(dataset)) == {}


@pytest.mark.parametrize('expand_cases', [False, True])
def test_interactive_hooks_receive_unchanged_case_samples(expand_cases):
    sample = object()
    cases = [object(), object()] if expand_cases else [sample]
    prompt_samples, policy_samples = [], []

    def build_prompt(case):
        prompt_samples.append(case)
        return [{'role': 'user', 'content': '10'}]

    def build_policy(case, messages):
        policy_samples.append(case)
        return AdaptivePolicy()

    dataset = SimpleNamespace(
        INFERENCE_RUNNER=InteractiveMTRunner, build_prompt=build_prompt, build_policy=build_policy)
    if expand_cases:
        def build_cases(value):
            assert value is sample
            return cases
        dataset.build_cases = build_cases
    result = prepared_dispatch(ChatModel(), dataset, sample)
    prediction = {'responses': ['answer 10', 'END'], 'error': None}
    assert result == ([prediction, prediction] if expand_cases else prediction)
    assert all(actual is expected for actual, expected in zip(prompt_samples, cases))
    assert all(actual is expected for actual, expected in zip(policy_samples, cases))
    assert len(prompt_samples) == len(policy_samples) == len(cases)


@pytest.mark.parametrize('value', ['InteractiveMTRunner', InteractiveMTRunner(), None])
def test_runner_must_be_a_class(value):
    dataset = InteractiveDataset()
    dataset.INFERENCE_RUNNER = value
    with pytest.raises(TypeError, match='INFERENCE_RUNNER must be an InferenceRunner class'):
        get_inference_runner(dataset)


def test_mt_metadata_does_not_implicitly_select_a_runner():
    dataset = SimpleNamespace(TYPE='MT')
    assert isinstance(get_inference_runner(dataset), SingleTurnRunner)


@pytest.mark.parametrize('prediction, single_turn_failed, interactive_failed', [
    ('answer', False, False),
    (FAIL_MSG + ' rate limited', True, True),
    ('Failed to obtain answer: RuntimeError', False, True),
    ({'cases': ['answer', FAIL_MSG + ' rate limited']}, True, True),
    ([{'responses': ['answer'], 'error': 'Fail to obtain answer'}], False, True),
    ({'responses': ['answer'], 'error': None}, False, False),
])
def test_runner_failure_rules_handle_saved_predictions(
    prediction, single_turn_failed, interactive_failed,
):
    original = deepcopy(prediction)
    assert SingleTurnRunner().is_failed(prediction) is single_turn_failed
    assert InteractiveMTRunner().is_failed(prediction) is interactive_failed
    assert prediction == original


@pytest.mark.parametrize('hook', ['build_prompt', 'build_policy'])
def test_required_hooks_fail_before_any_inference(hook, tmp_path):
    dataset, model = InteractiveDataset(), ChatModel()
    setattr(dataset, hook, None)
    with pytest.raises(TypeError, match=hook):
        inference.infer_data(model, 'mock', str(tmp_path), dataset, str(tmp_path / 'part.pkl'))
    assert model.calls == []


@pytest.mark.parametrize('hook', ['build_cases', 'build_policy_llm'])
def test_invalid_optional_hooks_are_not_silently_ignored(hook):
    dataset = InteractiveDataset()
    setattr(dataset, hook, 'invalid')
    with pytest.raises(TypeError, match=hook):
        get_inference_runner(dataset)


def test_empty_cases_and_invalid_initial_context_fail_before_chat():
    dataset, model = InteractiveDataset(), ChatModel()
    dataset.build_cases = lambda sample: []
    with pytest.raises(ValueError, match='at least one case'):
        prepared_dispatch(model, dataset, dataset[0])
    del dataset.build_cases
    dataset.build_prompt = lambda sample: [{'role': 'assistant', 'content': 'bad'}]
    with pytest.raises(ValueError, match='end with a user'):
        prepared_dispatch(model, dataset, dataset[0])
    assert model.calls == []


@pytest.mark.parametrize('raises', [False, True])
def test_model_failure_finalizes_policy_and_stops_turns(raises):
    dataset, model = InteractiveDataset(), ChatModel(fail_index='10')
    if raises:
        def chat(history, dataset):
            raise RuntimeError('rate limited')
        model.chat = chat
    prediction = prepared_dispatch(model, dataset, dataset[0])
    assert prediction['responses'] == []
    assert 'rate limited' in prediction['error']
    failure = dataset.policies[0].state['failure']
    assert failure['reason'] == 'model'
    assert FAIL_MSG not in str(failure['history'])


BACKENDS = ['local_image', 'local_video', 'api_image', 'api_video',
            'pipeline_image', 'pipeline_video']


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('retry_failed', [True, False])
def test_backends_preserve_predictions_checkpoints_and_case_retry(tmp_path, backend, retry_failed):
    dataset, model = InteractiveDataset(), ChatModel(fail_index='20')
    target = tmp_path / 'mock_Example.tsv'
    is_video = backend.endswith('video')
    if is_video:
        dataset.MODALITY = 'VIDEO'

    def execute(ds, mdl):
        if backend.startswith('pipeline'):
            cfg = inference_api.DatasetConfig(
                dataset_name='Example', dataset_obj=ds, model_name='mock', model_obj=mdl,
                work_dir=str(tmp_path), result_file=str(target), judge_kwargs={},
                dataset_type='video' if is_video else 'image')
            pipeline = inference_api.APIEvalPipeline(
                [cfg], concurrency=2, run_eval=False, retry_failed=retry_failed)
            asyncio.run(asyncio.wait_for(pipeline.run(), timeout=15))
            assert cfg.processed == 2
        else:
            mdl.is_api = backend.startswith('api')
            module = inference_video if is_video else inference
            job = module.infer_data_job_video if is_video else module.infer_data_job
            job(mdl, str(tmp_path), 'mock', ds, result_file=str(target),
                retry_failed=retry_failed, api_nproc=2)
        return [literal_eval(value) for value in load_prediction_file(str(target))['prediction']]

    first = execute(dataset, model)
    assert first[0] == {'responses': ['answer 10', 'END'], 'error': None}
    assert first[1]['responses'] == [] and FAIL_MSG in first[1]['error']
    second_model = ChatModel()
    second = execute(InteractiveDataset(), second_model)
    assert second[0] == first[0]
    if retry_failed:
        assert second[1] == {'responses': ['answer 20', 'END'], 'error': None}
        assert len(second_model.calls) == 2
        assert second_model.calls[0][0][-1]['content'] == '20'
    else:
        assert second == first
        assert second_model.calls == []


class SingleTurnDataset(InteractiveDataset):
    INFERENCE_RUNNER = SingleTurnRunner

    def build_prompt(self, sample, video_llm=None):
        if isinstance(sample, int):
            sample = self.data.iloc[sample]
        index = sample if isinstance(sample, str) else sample['index']
        return [{'type': 'text', 'value': f'dataset {index} / {video_llm}'}]


class GenerateModel:
    is_api = False
    VIDEO_LLM = True

    def __init__(self):
        self.custom_calls = []
        self.prompt_calls = []
        self.calls = []

    def set_dump_image(self, dump_image):
        pass

    def use_custom_prompt(self, dataset):
        self.custom_calls.append(dataset)
        return True

    def build_prompt(self, sample, **kwargs):
        self.prompt_calls.append((sample, kwargs))
        return [{'type': 'text', 'value': 'model prompt'}]

    def generate(self, message, dataset):
        self.calls.append((message, dataset))
        return 'answer'


@pytest.mark.parametrize('backend, cache_source', [
    *[(backend, 'prediction') for backend in BACKENDS],
    *[(backend, 'checkpoint') for backend in BACKENDS if not backend.startswith('local')],
])
@pytest.mark.parametrize('retry_failed', [False, True])
def test_backends_use_runner_failure_predicate_for_reuse(
    tmp_path, backend, cache_source, retry_failed,
):
    from vlmeval.smp import dump

    class CustomFailureRunner(SingleTurnRunner):
        def is_failed(self, prediction):
            return prediction == 'retry requested'

    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.INFERENCE_RUNNER = CustomFailureRunner
    model.use_custom_prompt = lambda name: False
    video = backend.endswith('video')
    if video:
        dataset.MODALITY = 'VIDEO'
    model.is_api = not backend.startswith('local')
    target = tmp_path / 'mock_Example.tsv'
    if cache_source == 'prediction':
        frame = dataset.data.copy()
        frame['prediction'] = ['cached', 'retry requested']
        dump(frame, str(target))
    else:
        suffix = '_8frame_nopack' if backend == 'api_video' else ''
        checkpoint = tmp_path / f'mock_Example{suffix}_checkpoint.pkl'
        dump({'10': 'cached', '20': 'retry requested'}, str(checkpoint))

    if backend.startswith('pipeline'):
        cfg = inference_api.DatasetConfig(
            dataset_name='Example', dataset_obj=dataset, model_name='mock', model_obj=model,
            work_dir=str(tmp_path), result_file=str(target), judge_kwargs={},
            dataset_type='video' if video else 'image')
        pipeline = inference_api.APIEvalPipeline(
            [cfg], concurrency=1, run_eval=False, retry_failed=retry_failed)
        asyncio.run(asyncio.wait_for(pipeline.run(), timeout=15))
    else:
        module = inference_video if video else inference
        job = module.infer_data_job_video if video else module.infer_data_job
        job(model, str(tmp_path), 'mock', dataset, result_file=str(target),
            retry_failed=retry_failed, api_nproc=1)

    expected = ['cached', 'answer' if retry_failed else 'retry requested']
    assert list(load_prediction_file(str(target))['prediction']) == expected
    assert len(model.calls) == int(retry_failed)
    if retry_failed:
        assert model.calls[0][0] == [
            {'type': 'text', 'value': f'dataset 20 / {True if video else None}'}]


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('force_dataset', [True, False])
@pytest.mark.parametrize('use_custom', [True, False])
def test_single_turn_unifies_prompt_selection_and_keeps_video_arguments(
    tmp_path, backend, force_dataset, use_custom,
):
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.data = dataset.data.iloc[:1].copy()
    dataset.force_use_dataset_prompt = force_dataset
    model.use_custom_prompt = lambda name: use_custom
    video = backend.endswith('video')
    if video:
        dataset.MODALITY = 'VIDEO'
    else:
        dataset_prompt = dataset.build_prompt
        dataset.build_prompt = lambda sample: dataset_prompt(sample)
    real_build_prompt = model.build_prompt

    def build_prompt(sample, **kwargs):
        if video and not backend.startswith('local'):
            assert sample == 0
        else:
            assert sample['index'] == 10
        if video:
            assert kwargs == {'dataset': dataset, 'video_llm': True}
        else:
            assert kwargs == {'dataset': 'Logical'}
        return real_build_prompt(sample, **kwargs)

    model.build_prompt = build_prompt
    target = tmp_path / 'out.tsv'
    if backend.startswith('pipeline'):
        cfg = inference_api.DatasetConfig(
            dataset_name='Logical', dataset_obj=dataset, model_name='mock', model_obj=model,
            work_dir=str(tmp_path), result_file=str(target), judge_kwargs={},
            dataset_type='video' if video else 'image')
        pipeline = inference_api.APIEvalPipeline([cfg], concurrency=1, run_eval=False)
        asyncio.run(asyncio.wait_for(pipeline.run(), timeout=15))
    else:
        model.is_api = backend.startswith('api')
        module = inference_video if video else inference
        job = module.infer_data_job_video if video else module.infer_data_job
        job(model, str(tmp_path), 'mock', dataset, result_file=str(target),
            dataset_name='Logical', api_nproc=1)
    expected_custom = use_custom and not force_dataset
    if expected_custom:
        assert model.calls[0][0] == [{'type': 'text', 'value': 'model prompt'}]
    else:
        assert model.calls[0][0] == [
            {'type': 'text', 'value': f'dataset 10 / {True if video else None}'}]
    assert model.calls[0][1] == 'Logical'
    assert list(load_prediction_file(str(target))['prediction']) == ['answer']


@pytest.mark.parametrize('backend', BACKENDS)
def test_backends_use_runner_instance_for_prompt_preparation(tmp_path, backend):
    class PreparedRunner(SingleTurnRunner):
        def prepare_prompt(self, model, dataset, sample, dataset_name, **prompt_options):
            index = sample if isinstance(sample, int) else sample['index']
            return [{'type': 'text', 'value': f'prepared {index} for {dataset_name}'}]

        def run(self, model, dataset, sample, dataset_name, *, message, policy_llm=None):
            index = sample if isinstance(sample, int) else sample['index']
            assert message[0]['value'] == f'prepared {index} for {dataset_name}'
            return super().run(model, dataset, sample, dataset_name, message=message)

    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.INFERENCE_RUNNER = PreparedRunner
    dataset.build_prompt = lambda *a, **kw: pytest.fail('Runner override must select the prompt.')
    model.use_custom_prompt = lambda name: pytest.fail('Default prompt selection must be bypassed')
    target = tmp_path / 'prepared.tsv'
    video = backend.endswith('video')
    if video:
        dataset.MODALITY = 'VIDEO'
    if backend.startswith('pipeline'):
        cfg = inference_api.DatasetConfig(
            dataset_name='Logical', dataset_obj=dataset, model_name='mock', model_obj=model,
            work_dir=str(tmp_path), result_file=str(target), judge_kwargs={},
            dataset_type='video' if video else 'image')
        pipeline = inference_api.APIEvalPipeline([cfg], concurrency=1, run_eval=False)
        asyncio.run(asyncio.wait_for(pipeline.run(), timeout=15))
    else:
        model.is_api = backend.startswith('api')
        module = inference_video if video else inference
        job = module.infer_data_job_video if video else module.infer_data_job
        job(model, str(tmp_path), 'mock', dataset, result_file=str(target),
            dataset_name='Logical', api_nproc=1)
    indices = (0, 1) if video and not backend.startswith('local') else (10, 20)
    assert model.calls == [
        ([{'type': 'text', 'value': f'prepared {index} for Logical'}], 'Logical')
        for index in indices]
    assert list(load_prediction_file(str(target))['prediction']) == ['answer', 'answer']


@pytest.mark.parametrize('backend', ['local_video', 'api_video', 'pipeline_video'])
def test_video_custom_prompt_accepts_standard_model_builder_signature(tmp_path, backend):
    class StandardModel(GenerateModel):
        def build_prompt(self, sample, dataset):
            assert dataset == 'Logical'
            index = sample if isinstance(sample, int) else sample['index']
            return [{'type': 'text', 'value': f'custom {index}'}]

    dataset, model = SingleTurnDataset(), StandardModel()
    dataset.MODALITY = 'VIDEO'
    target = tmp_path / 'standard.tsv'
    if backend.startswith('pipeline'):
        cfg = inference_api.DatasetConfig(
            dataset_name='Logical', dataset_obj=dataset, model_name='mock', model_obj=model,
            work_dir=str(tmp_path), result_file=str(target), judge_kwargs={}, dataset_type='video')
        pipeline = inference_api.APIEvalPipeline([cfg], concurrency=1, run_eval=False)
        asyncio.run(asyncio.wait_for(pipeline.run(), timeout=15))
    else:
        model.is_api = backend.startswith('api')
        inference_video.infer_data_job_video(
            model, str(tmp_path), 'mock', dataset, result_file=str(target),
            dataset_name='Logical', api_nproc=1)
    indices = (10, 20) if backend.startswith('local') else (0, 1)
    assert [message[0]['value'] for message, _ in model.calls] == [f'custom {i}' for i in indices]


def test_custom_prompt_builder_errors_are_not_retried_with_another_signature():
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.MODALITY = 'VIDEO'
    calls = []

    def build_prompt(sample, dataset, video_llm):
        calls.append(True)
        raise TypeError('error inside builder')

    model.build_prompt = build_prompt
    with pytest.raises(TypeError, match='error inside builder'):
        get_inference_runner(dataset).prepare_prompt(model, dataset, dataset[0], 'Example')
    assert calls == [True]


@pytest.mark.parametrize('video', [False, True])
@pytest.mark.parametrize('skip', [False, True])
def test_single_turn_skip_err_behavior_is_unchanged(tmp_path, monkeypatch, video, skip):
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.data = dataset.data.iloc[:1].copy()
    if video:
        dataset.MODALITY = 'VIDEO'
    model.generate = lambda **kwargs: (_ for _ in ()).throw(RuntimeError('out of memory'))
    monkeypatch.setenv('SKIP_ERR', '1' if skip else '0')
    synchronized = []
    monkeypatch.setattr(inference.torch.cuda, 'synchronize', lambda: synchronized.append(True))
    module = inference_video if video else inference
    args = (model, 'mock', str(tmp_path), dataset, str(tmp_path / 'part.pkl'))
    if skip:
        with pytest.warns(UserWarning, match='out of memory'):
            module.infer_data(*args)
        assert 'Failed to obtain answer:' in load(str(tmp_path / 'part.pkl'))[10]
        assert synchronized == [True]
    else:
        with pytest.raises(RuntimeError, match='out of memory'):
            module.infer_data(*args)
        assert synchronized == []


def test_mmdu_explicit_runner_preserves_media_and_static_turns():
    from vlmeval.dataset.image_base import ImageBaseDataset
    from vlmeval.dataset.image_mt import MMDUDataset
    assert MMDUDataset.__bases__ == (ImageBaseDataset,)
    dataset = MMDUDataset.__new__(MMDUDataset)
    dataset.meta_only = True
    sample = {
        'index': 1, 'question': repr(['first<ImageHere>end', 'second<ImageHere>']),
        'image_path': repr(['one.png', 'two.png']), 'answer': repr(['private 1', 'private 2']),
    }

    class Model:
        def __init__(self):
            self.calls = []

        def chat(self, history, dataset):
            self.calls.append(deepcopy(history))
            return f'answer {len(self.calls)}'

    model = Model()
    assert isinstance(get_inference_runner(dataset), InteractiveMTRunner)
    assert prepared_dispatch(model, dataset, sample) == ['answer 1', 'answer 2']
    assert model.calls[0] == [{'role': 'user', 'content': [
        {'type': 'text', 'value': 'first'}, {'type': 'image', 'value': 'one.png'},
        {'type': 'text', 'value': 'end'}]}]
    assert model.calls[1][1:] == [
        {'role': 'assistant', 'content': 'answer 1'},
        {'role': 'user', 'content': [
            {'type': 'text', 'value': 'second'}, {'type': 'image', 'value': 'two.png'}]},
    ]
    assert 'private' not in str(model.calls)


def test_static_policy_keeps_complete_initial_context_and_failure_slot():
    initial = [{'role': 'system', 'content': 's'},
               {'role': 'assistant', 'content': 'example'},
               {'role': 'user', 'content': 'q'}]
    policy = StaticTurnPolicy(initial, [{'role': 'user', 'content': 'next'}])
    assert policy.step('first', initial).messages == [{'role': 'user', 'content': 'next'}]
    policy.on_failure(FAIL_MSG, initial, 1)
    assert policy.finalize() == ['first', FAIL_MSG]
    assert policy.initial_messages == initial


def test_custom_strategy_uses_the_same_dispatch_contract():
    class CustomRunner(InferenceRunner):
        @classmethod
        def validate_dataset(cls, dataset):
            if not isinstance(dataset.sample_key, str):
                raise TypeError('CustomRunner requires a string sample_key.')

        def prepare_prompt(self, model, dataset, sample, dataset_name, *, video_llm=None):
            return dataset.build_prompt(sample)

        def run(self, model, dataset, sample, dataset_name, *, message, policy_llm=None):
            return {'index': sample[dataset.sample_key], 'input': message, 'name': dataset_name}

        def is_failed(self, prediction):
            return False

    dataset = SimpleNamespace(INFERENCE_RUNNER=CustomRunner, sample_key='index',
                              build_prompt=lambda sample: 'ready')
    runner = get_inference_runner(dataset)
    sample = {'index': 7}
    message = runner.prepare_prompt(None, dataset, sample, 'Custom')
    assert dispatch_sample(None, dataset, sample, 'Custom', message=message) == {
        'index': 7, 'input': 'ready', 'name': 'Custom'}
    dataset.sample_key = None
    with pytest.raises(TypeError, match='CustomRunner requires a string sample_key'):
        get_inference_runner(dataset)


def test_required_hook_signature_is_validated_before_chat():
    dataset, model = InteractiveDataset(), ChatModel()
    dataset.build_policy = lambda sample: AdaptivePolicy()
    with pytest.raises(TypeError, match='build_policy has an invalid hook signature'):
        prepared_dispatch(model, dataset, dataset[0])
    assert model.calls == []


def test_policy_history_reset_sampling_and_next_call_kwargs():
    class Policy(AdaptivePolicy):
        def step(self, response, history):
            self.state['responses'].append(response)
            if len(self.state['responses']) == 2:
                return Step(done=True)
            return Step(history=[{'role': 'user', 'content': 'reset'}],
                        num_samples=2, chat_kwargs={'temperature': 0.2})

        def on_samples(self, candidates, history):
            self.state['candidates'] = candidates
            assert history == [{'role': 'user', 'content': 'reset'}]
            return Step(history=[{'role': 'user', 'content': 'chosen'}],
                        chat_kwargs={'max_tokens': 8})

    class Model:
        def __init__(self):
            self.calls = []

        def chat(self, history, dataset, **kwargs):
            self.calls.append((deepcopy(history), kwargs))
            return 'response ' + str(len(self.calls))

    dataset, model = InteractiveDataset(), Model()
    policy = Policy()
    dataset.build_policy = lambda sample, messages: policy
    result = prepared_dispatch(model, dataset, dataset[0])
    assert result == {'responses': ['response 1', 'response 4'], 'error': None}
    assert policy.state['candidates'] == ['response 2', 'response 3']
    assert [kwargs for _, kwargs in model.calls] == [
        {}, {'temperature': 0.2}, {'temperature': 0.2}, {'max_tokens': 8}]
    assert model.calls[-1][0] == [{'role': 'user', 'content': 'chosen'}]


@pytest.mark.parametrize('policy_value', [None, SimpleNamespace(max_turns=1), AdaptivePolicy()])
def test_invalid_policy_fails_before_model_call(policy_value):
    dataset, model = InteractiveDataset(), ChatModel()
    if isinstance(policy_value, TurnPolicy):
        policy_value.max_turns = 0
    dataset.build_policy = lambda sample, messages: policy_value
    with pytest.raises((TypeError, ValueError), match='TurnPolicy|at least one turn'):
        prepared_dispatch(model, dataset, dataset[0])
    assert model.calls == []


def test_direct_video_api_keeps_pack_prompt_and_checkpoint(tmp_path):
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.MODALITY = 'VIDEO'
    dataset.force_use_dataset_prompt = True
    dataset.pack = True
    dataset.data['video'] = ['clip', 'clip']
    model.is_api = True
    first = inference_video.infer_data_api(
        model, str(tmp_path), 'mock', dataset, samples_dict={'clip': 'clip'}, api_nproc=1)
    assert first == {'clip': 'answer'}
    assert model.calls[0][0][0]['value'] == 'dataset clip / True'
    model.calls.clear()
    second = inference_video.infer_data_api(
        model, str(tmp_path), 'mock', dataset, samples_dict={'clip': 'clip'}, api_nproc=1)
    assert second == first and model.calls == []
    assert (tmp_path / 'mock_Example_8frame_pack_checkpoint.pkl').exists()


@pytest.mark.parametrize('pack', [False, True])
@pytest.mark.parametrize('use_custom', [False, True])
def test_direct_video_api_forwards_samples_without_accessing_dataset_rows(
    tmp_path, pack, use_custom,
):
    sample = 'clip' if pack else 3
    prompt_calls = []

    def build_prompt(value, video_llm):
        prompt_calls.append(value)
        assert value is sample
        assert video_llm is True
        return [{'type': 'text', 'value': 'unchanged sample'}]

    # No data or __getitem__: the builder owns interpretation of its input.
    dataset = SimpleNamespace(dataset_name='Example', MODALITY='VIDEO',
                              nframe=8, fps=-1, pack=pack, build_prompt=build_prompt)
    model = GenerateModel()
    model.is_api = True
    model.use_custom_prompt = lambda name: use_custom

    def custom_prompt(value, dataset, video_llm):
        return build_prompt(value, video_llm)

    model.build_prompt = custom_prompt
    result = inference_video.infer_data_api(
        model, str(tmp_path), 'mock', dataset, samples_dict={97: sample}, api_nproc=1)
    assert result == {97: 'answer'}
    assert prompt_calls == [sample]
    assert model.calls[0][0] == [{'type': 'text', 'value': 'unchanged sample'}]


@pytest.mark.parametrize('force_dataset', [False, True])
def test_video_pipeline_keeps_explicit_video_llm_override(tmp_path, force_dataset):
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.MODALITY = 'VIDEO'
    dataset.force_use_dataset_prompt = force_dataset

    def build_prompt(sample, dataset, video_llm):
        assert video_llm is False
        return [{'type': 'text', 'value': 'custom video=False'}]

    model.build_prompt = build_prompt
    target = tmp_path / 'override.tsv'
    cfg = inference_api.DatasetConfig(
        dataset_name='Example', dataset_obj=dataset, model_name='mock', model_obj=model,
        work_dir=str(tmp_path), result_file=str(target), judge_kwargs={},
        dataset_type='video', video_llm=False)
    pipeline = inference_api.APIEvalPipeline([cfg], concurrency=1, run_eval=False)
    asyncio.run(asyncio.wait_for(pipeline.run(), timeout=15))
    expected = ([f'dataset {i} / False' for i in (10, 20)] if force_dataset
                else ['custom video=False'] * 2)
    assert [message[0]['value'] for message, _ in model.calls] == expected


def test_local_video_keeps_sub_dataset_and_custom_prompt_frame_requirement(tmp_path):
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.MODALITY = 'VIDEO'
    dataset.data = dataset.data.iloc[:1].copy()
    dataset.data['SUB_DATASET'] = 'ChildDataset'
    inference_video.infer_data(model, 'mock', str(tmp_path), dataset, str(tmp_path / 'ok.pkl'))
    assert model.calls[0][1] == 'ChildDataset'
    assert model.custom_calls == ['ChildDataset']
    dataset.nframe = 0
    with pytest.raises(ValueError, match='nframe must be set for custom prompt'):
        inference_video.infer_data(
            model, 'mock', str(tmp_path), dataset, str(tmp_path / 'bad.pkl'))


def test_single_turn_keeps_structured_prediction_and_extra_records(tmp_path):
    dataset, model = SingleTurnDataset(), GenerateModel()
    dataset.data = dataset.data.iloc[:1].copy()
    model.generate = lambda **kwargs: {'prediction': 'answer', 'extra_records': 'trace'}
    target = tmp_path / 'structured.tsv'
    inference.infer_data_job(model, str(tmp_path), 'mock', dataset, result_file=str(target))
    frame = load_prediction_file(str(target))
    assert list(frame['prediction']) == ['answer']
    assert list(frame['extra_records']) == ['trace']


@pytest.mark.parametrize('video', [False, True])
@pytest.mark.parametrize('retry_failed', [False, True])
def test_direct_api_reuses_checkpoint_and_retries_whole_failed_sample(
    tmp_path, video, retry_failed,
):
    from vlmeval.smp import dump

    dataset, model = InteractiveDataset(), ChatModel()
    model.is_api = True
    success = {'responses': ['cached', 'END'], 'error': None}
    failed = {'responses': ['earlier turn'], 'error': FAIL_MSG + ' rate limited'}
    suffix = '_8frame_nopack' if video else ''
    checkpoint = tmp_path / f'mock_Example{suffix}_checkpoint.pkl'
    dump({'10': success, '20': failed}, str(checkpoint))
    if video:
        result = inference_video.infer_data_api(
            model, str(tmp_path), 'mock', dataset, samples_dict={10: 0, 20: 1},
            retry_failed=retry_failed, api_nproc=1)
    else:
        result = inference.infer_data_api(
            model, str(tmp_path), 'mock', dataset, retry_failed=retry_failed, api_nproc=1)
    assert result[10] == success
    if retry_failed:
        assert result[20] == {'responses': ['answer 20', 'END'], 'error': None}
        assert len(model.calls) == 2
        assert model.calls[0][0] == [
            {'role': 'system', 'content': 'context'}, {'role': 'user', 'content': '20'}]
    else:
        assert result[20] == failed
        assert model.calls == []
    assert load(str(checkpoint)) == {str(index): value for index, value in result.items()}
