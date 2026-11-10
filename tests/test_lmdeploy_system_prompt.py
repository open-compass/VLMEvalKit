import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vlmeval.api.base import BaseAPI
from vlmeval.api.gemini import GeminiWrapper
from vlmeval.api.gpt import GPTAudio
from vlmeval.api.lmdeploy import LMDeployWrapper
from vlmeval.api.rbdashmm_chat3_5_api import RBdashMMChat3_5_38B_Wrapper, RBdashMMChat3_78B_Wrapper
from vlmeval.api.rbdashmm_chat3_api import RBdashMMChat3_5_Wrapper, RBdashMMChat3Wrapper
from vlmeval.api.taichu import TaichuVLRWrapper, TaichuVLWrapper
from vlmeval.api.telemm import TeleMM2_Wrapper
from vlmeval.api.telemm_thinking import TeleMM2Thinking_Wrapper


def test_all_api_generate_inner_methods_accept_system_prompt():
    missing = []
    api_root = Path(__file__).parents[1] / 'vlmeval' / 'api'

    for path in api_root.rglob('*.py'):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != 'generate_inner':
                continue
            args = [arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)]
            if 'system_prompt' not in args:
                missing.append(f'{path.relative_to(api_root)}:{node.lineno}')

    assert missing == []


def test_preprocess_message_uses_system_role_when_model_prompt_is_empty():
    model = BaseAPI(system_prompt=None)
    message = [
        {'role': 'system', 'type': 'text', 'value': 'Request prompt.'},
        {'role': 'user', 'type': 'text', 'value': 'Question'},
    ]

    processed, system_prompt = model.preprocess_message_with_role(message)

    assert processed == [{'type': 'text', 'value': 'Question'}]
    assert system_prompt == 'Request prompt.'
    assert model.system_prompt is None
    assert message[0]['role'] == 'system'


def test_gpt_audio_preserves_request_system_prompt_through_normalization(monkeypatch):
    model = GPTAudio.__new__(GPTAudio)
    BaseAPI.__init__(model, retry=1, wait=0, system_prompt='Configured prompt.', verbose=False)
    model.audio_cache_dir = None
    model.audio_sample_rate = 'auto'
    model.audio_channels = 'auto'
    model.audio_max_file_size = 1024
    model.audio_max_payload_size = 1024
    model.generate_inner = Mock(return_value=(0, 'ok', ''))
    payload = SimpleNamespace(data=b'audio', format='wav')
    normalize = Mock(return_value=payload)
    monkeypatch.setattr('vlmeval.api.gpt.normalize_audio', normalize)
    monkeypatch.setattr('vlmeval.api.gpt.resolve_media_source', lambda value, **kwargs: value)
    monkeypatch.setattr('vlmeval.api.gpt.parse_file', lambda value: (None, value))
    monkeypatch.setattr('vlmeval.api.base.time.sleep', lambda _: None)
    message = [
        {'role': 'system', 'type': 'text', 'value': 'Request prompt.'},
        {'role': 'user', 'type': 'audio', 'value': 'sample.wav'},
        {'role': 'user', 'type': 'text', 'value': 'Question'},
    ]
    original = [item.copy() for item in message]

    assert model.generate(message) == 'ok'

    assert model.generate_inner.call_args.kwargs['system_prompt'] == 'Configured prompt.\nRequest prompt.'
    content = model.generate_inner.call_args.args[0]
    assert model.prepare_itlist(content) == [
        {'type': 'text', 'text': 'Question'},
        {'type': 'input_audio', 'input_audio': {'data': 'YXVkaW8=', 'format': 'wav'}},
    ]
    normalize.assert_called_once()
    assert message == original
    assert model.generate([{'type': 'text', 'value': 'Next question'}]) == 'ok'
    assert model.generate_inner.call_args.kwargs['system_prompt'] == 'Configured prompt.'
    assert model.system_prompt == 'Configured prompt.'


def test_gemini_audio_uses_request_system_prompt_and_preserves_media_order(monkeypatch):
    model = GeminiWrapper.__new__(GeminiWrapper)
    model.backend = 'genai'
    model.model = 'test-model'
    model.system_prompt = 'Configured prompt.'
    model.temperature = 0
    model.max_tokens = 32
    model.media_resolution = None
    model.thinking_budget = None
    model.audio_max_file_size = 1024
    model.audio_inline_max_file_size = 1024
    model.audio_cache_dir = None
    audio_part = object()
    model.types = SimpleNamespace(
        Part=SimpleNamespace(from_bytes=Mock(return_value=audio_part)),
        GenerateContentConfig=lambda **kwargs: kwargs,
    )
    generate_content = Mock(return_value=SimpleNamespace(text='ok'))
    model.client = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    monkeypatch.setattr('vlmeval.api.gemini.resolve_media_source', lambda value, **kwargs: value)
    monkeypatch.setattr(
        'vlmeval.api.gemini.normalize_audio',
        Mock(return_value=SimpleNamespace(data=b'audio', mime_type='audio/wav')),
    )
    message = [
        {'type': 'text', 'value': 'Before audio'},
        {'type': 'audio', 'value': 'sample.wav'},
        {'type': 'text', 'value': 'After audio'},
    ]

    assert model.generate_inner(message, system_prompt='Request prompt.') == (0, 'ok', 'Succeeded! ')
    assert generate_content.call_args.kwargs['contents'] == [
        'Request prompt.', 'Before audio', audio_part, 'After audio',
    ]
    assert model.generate_inner(message) == (0, 'ok', 'Succeeded! ')
    assert generate_content.call_args.kwargs['contents'] == ['Before audio', audio_part, 'After audio']
    assert model.system_prompt == 'Configured prompt.'


def test_lmdeploy_uses_message_system_prompt_without_mutating_model(monkeypatch):
    model = LMDeployWrapper.__new__(LMDeployWrapper)
    BaseAPI.__init__(
        model,
        retry=1,
        wait=0,
        system_prompt='Configured prompt.',
        verbose=False,
    )
    model.adapter = None
    model.api_base = 'http://localhost/v1/chat/completions'
    model.key = 'test-key'
    model.model = 'test-model'
    model.response_charset = None
    model.stream = False
    model.timeout = 1
    model.local_media = False
    model.img_size = -1
    model.total_img_size = -1
    model.max_file_size = 1e9

    response = Mock()
    response.status_code = 200
    response.text = json.dumps({
        'choices': [{'message': {'content': 'ok'}}],
    })
    post = Mock(return_value=response)
    monkeypatch.setattr('vlmeval.api.openai_sdk.requests.post', post)
    monkeypatch.setattr('vlmeval.api.base.time.sleep', lambda _: None)

    message = [
        {'role': 'system', 'type': 'text', 'value': 'Request prompt.'},
        {'role': 'user', 'type': 'text', 'value': 'Question'},
    ]
    original_message = [item.copy() for item in message]

    assert model.generate(message) == 'ok'

    payload = json.loads(post.call_args.kwargs['data'])
    assert payload['messages'][0] == {
        'role': 'system',
        'content': 'Configured prompt.\nRequest prompt.',
    }
    assert payload['messages'][1]['role'] == 'user'
    assert model.system_prompt == 'Configured prompt.'
    assert message == original_message


@pytest.mark.parametrize(
    ('wrapper_cls', 'module_name'),
    [
        (RBdashMMChat3_78B_Wrapper, 'vlmeval.api.rbdashmm_chat3_5_api'),
        (RBdashMMChat3_5_38B_Wrapper, 'vlmeval.api.rbdashmm_chat3_5_api'),
        (RBdashMMChat3Wrapper, 'vlmeval.api.rbdashmm_chat3_api'),
        (RBdashMMChat3_5_Wrapper, 'vlmeval.api.rbdashmm_chat3_api'),
        (TeleMM2_Wrapper, 'vlmeval.api.telemm'),
        (TeleMM2Thinking_Wrapper, 'vlmeval.api.telemm_thinking'),
    ],
)
def test_dataset_scoped_models_ignore_request_system_prompt_for_other_datasets(
        monkeypatch, wrapper_cls, module_name):
    model = wrapper_cls.__new__(wrapper_cls)
    model.max_tokens = 32
    model.temperature = 0.0
    model.key = 'test-key'
    model.authorization = 'test-key'
    model.app_id = 'test-app'
    model.model = 'test-model'
    model.api_base = 'http://localhost/v1/chat/completions'
    model.timeout = 1
    model.fail_msg = 'failed'
    model.prepare_inputs = Mock(return_value=[])

    response = Mock()
    response.status_code = 200
    response.text = json.dumps({
        'choices': [{'message': {'content': 'ok'}}],
    })
    monkeypatch.setattr(f'{module_name}.requests.post', Mock(return_value=response))

    model.generate_inner([], system_prompt='Request prompt.', dataset='OtherDataset')

    assert model.prepare_inputs.call_args.args[-1] is None


def test_taichu_vl_keeps_fixed_system_prompt_marker():
    model = TaichuVLWrapper(
        key='test-key',
        url='http://localhost/v1/chat/completions',
        system_prompt='Configured prompt.',
        verbose=False,
    )
    fixed_prompt = '[sys]You are a helpful assistant.[/sys]'

    messages, _ = model.prompt_to_request_messages([
        {'type': 'text', 'value': fixed_prompt},
        {'type': 'text', 'value': 'Question'},
    ])

    assert model.system_prompt == fixed_prompt
    assert messages[0] == {
        'role': 'system',
        'content': [{'type': 'text', 'text': fixed_prompt}],
    }


@pytest.mark.parametrize(
    ('wrapper_cls', 'kwargs'),
    [
        (TaichuVLWrapper, {}),
        (TaichuVLRWrapper, {'use_reasoning_prompt': False, 'post_process': False}),
    ],
)
def test_taichu_wrappers_ignore_request_scoped_system_prompt(monkeypatch, wrapper_cls, kwargs):
    model = wrapper_cls(
        key='test-key',
        url='http://localhost/v1/chat/completions',
        system_prompt='Configured prompt.',
        verbose=False,
        **kwargs,
    )
    response = Mock()
    response.content = json.dumps({
        'choices': [{'message': {'content': 'ok'}}],
    }).encode()
    post = Mock(return_value=response)
    monkeypatch.setattr('vlmeval.api.taichu.requests.post', post)

    model.generate_inner(
        [{'type': 'text', 'value': 'Question'}],
        system_prompt='Request prompt.',
    )

    payload = post.call_args.kwargs['json']
    assert [message['role'] for message in payload['messages']] == ['user']
