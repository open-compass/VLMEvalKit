import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from vlmeval.api.base import BaseAPI
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
