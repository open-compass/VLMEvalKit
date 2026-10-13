import ast
import json
from pathlib import Path
from unittest.mock import Mock

from vlmeval.api.base import BaseAPI
from vlmeval.api.lmdeploy import LMDeployWrapper


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
