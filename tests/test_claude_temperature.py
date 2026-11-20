"""Offline regressions for the Claude wrapper's sampling configuration.

Claude_Wrapper.__init__ declares `temperature: float = 0` and stores it, and
every Claude entry in vlmeval/config.py passes `temperature=0`, but
generate_inner never put it in the request body. The Anthropic Messages API
defaults to temperature 1.0, so the configured greedy setting was a no-op.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


class _FakeResponse:
    status_code = 200
    text = json.dumps({'content': [{'text': 'ok'}], 'data': {'content': [{'text': 'ok'}]}})


@pytest.fixture
def claude_module(monkeypatch):
    pkg = sys.modules.setdefault('vlmeval', types.ModuleType('vlmeval'))
    pkg.__path__ = []
    api_pkg = sys.modules.setdefault('vlmeval.api', types.ModuleType('vlmeval.api'))
    api_pkg.__path__ = []

    base = types.ModuleType('vlmeval.api.base')

    class _BaseAPI:
        fail_msg = 'Failed to obtain answer via API.'

        def __init__(self, *args, **kwargs):
            pass

    base.BaseAPI = _BaseAPI
    sys.modules['vlmeval.api.base'] = base

    smp = types.ModuleType('vlmeval.smp')
    smp.encode_image_to_base64 = lambda *a, **k: ''
    smp.get_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None
    )
    sys.modules['vlmeval.smp'] = smp

    spec = importlib.util.spec_from_file_location(
        'claude_under_test', REPO / 'vlmeval/api/claude.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wrapper(module, temperature=0, max_tokens=2048, backend='official'):
    """Build a Claude_Wrapper without touching the network or the env."""
    obj = object.__new__(module.Claude_Wrapper)
    obj.backend = backend
    obj.url = 'https://example.invalid/v1/messages'
    obj.model = 'claude-3-opus-20240229'
    obj.temperature = temperature
    obj.max_tokens = max_tokens
    obj.headers = {}
    obj.timeout = 60
    obj.verbose = False
    obj.fail_msg = 'Failed to obtain answer via API.'
    return obj


def _captured_payload(module, monkeypatch, wrapper, **kwargs):
    sent = {}

    def _fake_request(method, url, headers=None, data=None, timeout=None):
        sent.update(json.loads(data))
        return _FakeResponse()

    monkeypatch.setattr(module.requests, 'request', _fake_request)
    wrapper.generate_inner([{'type': 'text', 'value': 'hello'}], **kwargs)
    return sent


def test_configured_temperature_is_sent(claude_module, monkeypatch):
    payload = _captured_payload(claude_module, monkeypatch, _wrapper(claude_module, temperature=0))
    assert 'temperature' in payload
    assert payload['temperature'] == 0


@pytest.mark.parametrize('temperature', [0, 0.2, 0.7, 1.0])
def test_any_configured_temperature_is_sent(claude_module, monkeypatch, temperature):
    payload = _captured_payload(
        claude_module, monkeypatch, _wrapper(claude_module, temperature=temperature)
    )
    assert payload['temperature'] == temperature


def test_per_call_temperature_overrides_the_configured_one(claude_module, monkeypatch):
    """BaseAPI.generate forwards default_kwargs, so a per-call value must win."""
    payload = _captured_payload(
        claude_module, monkeypatch, _wrapper(claude_module, temperature=0), temperature=0.9
    )
    assert payload['temperature'] == 0.9


def test_max_tokens_and_model_still_sent(claude_module, monkeypatch):
    payload = _captured_payload(
        claude_module, monkeypatch, _wrapper(claude_module, max_tokens=1234)
    )
    assert payload['max_tokens'] == 1234
    assert payload['model'] == 'claude-3-opus-20240229'
    assert payload['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}
    ]


def test_per_call_max_tokens_overrides_the_configured_one(claude_module, monkeypatch):
    payload = _captured_payload(
        claude_module, monkeypatch, _wrapper(claude_module, max_tokens=2048), max_tokens=64
    )
    assert payload['max_tokens'] == 64


def test_system_prompt_and_extra_kwargs_still_forwarded(claude_module, monkeypatch):
    sent = {}

    def _fake_request(method, url, headers=None, data=None, timeout=None):
        sent.update(json.loads(data))
        return _FakeResponse()

    monkeypatch.setattr(claude_module.requests, 'request', _fake_request)
    _wrapper(claude_module).generate_inner(
        [{'type': 'text', 'value': 'hello'}], system_prompt='be terse', top_p=0.5
    )
    assert sent['system'] == 'be terse'
    assert sent['top_p'] == 0.5


def test_answer_is_still_parsed(claude_module, monkeypatch):
    wrapper = _wrapper(claude_module)

    monkeypatch.setattr(
        claude_module.requests, 'request',
        lambda *a, **k: _FakeResponse()
    )
    ret_code, answer, _resp = wrapper.generate_inner([{'type': 'text', 'value': 'hello'}])
    assert ret_code == 0
    assert answer == 'ok'
