"""Tests for the OrcaRouter API provider (vlmeval/api/orcarouter.py).

Runs without heavy VLMEvalKit dependencies — stubs out vlmeval.smp
and vlmeval.api.base so only orcarouter.py is exercised.
"""
import importlib.util
import logging
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Stub the vlmeval package so we can import orcarouter in isolation
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _stub_vlmeval(monkeypatch):
    """Install minimal stubs for vlmeval.smp and vlmeval.api.base."""
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = []
    monkeypatch.setitem(sys.modules, 'vlmeval', vlmeval)

    smp = types.ModuleType('vlmeval.smp')
    smp.concat_images_vlmeval = lambda *a, **k: None
    smp.get_logger = lambda name: logging.getLogger(name)
    smp.parse_file = lambda x: (None, x)
    smp.encode_image_to_base64 = mock.MagicMock(return_value='c3R1Yg==')
    monkeypatch.setitem(sys.modules, 'vlmeval.smp', smp)
    vlmeval.smp = smp

    base_spec = importlib.util.spec_from_file_location(
        'vlmeval.api.base',
        'vlmeval/api/base.py',
    )
    base_mod = importlib.util.module_from_spec(base_spec)
    monkeypatch.setitem(sys.modules, 'vlmeval.api.base', base_mod)
    base_spec.loader.exec_module(base_mod)

    yield


def _load_module():
    """Load the orcarouter module and return it."""
    spec = importlib.util.spec_from_file_location(
        'vlmeval.api.orcarouter',
        'vlmeval/api/orcarouter.py',
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['vlmeval.api.orcarouter'] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_orcarouter_api():
    return _load_module().OrcaRouterAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_response(content='4'):
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        'choices': [{'message': {'content': content}}],
    }
    return resp


@pytest.fixture
def provider_and_mod(monkeypatch):
    """Return (OrcaRouterAPI instance, module) with requests mocked."""
    mod = _load_module()
    fake_post = mock.MagicMock(return_value=_make_response())
    monkeypatch.setattr(mod.requests, 'post', fake_post)
    p = mod.OrcaRouterAPI(model='orcarouter/auto', key='sk-test', retry=1)
    return p, mod, fake_post


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestOrcaRouterAPIInit:

    def test_default_params(self):
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='sk-test', retry=1)
        assert p.model == 'orcarouter/auto'
        assert p.temperature == 0
        assert p.max_tokens == 2048
        assert p.timeout == 300
        assert p.key == 'sk-test'
        assert p.api_base == 'https://api.orcarouter.ai/v1/chat/completions'

    def test_custom_params(self):
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(
            model='orcarouter/deepseek/deepseek-v4-pro',
            key='sk-test-key',
            api_base='https://api.orcarouter.ai/v1/chat/completions',
            temperature=0.7,
            max_tokens=100,
            timeout=60,
            retry=1,
        )
        assert p.model == 'orcarouter/deepseek/deepseek-v4-pro'
        assert p.api_base == 'https://api.orcarouter.ai/v1/chat/completions'
        assert p.temperature == 0.7
        assert p.max_tokens == 100

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv('ORCAROUTER_API_KEY', 'env-key-123')
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', retry=1)
        assert p.key == 'env-key-123'

    def test_key_param_overrides_env(self, monkeypatch):
        monkeypatch.setenv('ORCAROUTER_API_KEY', 'env-key')
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='param-key', retry=1)
        assert p.key == 'param-key'

    def test_api_base_from_env(self, monkeypatch):
        monkeypatch.setenv('ORCAROUTER_API_BASE', 'https://custom.example/v1/chat/completions')
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='sk', retry=1)
        assert p.api_base == 'https://custom.example/v1/chat/completions'

    def test_missing_key_raises(self):
        OrcaRouterAPI = _load_orcarouter_api()
        with pytest.raises(ValueError, match='OrcaRouter API key is required'):
            OrcaRouterAPI(model='orcarouter/auto', key=None, retry=1)


class TestPrepareContent:

    def test_text_only(self):
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='sk', retry=1)
        inputs = [
            {'type': 'text', 'value': 'Hello'},
            {'type': 'text', 'value': 'World'},
        ]
        result = p._prepare_content(inputs)
        assert len(result) == 1
        assert result[0]['type'] == 'text'
        assert result[0]['text'] == 'Hello\nWorld'

    def test_image_and_text(self, tmp_path):
        from PIL import Image
        img_path = str(tmp_path / 'test.jpg')
        Image.new('RGB', (10, 10), color='red').save(img_path)

        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='sk', retry=1)
        inputs = [
            {'type': 'text', 'value': 'Describe this image'},
            {'type': 'image', 'value': img_path},
        ]
        result = p._prepare_content(inputs)
        assert len(result) == 2
        assert result[0] == {'type': 'text', 'text': 'Describe this image'}
        assert result[1]['type'] == 'image_url'
        assert 'data:image/jpeg;base64,' in result[1]['image_url']['url']


class TestPrepareMessages:

    def test_flat_inputs(self):
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='sk', retry=1)
        inputs = [{'type': 'text', 'value': 'Hello'}]
        msgs = p._prepare_messages(inputs)
        assert len(msgs) == 1
        assert msgs[0]['role'] == 'user'

    def test_system_prompt(self):
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', system_prompt='Be concise.', key='sk', retry=1)
        inputs = [{'type': 'text', 'value': 'Hello'}]
        msgs = p._prepare_messages(inputs)
        assert len(msgs) == 2
        assert msgs[0]['role'] == 'system'
        assert msgs[0]['content'] == 'Be concise.'
        assert msgs[1]['role'] == 'user'

    def test_role_based_inputs(self):
        OrcaRouterAPI = _load_orcarouter_api()
        p = OrcaRouterAPI(model='orcarouter/auto', key='sk', retry=1)
        inputs = [
            {'role': 'user', 'content': [{'type': 'text', 'value': 'Hi'}]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': 'Hello!'}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': 'Bye'}]},
        ]
        msgs = p._prepare_messages(inputs)
        assert len(msgs) == 3
        assert msgs[0]['role'] == 'user'
        assert msgs[1]['role'] == 'assistant'
        assert msgs[2]['role'] == 'user'


class TestGenerateInner:

    def test_success(self, provider_and_mod):
        p, mod, fake_post = provider_and_mod

        inputs = [{'type': 'text', 'value': 'What is 6*7?'}]
        ret_code, answer, log = p.generate_inner(inputs)

        assert ret_code == 0
        assert answer == '4'
        call_args = fake_post.call_args
        assert call_args.args[0] == 'https://api.orcarouter.ai/v1/chat/completions'
        assert call_args.kwargs['headers']['Authorization'] == 'Bearer sk-test'
        import json as _json
        payload = _json.loads(call_args.kwargs['data'])
        assert payload['model'] == 'orcarouter/auto'
        assert payload['messages'][0]['role'] == 'user'
        assert payload['temperature'] == 0
        assert payload['max_tokens'] == 2048

    def test_error_returns_negative_one(self, provider_and_mod, monkeypatch):
        p, mod, fake_post = provider_and_mod
        p.verbose = False
        fake_post.side_effect = Exception('API down')

        inputs = [{'type': 'text', 'value': 'test'}]
        ret_code, answer, log = p.generate_inner(inputs)

        assert ret_code == -1
        assert 'Failed' in answer
        assert 'API down' in log

    def test_http_error_returns_status(self, provider_and_mod):
        p, mod, fake_post = provider_and_mod
        resp = mock.MagicMock(status_code=401, text='Unauthorized')
        resp.json.side_effect = ValueError('no json')
        fake_post.return_value = resp

        inputs = [{'type': 'text', 'value': 'test'}]
        ret_code, answer, log = p.generate_inner(inputs)

        assert ret_code == 401
        assert 'Failed' in answer

    def test_temperature_override(self, provider_and_mod):
        p, mod, fake_post = provider_and_mod

        p.generate_inner(
            [{'type': 'text', 'value': 'test'}],
            temperature=0.9,
        )
        import json as _json
        payload = _json.loads(fake_post.call_args.kwargs['data'])
        assert payload['temperature'] == 0.9

    def test_max_tokens_override(self, provider_and_mod):
        p, mod, fake_post = provider_and_mod

        p.generate_inner(
            [{'type': 'text', 'value': 'test'}],
            max_tokens=500,
        )
        import json as _json
        payload = _json.loads(fake_post.call_args.kwargs['data'])
        assert payload['max_tokens'] == 500


class TestConfigRegistration:

    def test_orcarouter_entries_in_config(self):
        with open('vlmeval/config.py') as f:
            content = f.read()
        assert 'OrcaRouter_Auto' in content
        assert 'OrcaRouter_DeepSeek_V4_Pro' in content
        assert 'OrcaRouter_Qwen3_5_Flash' in content
        assert 'api.OrcaRouterAPI' in content

    def test_orcarouter_in_init_all(self):
        with open('vlmeval/api/__init__.py') as f:
            content = f.read()
        assert 'from .orcarouter import OrcaRouterAPI' in content
        assert "'OrcaRouterAPI'" in content

    def test_api_base_constant(self):
        with open('vlmeval/api/orcarouter.py') as f:
            content = f.read()
        assert 'https://api.orcarouter.ai/v1/chat/completions' in content
