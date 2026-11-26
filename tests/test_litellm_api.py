"""Tests for the LiteLLM API provider (vlmeval/api/litellm_api.py).

Runs without heavy VLMEvalKit dependencies — stubs out vlmeval.smp
and vlmeval.api.base so only litellm_api.py is exercised.
"""
import importlib.util
import logging
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Stub the vlmeval package so we can import litellm_api in isolation
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
    """Load the litellm_api module and return it."""
    spec = importlib.util.spec_from_file_location(
        'vlmeval.api.litellm_api',
        'vlmeval/api/litellm_api.py',
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules['vlmeval.api.litellm_api'] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_litellm_api():
    return _load_module().LiteLLMAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_completion_response(content='4'):
    """Return a mock object shaped like litellm.ModelResponse."""
    choice = mock.MagicMock()
    choice.message.content = content
    resp = mock.MagicMock()
    resp.choices = [choice]
    resp.model = 'gpt-4o-mini'
    resp.usage = mock.MagicMock(
        prompt_tokens=10, completion_tokens=2, total_tokens=12,
    )
    return resp


@pytest.fixture
def provider_and_mod():
    """Return (LiteLLMAPI instance, module) with litellm mocked."""
    mod = _load_module()
    fake_litellm = mock.MagicMock()
    fake_litellm.completion.return_value = _make_completion_response()
    mod.litellm = fake_litellm
    p = mod.LiteLLMAPI(model='gpt-4o', retry=1)
    return p, mod, fake_litellm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLiteLLMAPIInit:

    def test_default_params(self):
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', retry=1)
        assert p.model == 'gpt-4o'
        assert p.temperature == 0
        assert p.max_tokens == 2048
        assert p.timeout == 300
        assert p.api_key is None
        assert p.api_base is None

    def test_custom_params(self):
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(
            model='anthropic/claude-haiku-4-5-20251001',
            key='sk-test-key',
            api_base='https://my-proxy.com',
            temperature=0.7,
            max_tokens=100,
            timeout=60,
            retry=1,
        )
        assert p.model == 'anthropic/claude-haiku-4-5-20251001'
        assert p.api_key == 'sk-test-key'
        assert p.api_base == 'https://my-proxy.com'
        assert p.temperature == 0.7
        assert p.max_tokens == 100

    def test_key_from_env(self, monkeypatch):
        monkeypatch.setenv('LITELLM_API_KEY', 'env-key-123')
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', retry=1)
        assert p.api_key == 'env-key-123'

    def test_key_param_overrides_env(self, monkeypatch):
        monkeypatch.setenv('LITELLM_API_KEY', 'env-key')
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', key='param-key', retry=1)
        assert p.api_key == 'param-key'


class TestPrepareContent:

    def test_text_only(self):
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', retry=1)
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

        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', retry=1)
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
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', retry=1)
        inputs = [{'type': 'text', 'value': 'Hello'}]
        msgs = p._prepare_messages(inputs)
        assert len(msgs) == 1
        assert msgs[0]['role'] == 'user'

    def test_system_prompt(self):
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', system_prompt='Be concise.', retry=1)
        inputs = [{'type': 'text', 'value': 'Hello'}]
        msgs = p._prepare_messages(inputs)
        assert len(msgs) == 2
        assert msgs[0]['role'] == 'system'
        assert msgs[0]['content'] == 'Be concise.'
        assert msgs[1]['role'] == 'user'

    def test_role_based_inputs(self):
        LiteLLMAPI = _load_litellm_api()
        p = LiteLLMAPI(model='gpt-4o', retry=1)
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
        p, mod, fake_litellm = provider_and_mod
        fake_litellm.completion.return_value = _make_completion_response('42')

        inputs = [{'type': 'text', 'value': 'What is 6*7?'}]
        ret_code, answer, log = p.generate_inner(inputs)

        assert ret_code == 0
        assert answer == '42'
        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['model'] == 'gpt-4o'
        assert call_kwargs['drop_params'] is True
        assert call_kwargs['temperature'] == 0
        assert call_kwargs['max_tokens'] == 2048

    def test_drop_params_default_true(self, provider_and_mod):
        p, mod, fake_litellm = provider_and_mod

        p.generate_inner([{'type': 'text', 'value': 'test'}])

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['drop_params'] is True

    def test_api_key_forwarded(self):
        mod = _load_module()
        fake_litellm = mock.MagicMock()
        fake_litellm.completion.return_value = _make_completion_response()
        mod.litellm = fake_litellm

        p = mod.LiteLLMAPI(model='gpt-4o', key='sk-test', retry=1)
        p.generate_inner([{'type': 'text', 'value': 'test'}])

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['api_key'] == 'sk-test'

    def test_api_key_omitted_when_none(self, provider_and_mod):
        p, mod, fake_litellm = provider_and_mod

        p.generate_inner([{'type': 'text', 'value': 'test'}])

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert 'api_key' not in call_kwargs

    def test_api_base_forwarded(self):
        mod = _load_module()
        fake_litellm = mock.MagicMock()
        fake_litellm.completion.return_value = _make_completion_response()
        mod.litellm = fake_litellm

        p = mod.LiteLLMAPI(
            model='gpt-4o', api_base='https://proxy.example.com', retry=1,
        )
        p.generate_inner([{'type': 'text', 'value': 'test'}])

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['api_base'] == 'https://proxy.example.com'

    def test_error_returns_negative_one(self, provider_and_mod):
        p, mod, fake_litellm = provider_and_mod
        p.verbose = False
        fake_litellm.completion.side_effect = Exception('API down')

        inputs = [{'type': 'text', 'value': 'test'}]
        ret_code, answer, log = p.generate_inner(inputs)

        assert ret_code == -1
        assert 'Failed' in answer
        assert 'API down' in log

    def test_litellm_not_installed(self):
        mod = _load_module()
        mod.litellm = None
        p = mod.LiteLLMAPI(model='gpt-4o', retry=1)

        with pytest.raises(ImportError, match='LiteLLM is required'):
            p.generate_inner([{'type': 'text', 'value': 'test'}])

    def test_temperature_override(self, provider_and_mod):
        p, mod, fake_litellm = provider_and_mod
        p.temperature = 0.5

        p.generate_inner(
            [{'type': 'text', 'value': 'test'}],
            temperature=0.9,
        )

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['temperature'] == 0.9

    def test_max_tokens_override(self, provider_and_mod):
        p, mod, fake_litellm = provider_and_mod
        p.max_tokens = 100

        p.generate_inner(
            [{'type': 'text', 'value': 'test'}],
            max_tokens=500,
        )

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['max_tokens'] == 500

    def test_litellm_kwargs_passthrough(self):
        mod = _load_module()
        fake_litellm = mock.MagicMock()
        fake_litellm.completion.return_value = _make_completion_response()
        mod.litellm = fake_litellm

        p = mod.LiteLLMAPI(
            model='gpt-4o', retry=1,
            litellm_kwargs={'seed': 42, 'top_p': 0.9},
        )
        p.generate_inner([{'type': 'text', 'value': 'test'}])

        call_kwargs = fake_litellm.completion.call_args.kwargs
        assert call_kwargs['seed'] == 42
        assert call_kwargs['top_p'] == 0.9


class TestConfigRegistration:

    def test_litellm_entries_in_config(self):
        with open('vlmeval/config.py') as f:
            content = f.read()
        assert 'LiteLLM_GPT4o' in content
        assert 'LiteLLM_GPT4o_Mini' in content
        assert 'LiteLLM_Claude_Sonnet4' in content
        assert 'LiteLLM_Gemini_2.5_Flash' in content
        assert 'LiteLLM_Gemini_2.5_Pro' in content
        assert 'LiteLLM_Bedrock_Claude' in content
        assert 'LiteLLM_Llama_Vision' in content
        assert 'LiteLLM_Groq_Llama4' in content
        assert 'api.LiteLLMAPI' in content

    def test_litellm_in_init_all(self):
        with open('vlmeval/api/__init__.py') as f:
            content = f.read()
        assert 'from .litellm_api import LiteLLMAPI' in content
        assert "'LiteLLMAPI'" in content

    def test_version_pin_in_docstring(self):
        with open('vlmeval/api/litellm_api.py') as f:
            content = f.read()
        assert "litellm>=1.55,<1.85" in content
