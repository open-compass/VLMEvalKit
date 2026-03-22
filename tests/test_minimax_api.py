"""Unit tests for MiniMax API provider.

Avoids importing the full vlmeval package (which has heavy deps like decord,
torch, etc.) by loading only the needed modules via importlib.
"""
import importlib.util
import json
import logging
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Minimal stubs so base.py and minimax_api.py can be loaded standalone ──
_vlmeval = types.ModuleType("vlmeval")
_vlmeval.__path__ = [os.path.join(_REPO, "vlmeval")]
sys.modules["vlmeval"] = _vlmeval

_smp = types.ModuleType("vlmeval.smp")
_smp.get_logger = lambda name: logging.getLogger(name)
# base.py also imports from ..smp
sys.modules["vlmeval.smp"] = _smp

_api = types.ModuleType("vlmeval.api")
_api.__path__ = [os.path.join(_REPO, "vlmeval", "api")]
sys.modules["vlmeval.api"] = _api
_vlmeval.api = _api  # link so patch("vlmeval.api.minimax_api...") works
_vlmeval.smp = _smp


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load base.py (needs parse_file, concat_images_vlmeval, etc. from smp – stub them)
_smp.parse_file = lambda s: (None, s)
_smp.concat_images_vlmeval = None
_smp.LMUDataRoot = lambda: "/tmp"
_smp.md5 = lambda x: "stub"
_smp.decode_base64_to_image_file = lambda *a, **kw: None

_base = _load_module("vlmeval.api.base", os.path.join(_REPO, "vlmeval", "api", "base.py"))
BaseAPI = _base.BaseAPI

# Load minimax_api.py
_mm = _load_module("vlmeval.api.minimax_api", os.path.join(_REPO, "vlmeval", "api", "minimax_api.py"))
MiniMaxAPI = _mm.MiniMaxAPI
MINIMAX_API_BASE = _mm.MINIMAX_API_BASE
_api.minimax_api = _mm  # link so patch("vlmeval.api.minimax_api.requests.post") works


# ═══════════════════════ Tests ═══════════════════════


class TestMiniMaxAPIInit(unittest.TestCase):
    """Test MiniMaxAPI initialization."""

    def test_default_init(self):
        api = MiniMaxAPI(key="test-key")
        assert api.model == "MiniMax-M2.7"
        assert api.api_base == MINIMAX_API_BASE
        assert api.key == "test-key"
        assert api.temperature == 0
        assert api.max_tokens == 2048
        assert api.timeout == 300

    def test_custom_model(self):
        api = MiniMaxAPI(model="MiniMax-M2.5", key="test-key")
        assert api.model == "MiniMax-M2.5"

    def test_custom_api_base(self):
        api = MiniMaxAPI(key="test-key", api_base="https://custom.api.com/v1/chat/completions")
        assert api.api_base == "https://custom.api.com/v1/chat/completions"

    def test_env_api_key(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False):
            api = MiniMaxAPI()
            assert api.key == "env-key"

    def test_env_api_base(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "key", "MINIMAX_API_BASE": "https://custom.base/v1"}, clear=False):
            api = MiniMaxAPI()
            assert api.api_base == "https://custom.base/v1"

    def test_missing_key_raises(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                MiniMaxAPI()

    def test_is_api_flag(self):
        api = MiniMaxAPI(key="test-key")
        assert api.is_api is True

    def test_custom_temperature(self):
        api = MiniMaxAPI(key="test-key", temperature=0.5)
        assert api.temperature == 0.5

    def test_custom_max_tokens(self):
        api = MiniMaxAPI(key="test-key", max_tokens=4096)
        assert api.max_tokens == 4096

    def test_custom_timeout(self):
        api = MiniMaxAPI(key="test-key", timeout=600)
        assert api.timeout == 600

    def test_inherits_base_api(self):
        api = MiniMaxAPI(key="test-key")
        assert isinstance(api, BaseAPI)

    def test_key_precedence_over_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "env-key"}, clear=False):
            api = MiniMaxAPI(key="explicit-key")
            assert api.key == "explicit-key"


class TestMiniMaxAPIPrepareMessages(unittest.TestCase):
    """Test message preparation logic."""

    def setUp(self):
        self.api = MiniMaxAPI(key="test-key")

    def test_single_turn_text(self):
        inputs = [{"type": "text", "value": "Hello, world!"}]
        messages = self.api._prepare_messages(inputs)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, world!"

    def test_single_turn_multi_text(self):
        inputs = [
            {"type": "text", "value": "Line 1"},
            {"type": "text", "value": "Line 2"},
        ]
        messages = self.api._prepare_messages(inputs)
        assert len(messages) == 1
        assert messages[0]["content"] == "Line 1\nLine 2"

    def test_with_system_prompt(self):
        api = MiniMaxAPI(key="test-key", system_prompt="You are helpful.")
        inputs = [{"type": "text", "value": "Hi"}]
        messages = api._prepare_messages(inputs)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"

    def test_multi_turn_chat(self):
        inputs = [
            {"role": "user", "content": [{"type": "text", "value": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "value": "Hello!"}]},
            {"role": "user", "content": [{"type": "text", "value": "How are you?"}]},
        ]
        messages = self.api._prepare_messages(inputs)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_image_inputs_ignored(self):
        """MiniMax is text-only, image inputs should be skipped."""
        inputs = [
            {"type": "image", "value": "/path/to/image.jpg"},
            {"type": "text", "value": "Describe this."},
        ]
        messages = self.api._prepare_messages(inputs)
        assert len(messages) == 1
        assert messages[0]["content"] == "Describe this."

    def test_empty_text(self):
        inputs = [{"type": "text", "value": ""}]
        messages = self.api._prepare_messages(inputs)
        assert messages[-1]["content"] == ""

    def test_multi_turn_with_system_prompt(self):
        api = MiniMaxAPI(key="test-key", system_prompt="Be concise.")
        inputs = [
            {"role": "user", "content": [{"type": "text", "value": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "value": "Hello!"}]},
        ]
        messages = api._prepare_messages(inputs)
        assert len(messages) == 3
        assert messages[0]["role"] == "system"


class TestMiniMaxAPIGenerate(unittest.TestCase):
    """Test generate_inner with mocked HTTP responses."""

    def setUp(self):
        self.api = MiniMaxAPI(key="test-key")

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_successful_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "The answer is 42."}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "What is the answer?"}]
        ret_code, answer, response = self.api.generate_inner(inputs)

        assert ret_code == 0
        assert answer == "The answer is 42."
        mock_post.assert_called_once()

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_api_error_status(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.json.return_value = {"error": {"message": "Rate limit exceeded"}}
        mock_resp.text = '{"error": {"message": "Rate limit exceeded"}}'
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        ret_code, answer, response = self.api.generate_inner(inputs)

        assert ret_code == 429

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_network_error(self, mock_post):
        mock_post.side_effect = ConnectionError("Network unreachable")

        inputs = [{"type": "text", "value": "test"}]
        ret_code, answer, response = self.api.generate_inner(inputs)

        assert ret_code == -1
        assert "Failed" in answer

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_malformed_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = json.JSONDecodeError("err", "", 0)
        mock_resp.text = "not json"
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        ret_code, answer, response = self.api.generate_inner(inputs)

        assert ret_code == 0
        assert "Failed" in answer

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_request_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "Hello"}]
        self.api.generate_inner(inputs)

        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])
        assert payload["model"] == "MiniMax-M2.7"
        assert payload["temperature"] == 0
        assert payload["max_tokens"] == 2048
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello"

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_custom_kwargs_override(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        self.api.generate_inner(inputs, temperature=0.7, max_tokens=512)

        call_args = mock_post.call_args
        payload = json.loads(call_args[1]["data"])
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 512

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_headers(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        self.api.generate_inner(inputs)

        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_timeout_value(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        self.api.generate_inner(inputs)

        call_args = mock_post.call_args
        assert call_args[1]["timeout"] == 300 * 1.1

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_whitespace_stripping(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "  answer with spaces  "}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        ret_code, answer, _ = self.api.generate_inner(inputs)

        assert answer == "answer with spaces"

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_api_base_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        self.api.generate_inner(inputs)

        call_args = mock_post.call_args
        assert call_args[0][0] == MINIMAX_API_BASE

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_m25_model(self, mock_post):
        api = MiniMaxAPI(model="MiniMax-M2.5", key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        api.generate_inner(inputs)

        payload = json.loads(mock_post.call_args[1]["data"])
        assert payload["model"] == "MiniMax-M2.5"

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_m25_highspeed_model(self, mock_post):
        api = MiniMaxAPI(model="MiniMax-M2.5-highspeed", key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        api.generate_inner(inputs)

        payload = json.loads(mock_post.call_args[1]["data"])
        assert payload["model"] == "MiniMax-M2.5-highspeed"

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_500_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.json.return_value = {"error": "internal server error"}
        mock_resp.text = '{"error": "internal server error"}'
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        ret_code, answer, _ = self.api.generate_inner(inputs)
        assert ret_code == 500

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_timeout_exception(self, mock_post):
        import requests
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        inputs = [{"type": "text", "value": "test"}]
        ret_code, answer, _ = self.api.generate_inner(inputs)
        assert ret_code == -1
        assert "Failed" in answer

    @patch("vlmeval.api.minimax_api.requests.post")
    def test_m27_model(self, mock_post):
        api = MiniMaxAPI(model="MiniMax-M2.7", key="test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_post.return_value = mock_resp

        inputs = [{"type": "text", "value": "test"}]
        api.generate_inner(inputs)

        payload = json.loads(mock_post.call_args[1]["data"])
        assert payload["model"] == "MiniMax-M2.7"


class TestMiniMaxAPIIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API (skipped without API key)."""

    def setUp(self):
        self.api_key = os.environ.get("MINIMAX_API_KEY")
        if not self.api_key or self.api_key == "test-key-for-unit-tests":
            self.skipTest("MINIMAX_API_KEY not set for integration tests")

    def test_live_generate(self):
        api = MiniMaxAPI(model="MiniMax-M2.5-highspeed", retry=2, verbose=False)
        result = api.generate("What is 2 + 3? Answer with just the number.")
        assert result is not None
        assert "5" in result

    def test_live_m27(self):
        api = MiniMaxAPI(model="MiniMax-M2.7", retry=2, verbose=False)
        result = api.generate("Reply with the word 'hello' only.")
        assert result is not None
        assert len(result) > 0

    def test_live_with_system_prompt(self):
        api = MiniMaxAPI(
            model="MiniMax-M2.5-highspeed",
            system_prompt="You always respond in exactly one word.",
            retry=2,
            verbose=False,
        )
        result = api.generate("What color is the sky?")
        assert result is not None
        assert len(result) > 0


if __name__ == "__main__":
    unittest.main()
