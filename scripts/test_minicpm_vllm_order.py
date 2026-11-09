"""MiniCPM vLLM chat path must keep multimodal content order (SLP != VVTT).

MiniCPM_V_4_5 feeds vLLM through ``llm.chat(...,
chat_template_content_format="string")``. With vLLM's default
``interleave_mm_strings=False`` the chat parser moves every image placeholder
to the front of the joined text, so <V,T,V,T> reaches the engine as <V,V,T,T>.
These tests run vLLM's own parser (``parse_chat_messages``) against the
checkpoint's ModelConfig, i.e. the code path ``llm.chat`` takes, and check
that the wrapper enables interleaving.

CPU only. Needs vllm plus the checkpoint (``MODEL_PATH_MINICPMO45`` or
``MODEL_PATH``); those tests are skipped otherwise. Run e.g.:

    MODEL_PATH_MINICPMO45=/path/MiniCPM-o-4_5 VLMEVAL_VLM_MINIMAL_IMPORT=1 \
        python scripts/test_minicpm_vllm_order.py
"""
import ast
import os
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WRAPPER = _REPO / "vlmeval/vlm/minicpm_v_4_5_replay.py"
_MODEL_PATH = os.environ.get("MODEL_PATH_MINICPMO45") or os.environ.get("MODEL_PATH", "")
_PATTERN = "(<image>./</image>)"
# V = image, T = text; the replay orders the wrapper can emit, plus image-first ones.
_SHAPES = ["VT", "VTT", "VVTT", "VTTT", "VTV", "VTVT", "VTVVT", "VTVVVT", "TV"]
_IMAGE_FIRST = [s for s in _SHAPES if "TV" not in s]


def _vllm_available():
    try:
        import vllm  # noqa: F401
    except Exception:
        return False
    return True


def _messages(shape, image):
    """(vLLM content parts, VLMEval message items) for one shape."""
    parts, items, n = [], [], 0
    for c in shape:
        if c == "V":
            parts.append({"type": "image_pil", "image_pil": image})
            items.append({"type": "image", "value": "unused.png"})
        else:
            n += 1
            parts.append({"type": "text", "text": f"T{n}"})
            items.append({"type": "text", "value": f"T{n}"})
    return parts, items


def _expected(shape):
    out, n = [], 0
    for c in shape:
        if c == "V":
            out.append(_PATTERN)
        else:
            n += 1
            out.append(f"T{n}")
    return "\n".join(out)


class WrapperSourceTests(unittest.TestCase):
    def test_llm_constructor_enables_interleave(self):
        tree = ast.parse(_WRAPPER.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "LLM"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one vllm.LLM(...) construction")
        flags = [kw.value for kw in calls[0].keywords if kw.arg == "interleave_mm_strings"]
        self.assertEqual(len(flags), 1, "LLM(...) must pass interleave_mm_strings")
        self.assertIs(getattr(flags[0], "value", None), True)


@unittest.skipUnless(_vllm_available() and _MODEL_PATH and Path(_MODEL_PATH).is_dir(),
                     "needs vllm and a MiniCPM checkpoint (MODEL_PATH_MINICPMO45)")
class EngineParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        from vllm.config import ModelConfig
        from vllm.entrypoints.chat_utils import parse_chat_messages

        cls.image = Image.new("RGB", (64, 64))
        cls.parse = staticmethod(parse_chat_messages)
        cls.configs = {
            flag: ModelConfig(
                model=_MODEL_PATH,
                trust_remote_code=True,
                interleave_mm_strings=flag,
                limit_mm_per_prompt={"image": 8},
            )
            for flag in (False, True)
        }

    def render(self, shape, interleave):
        parts, _ = _messages(shape, self.image)
        conversation, _, _ = self.parse(
            [{"role": "user", "content": parts}], self.configs[interleave], "string"
        )
        return conversation[0]["content"]

    def test_interleaved_parser_keeps_content_order(self):
        for shape in _SHAPES:
            with self.subTest(shape=shape):
                self.assertEqual(self.render(shape, True), _expected(shape))

    def test_slp_and_vvtt_differ_only_when_interleaved(self):
        self.assertNotEqual(self.render("VTVT", True), self.render("VVTT", True))

    def test_image_first_content_is_byte_identical(self):
        for shape in _IMAGE_FIRST:
            with self.subTest(shape=shape):
                self.assertEqual(self.render(shape, True), self.render(shape, False))


if __name__ == "__main__":
    unittest.main()
