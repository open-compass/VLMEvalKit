from __future__ import annotations

from .base import RBLNChatVLMBase
from .qwen3_5 import strip_think


class RBLNExaone4_5(RBLNChatVLMBase):
    """optimum-rbln backend for EXAONE 4.5 (``Exaone4_5_ForConditionalGeneration``).

    Upstream VLMEvalKit has no EXAONE 4.5 wrapper, so this follows the
    model's own chat template: images as ``{"type": "image"}`` entries,
    no system prompt, greedy decoding. The template turns reasoning on
    unless ``enable_thinking`` is false, so it is off by default here and
    any ``</think>`` block is dropped from the response.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        enable_thinking: bool = False,
        chat_template_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        chat_template_kwargs = {'enable_thinking': enable_thinking, **(chat_template_kwargs or {})}
        super().__init__(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            chat_template_kwargs=chat_template_kwargs,
            **kwargs,
        )

    def _pick_rbln_class(self):
        from optimum.rbln import RBLNExaone4_5_ForConditionalGeneration
        return RBLNExaone4_5_ForConditionalGeneration

    def _load_rbln_model_and_processor(self):
        from transformers import AutoProcessor

        model = self._from_pretrained(self._resolve_rbln_class())
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    def _postprocess_response(self, response: str) -> str:
        return strip_think(response)
