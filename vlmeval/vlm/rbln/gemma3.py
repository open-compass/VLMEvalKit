from __future__ import annotations

from .base import RBLNChatVLMBase

# Matches upstream vlm.Gemma3.__init__ default (vlmeval/vlm/gemma.py:108).
# Trailing space is preserved verbatim to keep tokenization identical to
# upstream — Gemma3 chat template emits this string as the system role's
# text content, and the trailing space lands inside the role's payload.
GEMMA3_DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant. '


class RBLNGemma3(RBLNChatVLMBase):
    """optimum-rbln backend for Gemma3 (4b / 12b / 27b) image-text-to-text.

    Gemma3 is the only family in the rbln-model-zoo that uses
    ``RBLNAutoModelForImageTextToText`` instead of
    ``RBLNAutoModelForVision2Seq``; the override is contained in
    ``_resolve_rbln_class``.

    Prompt parity with upstream ``vlm.Gemma3``: same default
    ``system_prompt='You are a helpful assistant. '`` so the chat template
    emits an identical system role across both backends.
    """

    def __init__(
        self,
        model_path: str,
        system_prompt: str | None = GEMMA3_DEFAULT_SYSTEM_PROMPT,
        # Matches upstream vlm.Gemma3 defaults (gemma.py:109-111): greedy
        # decoding with a 4096-token budget. Both transformers and vLLM
        # paths in upstream share these values.
        max_new_tokens: int = 4096,
        do_sample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

    def _resolve_rbln_class(self):
        from optimum.rbln import RBLNAutoModelForImageTextToText
        return RBLNAutoModelForImageTextToText

    def _load_rbln_model_and_processor(self):
        from transformers import AutoProcessor

        model = self._from_pretrained(self._resolve_rbln_class())
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor
