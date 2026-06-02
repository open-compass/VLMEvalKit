from __future__ import annotations

from .base import RBLNVLMBase


class RBLNPaliGemma2(RBLNVLMBase):
    """optimum-rbln backend for PaliGemma2 models (google/paligemma2-*).

    Reuses optimum-rbln's ``RBLNPaliGemmaForConditionalGeneration`` (no
    PaliGemma2-specific class is exposed). Kept as a dedicated wrapper
    so the registry stays explicit and so future PaliGemma2-only prompt
    tweaks have a clear seam.
    """

    INTERLEAVE = False

    def __init__(
        self,
        model_path: str,
        # Mirrors upstream vlm.PaliGemma.generate_inner (gemma.py:44).
        # PaliGemma2 has no dedicated upstream wrapper, so its parity
        # baseline is vlm.PaliGemma's hardcoded 512-token budget.
        max_new_tokens: int = 512,
        do_sample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )

    def _load_rbln_model_and_processor(self):
        from optimum.rbln import RBLNPaliGemmaForConditionalGeneration
        from transformers import AutoProcessor

        model = self._from_pretrained(RBLNPaliGemmaForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor
