from __future__ import annotations

from .base import RBLNVLMBase


class RBLNPaliGemma(RBLNVLMBase):
    """optimum-rbln backend for PaliGemma models.

    PaliGemma uses a raw task-prefix prompt (e.g. ``"describe en"``,
    ``"answer en <question>"``) with a single image — no chat template,
    no Qwen-style MCQ/VQA shaping.
    """

    INTERLEAVE = False

    def __init__(
        self,
        model_path: str,
        # Matches upstream vlm.PaliGemma.generate_inner (gemma.py:44):
        # task-prefix prompts emit short single-line answers, so a tight
        # 512-token budget is sufficient and matches upstream verbatim.
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
