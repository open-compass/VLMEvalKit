from __future__ import annotations

from .base import RBLNChatVLMBase


class RBLNGemma4(RBLNChatVLMBase):
    """optimum-rbln backend for Gemma 4 (``Gemma4ForConditionalGeneration``).

    Mirrors upstream ``vlm.Gemma4`` (vlmeval/vlm/gemma.py): no system
    prompt, greedy decoding with a 4096-token budget, ``enable_thinking``
    off by default, and the same answer extraction
    (``processor.parse_response`` when present, then
    ``Gemma4.extract_response_for_eval``, which drops thought channels and
    ``</think>`` blocks). Decoding keeps special tokens so the channel
    markers are still there to strip, as upstream does.
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
        from optimum.rbln import RBLNGemma4ForConditionalGeneration
        return RBLNGemma4ForConditionalGeneration

    def _load_rbln_model_and_processor(self):
        from transformers import AutoProcessor

        model = self._from_pretrained(self._resolve_rbln_class())
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    def _finalize_response(self, inputs, generated_ids) -> str:
        from ..gemma import Gemma4

        decoded = self._decode_generated(inputs, generated_ids, skip_special_tokens=False)
        if hasattr(self.processor, 'parse_response'):
            parsed = self.processor.parse_response(decoded)
            if isinstance(parsed, dict):
                # transformers 5 returns a chat message: {'role': ..., 'content': ...}
                decoded = next(
                    (parsed[k] for k in ('answer', 'response', 'content') if isinstance(parsed.get(k), str)),
                    str(parsed),
                )
            elif isinstance(parsed, tuple):
                decoded = parsed[-1]
            elif isinstance(parsed, str):
                decoded = parsed
        response = Gemma4.extract_response_for_eval(decoded)
        # Special tokens were kept for the channel parsing above; drop the
        # end-of-turn / padding markers that remain at the tail.
        for tok in ('<turn|>', '<eos>', '<pad>', '<end_of_turn>'):
            response = response.replace(tok, '')
        return response.strip()
