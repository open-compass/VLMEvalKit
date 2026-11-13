from __future__ import annotations

from ..qwen3_vl.prompt import Qwen3VLPromptMixin
from .qwen2_vl import RBLNQwen2VL


class RBLNQwen3VL(Qwen3VLPromptMixin, RBLNQwen2VL):
    """optimum-rbln backend for Qwen3-VL.

    Inherits all Qwen-VL family inference logic (``_prepare_content``,
    ``generate_inner``, etc.) from :class:`RBLNQwen2VL`, but swaps in
    :class:`Qwen3VLPromptMixin` to match upstream ``Qwen3VLChat`` —
    Qwen3's mixin widens VQA / Y-N coverage and excludes ``SSI_Bench``,
    among other small differences from Qwen2's mixin.

    The MRO ``[RBLNQwen3VL, Qwen3VLPromptMixin, RBLNQwen2VL,
    Qwen2VLPromptMixin, RBLNChatVLMBase, ...]`` means ``use_custom_prompt``
    and ``build_prompt`` resolve to Qwen3's overrides, while
    ``_use_custom_prompt`` is still initialized via the Qwen2 mixin's
    ``__init__`` (the two ``__init__`` bodies are identical, so either
    one is fine to run).

    Sampling defaults match upstream ``vlm.Qwen3VLChat``
    (qwen3_vl/model.py:52-83): much larger token budget (32k vs Qwen2's
    2k), looser nucleus sampling (top_p=0.8, top_k=20), and no
    ``do_sample`` flag — which means HF generate falls back to
    ``do_sample=False`` (greedy). The other sampling params become
    effective no-ops under greedy decoding but we pass them through
    verbatim to keep parity if a downstream caller flips ``do_sample``.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 32768,
        top_p: float = 0.8,
        top_k: int = 20,
        temperature: float = 0.01,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            **kwargs,
        )

    def _pick_rbln_class(self):
        from optimum.rbln import RBLNQwen3VLForConditionalGeneration
        return RBLNQwen3VLForConditionalGeneration
