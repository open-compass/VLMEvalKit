from __future__ import annotations

from .qwen3_vl import RBLNQwen3VL


def strip_think(response: str) -> str:
    """Keep only the text after the last ``</think>`` (reasoning dropped)."""
    if '</think>' in response:
        response = response.split('</think>')[-1]
    return response.strip()


class RBLNQwen3_5(RBLNQwen3VL):
    """optimum-rbln backend for Qwen3.5 / 3.6 / 3.8 (``Qwen3_5ForConditionalGeneration``).

    Upstream registers Qwen3.5 under ``vlm.Qwen3VLChat`` with
    ``use_custom_prompt=False`` (vlmeval/config.py ``qwen3_5_series``) and
    preprocesses images with ``image_patch_size=16`` and ``do_resize=False``
    (qwen3_vl/model.py). This wrapper keeps the Qwen3-VL inference path and
    matches those settings.

    Thinking is off by default (``enable_thinking=False``), as in upstream's
    InstructMode configs; with it on, the response is cut after ``</think>``.
    Decoding is greedy, like ``RBLNQwen3VL``: upstream's vLLM sampling
    parameters are passed through but inactive.
    """

    _IMAGE_PATCH_SIZE = 16
    _PROCESSOR_DO_RESIZE = False

    def __init__(
        self,
        model_path: str,
        use_custom_prompt: bool = False,
        enable_thinking: bool = False,
        chat_template_kwargs: dict | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        **kwargs,
    ) -> None:
        chat_template_kwargs = {'enable_thinking': enable_thinking, **(chat_template_kwargs or {})}
        super().__init__(
            model_path=model_path,
            use_custom_prompt=use_custom_prompt,
            chat_template_kwargs=chat_template_kwargs,
            top_p=top_p,
            top_k=top_k,
            **kwargs,
        )

    def _pick_rbln_class(self):
        from optimum.rbln import RBLNQwen3_5ForConditionalGeneration
        return RBLNQwen3_5ForConditionalGeneration

    def _postprocess_response(self, response: str) -> str:
        return strip_think(response)
