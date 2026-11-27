from __future__ import annotations

from .qwen2_vl import RBLNQwen2VL

# Matches the system message upstream ``vlm.Cosmos.generate_inner``
# unconditionally prepends to every call (vlmeval/vlm/cosmos.py:46-52).
# Cosmos-Reason1 is a reasoning model — the explicit format directive is
# what makes it emit ``<think>...</think>\n\n<answer>...</answer>`` outputs
# that downstream judges expect.
COSMOS_REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question in the following format:\n"
    "<think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>."
)


class RBLNCosmosReason1(RBLNQwen2VL):
    """optimum-rbln backend for nvidia/Cosmos-Reason1-7B.

    Cosmos-Reason1 is built on the Qwen2.5-VL architecture, so the
    existing Qwen wrapper handles model loading (via the architectures-
    based dispatch in ``_pick_rbln_class``), chat templating, and video
    frame-rate threading from P1. This subclass changes:

    1. video sampling defaults — fps=4, total_pixels=6_422_528 to match
       rbln-model-zoo/.../cosmos-reason1/inference.py.
    2. ``system_prompt`` — defaults to ``COSMOS_REASONING_SYSTEM_PROMPT``
       so RBLN emits the same ``<think>``/``<answer>`` formatted output
       as upstream ``vlm.Cosmos``.
    3. ``use_custom_prompt`` — returns False so RBLN uses the dataset's
       own ``build_prompt`` rather than the Qwen mixin (mirrors upstream
       Cosmos's plain ``BaseModel`` inheritance).
    """

    def __init__(
        self,
        model_path: str,
        fps: int | None = 4,
        total_pixels: int | None = 6_422_528,
        system_prompt: str | None = COSMOS_REASONING_SYSTEM_PROMPT,
        # Sampling defaults match upstream ``vlm.Cosmos.__init__``
        # (vlmeval/vlm/cosmos.py:19-23): the model is a reasoning VLM and
        # was tuned with these specific values. Greedy decoding tends to
        # collapse the ``<think>...</think><answer>...</answer>``
        # structure, so we keep the same temperature / nucleus / repetition
        # configuration as the upstream vLLM path.
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            fps=fps,
            total_pixels=total_pixels,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            **kwargs,
        )

    def use_custom_prompt(self, dataset: str) -> bool:
        return False
