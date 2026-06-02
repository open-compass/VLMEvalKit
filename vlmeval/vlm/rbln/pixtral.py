from __future__ import annotations

from .base import RBLNChatVLMBase


class RBLNPixtral(RBLNChatVLMBase):
    """optimum-rbln backend for Pixtral-12B (mistral-community/pixtral-12b).

    Uses the PixtralProcessor chat template from transformers (>=4.45)
    so multi-image interleaved input lines up with the Mistral
    ``<s>[INST]...[/INST]`` format that Pixtral expects.

    Prompt parity with upstream ``vlm.Pixtral`` (vlmeval/vlm/pixtral.py)
    is achieved at the token-sequence level rather than via shared code:

    * Upstream loads ``mistralai/Pixtral-12B-2409`` (Mistral-native)
      with ``mistral_common`` / ``mistral_inference`` and builds
      ``UserMessage(content=[TextChunk, ImageURLChunk, ...])``, then
      tokenizes via ``MistralTokenizer.encode_chat_completion``.
    * RBLN loads ``mistral-community/pixtral-12b`` (HF-converted) with
      ``AutoProcessor`` and uses ``apply_chat_template`` on the canonical
      ``[{role:'user', content:[{type:'image'}|{type:'text',text:...}]}]``
      messages structure.

    Both paths converge on the same Mistral ``<s>[INST] ... [/INST]``
    token sequence because ``chat_template.jinja`` shipped with the HF
    Pixtral checkpoint is the official reproduction of ``mistral_common``'s
    output format. Neither side injects a system prompt — the canonical
    Mistral chat format puts everything in a single user turn.

    No code-level differences are required for prompt parity; this
    docstring is the load-bearing argument.
    """

    def _load_rbln_model_and_processor(self):
        from transformers import AutoProcessor

        model = self._from_pretrained(self._resolve_rbln_class())
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor
