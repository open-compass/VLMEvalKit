from __future__ import annotations

from .base import RBLNVLMBase


class RBLNBlip2(RBLNVLMBase):
    """optimum-rbln backend for BLIP-2 models.

    BLIP-2 is a single-image VQA model (no chat template). Two
    family-specific behaviours, both verified against optimum-rbln 0.10.3
    + Salesforce/blip2-opt-2.7b on an RBLN NPU:

    * **Prompt format.** BLIP-2-OPT only emits an answer when the input
      follows the OPT VQA convention ``"Question: {q} Answer:"``. Fed a
      bare question (or a dataset's raw ``"{q}\\nAnswer ..."`` instruction)
      it returns an empty string. ``generate_inner`` therefore wraps the
      aggregated prompt in that convention.
    * **Decode trim.** ``model.generate`` returns the full sequence
      *including* the prompt tokens, so the prompt-trim that the default
      ``generate_inner`` applies (``_DECODE_TRIM=True``, inherited) is what
      strips the echoed prompt back off. ``_DECODE_STRIP=True`` drops the
      trailing whitespace to match rbln-model-zoo's ``inference.py``.
    """

    INTERLEAVE = False
    _DECODE_STRIP = True

    def _load_rbln_model_and_processor(self):
        from optimum.rbln import RBLNBlip2ForConditionalGeneration
        from transformers import AutoProcessor

        model = self._from_pretrained(RBLNBlip2ForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    def generate_inner(self, message, dataset=None):
        prompt, image = self.message_to_promptimg(message, dataset=dataset)
        if image is None:
            raise ValueError(f"{type(self).__name__} requires an image input.")
        # BLIP-2-OPT VQA convention — without it the model returns "".
        prompt = f"Question: {prompt} Answer:"
        inputs = self.processor(
            text=prompt,
            images=self._load_image(image),
            return_tensors='pt',
        )
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        return self._finalize_response(inputs, generated_ids)
