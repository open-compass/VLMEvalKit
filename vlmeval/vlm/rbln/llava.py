from __future__ import annotations
import string

import pandas as pd
from PIL import Image

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from .base import RBLNChatVLMBase

# Verbatim from upstream vlmeval/vlm/llava/llava.py:34-37. The trailing
# space inside the closing period is intentional — it sits between the
# system prompt and "USER:" in the final concatenation.
LLAVA_V1_5_SYSTEM_PROMPT = (
    'A chat between a curious human and an artificial intelligence assistant. '
    "The assistant gives helpful, detailed, and polite answers to the human's questions. "
)


class RBLNLlava(RBLNChatVLMBase):
    """optimum-rbln backend for LLaVA 1.5 (llava-hf/llava-1.5-*).

    Targets ``RBLNLlavaForConditionalGeneration`` — distinct from
    ``RBLNLlavaNextForConditionalGeneration`` used by LLaVA-Next.

    Prompt parity with upstream ``vlm.LLaVA`` (vlmeval/vlm/llava/llava.py).
    Every layer of upstream's prompt path is ported:

    * Default ``system_prompt`` — the Vicuna preamble verbatim.
    * ``use_custom_prompt(dataset)`` — True for MCQ datasets.
    * ``build_prompt(line, dataset)`` — appends per-language MCQ /
      free-form instruction (Chinese vs English chosen by
      :func:`cn_string`).
    * ``generate_inner`` — bypasses ``apply_chat_template`` and emits the
      literal ``"{system_prompt}USER: {content} ASSISTANT: "`` string
      upstream uses. Image tokens are inlined as ``" <image> "``
      (space-padded on both sides) by :meth:`_concat_tilist` to match
      upstream's :py:meth:`vlm.LLaVA.concat_tilist`.
    """

    def __init__(
        self,
        model_path: str,
        system_prompt: str | None = LLAVA_V1_5_SYSTEM_PROMPT,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path, system_prompt=system_prompt, **kwargs,
        )

    def _load_rbln_model_and_processor(self):
        from optimum.rbln import RBLNLlavaForConditionalGeneration
        from transformers import AutoProcessor

        model = self._from_pretrained(RBLNLlavaForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    # ------------------------------------------------------------------
    # Prompt construction — mirrors upstream vlm.LLaVA
    # ------------------------------------------------------------------

    def use_custom_prompt(self, dataset: str) -> bool:
        assert dataset is not None
        return DATASET_TYPE(dataset) == 'MCQ'

    def build_prompt(self, line, dataset: str | None = None):
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。'
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                '\n请直接回答问题。'
                if cn_string(prompt)
                else '\nAnswer the question directly.'
            )

        paths = tgt_path if isinstance(tgt_path, list) else [tgt_path]
        message = [dict(type='image', value=p) for p in paths]
        message.append(dict(type='text', value=prompt))
        return message

    # ------------------------------------------------------------------
    # Generation — literal Vicuna USER/ASSISTANT format
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_tilist(message):
        """Mirror upstream vlm.LLaVA.concat_tilist (llava.py:128-136).

        Concatenates text segments and inlines images as ``" <image> "``
        (space-padded on both sides). Returns ``(text, image_paths)``.
        """
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])
        return text, images

    def generate_inner(self, message, dataset=None):
        content, image_paths = self._concat_tilist(message)
        prompt = f'{self.system_prompt}USER: {content} ASSISTANT: '

        images = [Image.open(p).convert('RGB') for p in image_paths]
        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors='pt',
        )
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        return self._finalize_response(inputs, generated_ids)
