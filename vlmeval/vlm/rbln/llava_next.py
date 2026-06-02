from __future__ import annotations
import string

import pandas as pd

from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp import cn_string
from .base import RBLNChatVLMBase


class RBLNLlavaNext(RBLNChatVLMBase):
    """optimum-rbln backend for LLaVA-Next family models.

    Prompt parity with upstream ``vlm.LLaVA_Next`` (vlmeval/vlm/llava/llava.py
    lines 227-397). Three points to mirror:

    * ``use_custom_prompt(dataset)`` — True for MCQ datasets so that
      ``model.build_prompt`` wins over the dataset's ``build_prompt``.
    * ``build_prompt(line, dataset)`` — appends per-language MCQ /
      free-form instruction (Chinese vs English chosen by
      :func:`cn_string`) identical to upstream's. The MCQ wording differs
      from LLaVA-v1.5's only in wording style and is duplicated verbatim
      from upstream to keep token sequences identical.
    * ``generate_inner`` — upstream uses ``processor.apply_chat_template``
      on a single user turn with interleaved image/text content. The
      default ``RBLNChatVLMBase.generate_inner`` already does exactly
      this, so it's inherited unchanged.

    Upstream also defines an ``apply_prompt_template`` method that picks
    a Vicuna / Mistral / Yi-34B-style wrapper template based on the model
    path. That method exists in the upstream file but **is not invoked
    from the active code path** — the live ``generate_inner`` only uses
    ``apply_chat_template``. The HF processor's ``chat_template.jinja``
    shipped with each ``llava-hf/llava-v1.6-*-hf`` checkpoint already
    encodes the right Vicuna / Mistral / Yi-34B wrapping, so no manual
    template wrapping is necessary on either backend.
    """

    def _load_rbln_model_and_processor(self):
        from optimum.rbln import RBLNLlavaNextForConditionalGeneration
        from transformers import AutoProcessor

        model = self._from_pretrained(RBLNLlavaNextForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    # ------------------------------------------------------------------
    # Output post-processing — mirrors upstream vlm.LLaVA_Next.output_process
    # (llava.py:312-330). Upstream decodes with the (buggy) kwarg
    # ``skip_special_token=True`` — note the missing ``s`` — so special
    # tokens actually survive into the answer. Their ``output_process``
    # then strips role markers / EOS variants from both sides. We pass
    # ``skip_special_tokens=True`` correctly here, but still run the same
    # cleanup so any role markers the model emitted (e.g. extra
    # ``ASSISTANT:`` turns) get stripped identically.
    # ------------------------------------------------------------------

    @staticmethod
    def _output_process(answer: str) -> str:
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()
        elif '<|end_header_id|>\n\n' in answer:
            answer = answer.split('<|end_header_id|>\n\n')[2].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        elif '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        elif '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        # Upstream also drops ``<unk>`` markers (llava.py:396).
        return answer.replace('<unk>', '')

    def _finalize_response(self, inputs, generated_ids) -> str:
        out = super()._finalize_response(inputs, generated_ids)
        return self._output_process(out)

    # ------------------------------------------------------------------
    # Prompt construction — mirrors upstream vlm.LLaVA_Next
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
