from __future__ import annotations

from transformers.image_utils import load_image

from vlmeval.smp import listinstr
from .base import RBLNChatVLMBase


class RBLNIdefics3(RBLNChatVLMBase):
    """optimum-rbln backend for Idefics3 models.

    Upstream VLMEvalKit registers ``Idefics3-8B-Llama3`` against the
    ``IDEFICS2`` wrapper (see ``vlmeval/config.py:1711`` — comment:
    "Idefics3 follows Idefics2 Pattern"). That wrapper does **not** use
    ``processor.apply_chat_template`` — it constructs a literal
    ``"User:<image>...<end_of_utterance>\\nAssistant:"`` string with
    dataset-specific transformations applied to the user text.

    This class ports those dataset dispatch rules verbatim from
    upstream ``vlmeval/vlm/idefics.py`` so the same eight target
    benchmarks emit byte-identical prompts on RBLN and GPU.

    Dispatch table (from ``IDEFICS2.generate_inner``, lines 258-301):

    * ``MMBench_*`` family            -> :meth:`_build_prompt_mmbench`
    * ``MMMU_DEV_VAL``, ``MMMU_TEST`` -> :meth:`_build_prompt_mmmu`
    * ``MathVista_MINI``              -> :meth:`_build_prompt_mathvista`
    * ``MMStar``, ``SEEDBench_IMG``, ``AI2D_TEST``, ``ScienceQA_*``
                                      -> :meth:`_build_prompt_puremcq`
    * ``ChartQA_TEST``, ``DocVQA_*``, ``InfoVQA_*``, ``OCRVQA_*``,
      ``TextVQA_VAL``, ``MME``, ``MMVet``
                                      -> :meth:`_build_prompt_default` with ``add_brief=True``
    * ``HallusionBench``              -> ``_build_prompt_default`` with ``add_yes_or_no=True``
    * ``MLVU``, ``TempCompass``, ``MVBench``
                                      -> ``_build_prompt_default`` with ``change_the_img_place=True``
    * anything else                   -> ``_build_prompt_default``

    The 8 target benchmarks all have explicit branches above.
    """

    def __init__(
        self,
        model_path: str,
        # Matches upstream vlm.IDEFICS2 default (idefics.py:80): a 1024-token
        # budget. Idefics outputs are bounded by the ``<end_of_utterance>``
        # special token so 1024 is plenty for short-answer VQA / MCQ.
        max_new_tokens: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path, max_new_tokens=max_new_tokens, **kwargs,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_rbln_model_and_processor(self):
        from optimum.rbln import RBLNIdefics3ForConditionalGeneration
        from transformers import AutoProcessor

        model = self._from_pretrained(RBLNIdefics3ForConditionalGeneration)
        processor = AutoProcessor.from_pretrained(self.model_path)
        return model, processor

    # ------------------------------------------------------------------
    # Per-dataset prompt builders — verbatim from upstream IDEFICS2
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt_default(message, add_brief=False, add_yes_or_no=False,
                              change_the_img_place=False):
        if change_the_img_place:
            new_message = []
            for s in message:
                if s['type'] == 'image':
                    new_message.append(s)
            for s in message:
                if s['type'] == 'text':
                    new_message.append(s)
            message = new_message
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                images.append(load_image(msg['value']))
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value'].strip()
        if add_brief:
            prompt += '\nGive a very brief answer.'
        if add_yes_or_no:
            prompt += '\nAnswer yes or no.'
        prompt += '<end_of_utterance>\nAssistant:'
        return prompt, images

    @staticmethod
    def _build_prompt_puremcq(message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                images.append(load_image(msg['value']))
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    @staticmethod
    def _build_prompt_mmbench(message):
        replace_mapping = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                images.append(load_image(msg['value']))
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                # Swap hint and question (upstream lines 173-179)
                if instruction.startswith('Hint:'):
                    hint, question = instruction.split('\nQuestion:')
                    question, choices = question.split('\nChoices:')
                    instruction = (
                        'Question:' + question + '\n' + hint + '\nChoices:' + choices
                    )
                prompt += instruction
        prompt += '<end_of_utterance>\nAssistant: Answer:'
        return prompt, images

    @staticmethod
    def _build_prompt_mmmu(message):
        replace_mapping = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }
        prompt, images, img_counter = 'User: Question: ', [], 1
        # First pass: number each image upfront in the prompt
        for msg in message:
            if msg['type'] == 'image':
                prompt += f'<image {img_counter}>:<image>\n'
                img_counter += 1
        # Second pass: thread images + text in order, re-numbering inline
        img_counter = 1
        for msg in message:
            if msg['type'] == 'image':
                images.append(load_image(msg['value']))
                prompt += f' <image {img_counter}> '
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    @staticmethod
    def _build_prompt_mathvista(message):
        replace_mapping = {
            '(A) ': 'A. ',
            '(B) ': 'B. ',
            '(C) ': 'C. ',
            '(D) ': 'D. ',
            '(E) ': 'E. ',
            '(F) ': 'F. ',
            '(G) ': 'G. ',
            '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }
        prompt, images = 'User:', []
        for msg in message:
            if msg['type'] == 'image':
                images.append(load_image(msg['value']))
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction.strip()
        if 'A.' in prompt and 'B.' in prompt:
            prompt += '\nAnswer with the letter.'
        prompt += '<end_of_utterance>\nAssistant:'
        if 'A.' in prompt and 'B.' in prompt:
            prompt += ' Answer:'
        return prompt, images

    # ------------------------------------------------------------------
    # Dispatch + generation
    # ------------------------------------------------------------------

    _MMBENCH_DATASETS = frozenset({
        'MMBench_DEV_EN', 'MMBench_DEV_EN_V11',
        'MMBench_TEST_EN', 'MMBench_TEST_EN_V11',
        'MMBench_DEV_CN', 'MMBench_DEV_CN_V11',
        'MMBench_TEST_CN', 'MMBench_TEST_CN_V11',
        'MMBench', 'MMBench_V11', 'MMBench_CN', 'MMBench_CN_V11',
    })
    _MMMU_DATASETS = frozenset({'MMMU_DEV_VAL', 'MMMU_TEST'})
    _MATHVISTA_DATASETS = frozenset({'MathVista_MINI'})
    _PUREMCQ_DATASETS = frozenset({
        'MMStar', 'SEEDBench_IMG', 'AI2D_TEST', 'ScienceQA_VAL', 'ScienceQA_TEST',
    })
    _BRIEF_VQA_DATASETS = frozenset({
        'MME', 'MMVet',
        'OCRVQA_TEST', 'OCRVQA_TESTCORE',
        'TextVQA_VAL',
        'ChartQA_TEST',
        'DocVQA_VAL', 'DocVQA_TEST',
        'InfoVQA_VAL', 'InfoVQA_TEST',
    })
    _VIDEO_DATASETS_TOKENS = ('MLVU', 'TempCompass', 'MVBench')

    def _format_for_dataset(self, message, dataset):
        if dataset in self._MMBENCH_DATASETS:
            return self._build_prompt_mmbench(message)
        if dataset in self._MMMU_DATASETS:
            return self._build_prompt_mmmu(message)
        if dataset in self._MATHVISTA_DATASETS:
            return self._build_prompt_mathvista(message)
        if dataset in self._PUREMCQ_DATASETS:
            return self._build_prompt_puremcq(message)
        if dataset in self._BRIEF_VQA_DATASETS:
            return self._build_prompt_default(message, add_brief=True)
        if dataset == 'HallusionBench':
            return self._build_prompt_default(message, add_yes_or_no=True)
        if dataset is not None and listinstr(list(self._VIDEO_DATASETS_TOKENS), dataset):
            return self._build_prompt_default(message, change_the_img_place=True)
        return self._build_prompt_default(message)

    def generate_inner(self, message, dataset=None):
        prompt_text, images = self._format_for_dataset(message, dataset)
        if self.verbose:
            self._debug_log(prompt_text, color='red')

        inputs = self.processor(
            text=prompt_text,
            images=images if images else None,
            return_tensors='pt',
        )
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        response = self._finalize_response(inputs, generated_ids)
        if self.verbose:
            self._debug_log(response, color='green')
        return response
