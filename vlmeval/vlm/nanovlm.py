import os
import sys
import warnings

import torch
from PIL import Image

from .base import BaseModel

_NANOVLM_INSTALL_MSG = (
    'nanoVLM is not pip-installable. To use this model:\n'
    '  1. Clone the repo:  git clone https://github.com/huggingface/nanoVLM\n'
    '  2. Set the env var: export NANOVLM_PATH=/path/to/nanoVLM\n'
    '  3. Then run VLMEvalKit as usual.'
)


def _ensure_nanovlm_importable():
    nanovlm_path = os.environ.get('NANOVLM_PATH', '')
    if nanovlm_path and nanovlm_path not in sys.path:
        sys.path.insert(0, nanovlm_path)


class NanoVLM(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='lusxvr/nanoVLM-460M-8k', **kwargs):
        super().__init__()
        _ensure_nanovlm_importable()
        try:
            from models.vision_language_model import VisionLanguageModel
        except ImportError:
            raise ImportError(_NANOVLM_INSTALL_MSG)
        from data.processors import get_image_processor, get_tokenizer

        self.vlm = VisionLanguageModel.from_pretrained(model_path).to('cuda').eval()
        self.cfg = self.vlm.cfg

        extra_tokens = getattr(self.cfg, 'vlm_extra_tokens', None)
        chat_template = getattr(self.cfg, 'lm_chat_template', None)
        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer, extra_tokens, chat_template,
        )

        # Older checkpoints (e.g. nanoVLM-222M) were trained without image splitting
        # and don't have max_img_size / vlm_extra_tokens in their saved config.json.
        # VLMConfig fills in defaults for missing keys, so we check which keys were
        # actually present in the saved checkpoint to detect old vs new checkpoints.
        saved_keys = getattr(self.vlm, '_saved_config_keys', None)
        has_image_splitting = saved_keys is not None and 'vlm_extra_tokens' in saved_keys
        if has_image_splitting:
            max_img = getattr(self.cfg, 'max_img_size', self.cfg.vit_img_size)
            resize_to_max = getattr(self.cfg, 'resize_to_max_side_len', False)
        else:
            max_img = self.cfg.vit_img_size
            resize_to_max = False
        self.image_processor = get_image_processor(
            max_img, self.cfg.vit_img_size, resize_to_max
        )

        kwargs_default = {'max_new_tokens': 2048}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'NanoVLM kwargs: {self.kwargs}')
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Core generation entry point required by VLMEvalKit
    # ------------------------------------------------------------------
    def generate_inner(self, message, dataset=None):
        if dataset in self._MMBENCH_DATASETS:
            prompt, images = self._build_prompt_mmbench(message)
        elif dataset in ('MMMU_DEV_VAL', 'MMMU_TEST'):
            prompt, images = self._build_prompt_mmmu(message)
        elif dataset in ('MathVista_MINI',):
            prompt, images = self._build_prompt_mathvista(message)
        elif dataset in ('ChartQA_TEST',):
            prompt, images = self._build_prompt_chartqa(message)
        elif dataset in ('DocVQA_VAL', 'DocVQA_TEST'):
            prompt, images = self._build_prompt_docvqa(message)
        elif dataset in ('TextVQA_VAL', 'TextVQA_TEST'):
            prompt, images = self._build_prompt_textvqa(message)
        elif dataset in (
            'MME', 'MMVet', 'OCRVQA_TEST', 'OCRVQA_TESTCORE',
            'InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench',
        ):
            prompt, images = self._build_prompt_default(message, add_brief=True)
        elif dataset == 'HallusionBench':
            prompt, images = self._build_prompt_default(message, add_yes_or_no=True)
        elif dataset in (
            'MMStar', 'SEEDBench_IMG', 'AI2D_TEST',
            'ScienceQA_VAL', 'ScienceQA_TEST',
        ):
            prompt, images = self._build_prompt_puremcq(message)
        elif dataset in ('RealWorldQA',):
            prompt, images = self._build_prompt_puremcq(message)
        elif dataset in ('POPE',):
            prompt, images = self._build_prompt_default(message, add_brief=True)
        elif dataset in ('BLINK',):
            prompt, images = self._build_prompt_default(message, add_brief=True)
        elif dataset in ('MM-IFEval',):
            prompt, images = self._build_prompt_default(message)
        else:
            prompt, images = self._build_prompt_default(message)

        return self._run_generation(prompt, images)

    # ------------------------------------------------------------------
    # Shared generation logic
    # ------------------------------------------------------------------
    def _run_generation(self, prompt, pil_images):
        _ensure_nanovlm_importable()
        try:
            from data.processors import get_image_string
        except ImportError:
            raise ImportError(_NANOVLM_INSTALL_MSG)

        all_processed = []
        all_ratios = []
        for img in pil_images:
            processed, ratio = self.image_processor(img)
            if (not hasattr(self.tokenizer, 'global_image_token')
                    and ratio[0] * ratio[1] == len(processed) - 1):
                processed = processed[1:]
            all_processed.append(processed)
            all_ratios.append(ratio)

        image_string = ''
        for ratio in all_ratios:
            image_string += get_image_string(
                self.tokenizer, [ratio], self.cfg.mp_image_token_length
            )

        user_content = image_string + prompt
        messages = [{'role': 'user', 'content': user_content}]
        full_prompt = self.tokenizer.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True,
        )
        if isinstance(full_prompt, list):
            full_prompt = full_prompt[0]

        inputs = self.tokenizer(
            [full_prompt],
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=self.cfg.lm_max_position_embeddings,
        )
        input_ids = inputs['input_ids'].to('cuda')
        attention_mask = inputs['attention_mask'].to('cuda')

        images_for_model = all_processed if all_processed else None

        max_new = self.kwargs.get('max_new_tokens', 2048)
        generated_ids = self.vlm.generate(
            input_ids,
            images_for_model,
            attention_mask,
            max_new_tokens=max_new,
            greedy=True,
        )

        text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    # ------------------------------------------------------------------
    # Per-dataset prompt builders  (ported from SmolVLM2 adapter)
    # ------------------------------------------------------------------
    _MMBENCH_DATASETS = {
        'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN',
        'MMBench_TEST_CN', 'MMBench', 'MMBench_CN',
        'MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11',
        'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11',
        'MMBench_V11', 'MMBench_CN_V11', 'CCBench',
    }

    @staticmethod
    def _load_images(message):
        images = []
        for msg in message:
            if msg['type'] == 'image':
                img = Image.open(msg['value']).convert('RGB')
                images.append(img)
        return images

    @staticmethod
    def _get_text(message):
        return '\n'.join(m['value'].strip() for m in message if m['type'] == 'text')

    def _build_prompt_default(self, message, add_brief=False, add_yes_or_no=False):
        images = self._load_images(message)
        text = self._get_text(message)
        if add_brief:
            text += '\nGive a very brief answer.'
        if add_yes_or_no:
            text += '\nAnswer yes or no.'
        return text, images

    def _build_prompt_puremcq(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        replacements = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text += '\nAnswer:'
        return text, images

    def _build_prompt_mmbench(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        replacements = {
            '\nOptions:': '\nChoices:',
            'Please select the correct answer from the options above.': 'Answer with a letter.',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if text.startswith('Hint:'):
            try:
                hint, rest = text.split('\nQuestion:')
                question, choices = rest.split('\nChoices:')
                text = 'Question:' + question + '\n' + hint + '\nChoices:' + choices
            except ValueError:
                pass
        text += '\nAnswer:'
        return text, images

    def _build_prompt_mmmu(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        replacements = {
            'Question:': '',
            'Please select the correct answer from the options above.': 'Answer with the letter.',
            '\nOptions:': '\nChoices:',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        text = 'Question: ' + text.strip()
        if 'A.' in text and 'B.' in text:
            text += '\nAnswer:'
        return text, images

    def _build_prompt_mathvista(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        replacements = {
            '(A) ': 'A. ', '(B) ': 'B. ', '(C) ': 'C. ', '(D) ': 'D. ',
            '(E) ': 'E. ', '(F) ': 'F. ', '(G) ': 'G. ', '(H) ': 'H. ',
            '\nOptions:': '\nChoices:',
            'Hint: ': '',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        if 'A.' in text and 'B.' in text:
            text += '\nAnswer:'
        return text, images

    def _build_prompt_chartqa(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        prefix = (
            'For the question below, follow the following instructions:\n'
            '-The answer should contain as few words as possible.\n'
            "-Don't paraphrase or reformat the text you see in the image.\n"
            '-Answer a binary question with Yes or No.\n'
            '-When asked to give a numerical value, provide a number like 2 instead of Two.\n'
            '-If the final answer has two or more items, provide it in the list format like [1, 2].\n'
            '-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n'
            '-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n'
            "-Don't include any units in the answer.\n"
            '-Do not include any full stops at the end of the answer.\n'
            '-Try to include the full label from the graph when asked about an entity.\n'
            'Question: '
        )
        return prefix + text, images

    def _build_prompt_docvqa(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        prefix = (
            'Give a short and terse answer to the following question. '
            'Do not paraphrase or reformat the text you see in the image. '
            'Do not include any full stops. '
            'Just give the answer without additional explanation. Question: '
        )
        return prefix + text, images

    def _build_prompt_textvqa(self, message):
        images = self._load_images(message)
        text = self._get_text(message)
        prefix = (
            'Answer the following question about the image using as few words as possible. '
            'Follow these additional instructions:\n'
            '-Always answer a binary question with Yes or No.\n'
            '-When asked what time it is, reply with the time seen in the image.\n'
            '-Do not put any full stops at the end of the answer.\n'
            '-Do not put quotation marks around the answer.\n'
            '-An answer with one or two words is favorable.\n'
            '-Do not apply common sense knowledge. The answer can be found in the image.\n'
            'Question: '
        )
        return prefix + text, images
