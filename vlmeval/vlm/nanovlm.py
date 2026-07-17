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

_VISIONPSY_VLLM_INSTALL_MSG = (
    'NanoVLM with use_vllm=True additionally needs the VisionPsy vLLM plugin '
    '(registers the architecture with vLLM):\n'
    '  pip install "git+https://github.com/tether-ai-research/'
    'qvac-visionpsy-nano#subdirectory=vllm-inference"'
)


def _ensure_nanovlm_importable():
    nanovlm_path = os.environ.get('NANOVLM_PATH', '')
    if nanovlm_path and nanovlm_path not in sys.path:
        sys.path.insert(0, nanovlm_path)
        return
    # Without NANOVLM_PATH, fall back to the reference preprocessing bundled
    # with the VisionPsy vLLM plugin (enough for the use_vllm / Hub-packaged
    # paths; the eager path still needs the full nanoVLM clone).
    try:
        import visionpsy_vllm
        ref = os.path.join(os.path.dirname(visionpsy_vllm.__file__), 'reference')
        if os.path.isdir(ref) and ref not in sys.path:
            sys.path.append(ref)
    except ImportError:
        pass


class NanoVLM(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='lusxvr/nanoVLM-460M-8k', use_vllm=False, **kwargs):
        super().__init__()
        _ensure_nanovlm_importable()
        self.use_vllm = use_vllm

        raw_probe = self._read_raw_config(model_path)
        # 'visionpsynano' is the current packaged model_type; 'visionpsy' is
        # accepted for checkpoints packaged before the rename.
        self.is_visionpsy = bool(raw_probe) and raw_probe.get('model_type') in ('visionpsy', 'visionpsynano')
        if self.is_visionpsy and not use_vllm:
            # Hub-packaged VisionPsy checkpoint (trust_remote_code): the bundled
            # processor owns tokenization, tiling and the chat template, so none
            # of the nanoVLM reference plumbing below is needed.
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_path, trust_remote_code=True, dtype=torch.float32,
            ).to('cuda').eval()
            if hasattr(self.vlm, 'apply_eager_profile'):
                self.vlm.apply_eager_profile()
            self.processor = AutoProcessor.from_pretrained(
                model_path, trust_remote_code=True,
            )
            kwargs_default = {'max_new_tokens': 2048}
            kwargs_default.update(kwargs)
            self.kwargs = kwargs_default
            warnings.warn(f'NanoVLM (VisionPsy hub package) kwargs: {self.kwargs}')
            return

        if use_vllm:
            # Build the vLLM engine FIRST: HF-tokenizers and torch spawn worker
            # threads, and vLLM forks its engine-core subprocess -- forking a
            # threaded parent deadlocks the child. Only json/config reads happen
            # before the engine is up.
            raw_cfg = self._read_raw_config(model_path)
            if raw_cfg is None:
                raise RuntimeError(f'cannot read config.json for {model_path}')
            self.llm = self._build_vllm_engine(
                model_path, raw_cfg.get('lm_max_position_embeddings', 8192),
                self.is_visionpsy,
            )
            self.vlm = None
            try:
                from models.config import VLMConfig
            except ImportError:
                raise ImportError(_NANOVLM_INSTALL_MSG)
            from dataclasses import fields as dc_fields
            valid = {fld.name for fld in dc_fields(VLMConfig)}
            self.cfg = VLMConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
            saved_keys = set(raw_cfg.keys())
        else:
            try:
                from models.vision_language_model import VisionLanguageModel
            except ImportError:
                raise ImportError(_NANOVLM_INSTALL_MSG)
            self.vlm = VisionLanguageModel.from_pretrained(model_path).to('cuda').eval()
            self.cfg = self.vlm.cfg
            # Older checkpoints (e.g. nanoVLM-222M) were trained without image
            # splitting and don't have max_img_size / vlm_extra_tokens in their
            # saved config.json. VLMConfig fills in defaults for missing keys, so
            # we check which keys were actually present in the saved checkpoint.
            # Read the raw config.json directly: stock nanoVLM's from_pretrained
            # does not expose which keys were saved.
            saved_keys = self._read_saved_config_keys(model_path)
            if saved_keys is None:
                saved_keys = getattr(self.vlm, '_saved_config_keys', None)

        from data.processors import get_image_processor, get_tokenizer
        try:
            # Resolves the nano/flash resize policy (flash needs a min-side
            # clamp); absent in stock nanoVLM checkouts, which predate flash.
            from data.processors import apply_model_preprocess
        except ImportError:
            apply_model_preprocess = None

        extra_tokens = getattr(self.cfg, 'vlm_extra_tokens', None)
        chat_template = getattr(self.cfg, 'lm_chat_template', None)
        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer, extra_tokens, chat_template,
        )

        has_image_splitting = saved_keys is not None and 'vlm_extra_tokens' in saved_keys
        min_side = None
        if has_image_splitting:
            if apply_model_preprocess is not None:
                apply_model_preprocess(self.cfg)
                min_side = getattr(self.cfg, 'resize_min_side_len', None)
            max_img = (getattr(self.cfg, 'inference_max_img_size', None)
                       or getattr(self.cfg, 'max_img_size', self.cfg.vit_img_size))
            resize_to_max = getattr(self.cfg, 'resize_to_max_side_len', False)
        else:
            max_img = self.cfg.vit_img_size
            resize_to_max = False
        if min_side is not None:
            self.image_processor = get_image_processor(
                max_img, self.cfg.vit_img_size, resize_to_max, min_side
            )
        else:
            # Stock nanoVLM's get_image_processor has no min_side_len parameter.
            self.image_processor = get_image_processor(
                max_img, self.cfg.vit_img_size, resize_to_max
            )

        kwargs_default = {'max_new_tokens': 2048}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'NanoVLM kwargs: {self.kwargs}')
        if not use_vllm:
            torch.cuda.empty_cache()

    @staticmethod
    def _read_raw_config(model_path):
        """Return the checkpoint's raw config.json as a dict, or None."""
        import json
        import os.path as osp
        try:
            if osp.isdir(model_path):
                cfg_path = osp.join(model_path, 'config.json')
            else:
                from huggingface_hub import hf_hub_download
                cfg_path = hf_hub_download(repo_id=model_path, filename='config.json')
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            return None

    @classmethod
    def _read_saved_config_keys(cls, model_path):
        """Return the key set of the checkpoint's raw config.json, or None."""
        raw = cls._read_raw_config(model_path)
        return set(raw.keys()) if raw is not None else None

    @staticmethod
    def _build_vllm_engine(model_path, max_model_len, is_visionpsy):
        """Build an in-process vLLM engine on a Hub-packaged VisionPsy checkpoint.

        The architecture is out-of-tree for vLLM; the visionpsy-vllm-plugin
        package registers it, and the Hub-packaged checkpoint then loads
        directly (HF repo id or local folder).
        """
        if not is_visionpsy:
            raise ValueError(
                'use_vllm expects a Hub-packaged VisionPsy checkpoint '
                '(config model_type "visionpsynano"). Package raw training '
                'checkpoints first: hf-inference/scripts/package_hub_repo.py.'
            )
        try:
            import visionpsy_vllm
        except ImportError:
            raise ImportError(_VISIONPSY_VLLM_INSTALL_MSG)
        visionpsy_vllm.register()
        from vllm import LLM

        return LLM(
            model=model_path,
            dtype='float32',
            max_model_len=max_model_len,
            limit_mm_per_prompt={'image': 128},
            enforce_eager=True,
        )

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
        elif dataset in (
            'ChartQA_TEST', 'DocVQA_VAL', 'DocVQA_TEST',
            'TextVQA_VAL', 'TextVQA_TEST',
        ):
            prompt, images = self._build_prompt_default(message, add_direct=True)
        elif dataset in (
            'MME', 'OCRVQA_TEST', 'OCRVQA_TESTCORE',
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
        # use_vllm takes precedence: a Hub-packaged checkpoint under --use-vllm
        # runs through the vLLM engine, not the bundled transformers model.
        if self.use_vllm:
            return self._run_generation_vllm(prompt, pil_images)
        if self.is_visionpsy:
            return self._run_generation_visionpsy(prompt, pil_images)
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

    def _run_generation_visionpsy(self, prompt, pil_images):
        """Greedy generation via the Hub-packaged VisionPsy model.

        The bundled VisionPsyProcessor reproduces the reference preprocessing
        (tiling + image-token string + chat template), so the adapter only
        supplies the per-dataset prompt text and the raw images.
        """
        inputs = self.processor(
            images=pil_images if pil_images else None,
            text=prompt,
            return_tensors='pt',
        )
        inputs = {
            k: (v.to('cuda') if torch.is_tensor(v) else v)
            for k, v in inputs.items()
            if v is not None
        }
        inputs.pop('pixel_values', None)
        with torch.inference_mode():
            generated_ids = self.vlm.generate(
                **inputs,
                max_new_tokens=self.kwargs.get('max_new_tokens', 2048),
                greedy=True,
            )
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def _run_generation_vllm(self, prompt, pil_images):
        """Greedy generation through the in-process vLLM engine.

        Same preprocessing as the eager path: the image is tiled client-side and
        the prompt carries the global/tile position tokens with ONE image
        placeholder per tile -- the plugin's multimodal processor expands each
        placeholder to mp_image_token_length image tokens inside vLLM.
        """
        _ensure_nanovlm_importable()
        try:
            from data.processors import get_image_string
        except ImportError:
            raise ImportError(_NANOVLM_INSTALL_MSG)
        from vllm import SamplingParams

        image_string = ''
        tiles = []
        for img in pil_images:
            processed, ratio = self.image_processor(img)
            if (not hasattr(self.tokenizer, 'global_image_token')
                    and ratio[0] * ratio[1] == len(processed) - 1):
                processed = processed[1:]
            image_string += get_image_string(self.tokenizer, [ratio], 1)
            tiles.extend(processed)
        pil_tiles = [
            Image.fromarray(
                (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
            )
            for t in tiles
        ]

        messages = [{'role': 'user', 'content': image_string + prompt}]
        full_prompt = self.tokenizer.apply_chat_template(
            [messages], tokenize=False, add_generation_prompt=True,
        )
        if isinstance(full_prompt, list):
            full_prompt = full_prompt[0]

        inputs = {'prompt': full_prompt}
        if pil_tiles:
            inputs['multi_modal_data'] = {'image': pil_tiles}
        outputs = self.llm.generate(
            inputs,
            SamplingParams(
                temperature=0.0,
                max_tokens=self.kwargs.get('max_new_tokens', 2048),
            ),
            use_tqdm=False,
        )
        return outputs[0].outputs[0].text.strip()

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

    def _build_prompt_default(self, message, add_brief=False, add_yes_or_no=False, add_direct=False):
        images = self._load_images(message)
        text = self._get_text(message)
        if add_brief:
            text += '\nGive a very brief answer.'
        if add_yes_or_no:
            text += '\nAnswer yes or no.'
        if add_direct:
            text += '\nPlease answer directly with only the final answer, do not give any explanation.'
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
