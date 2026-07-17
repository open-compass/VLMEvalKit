import os
import sys
import warnings

import torch
from PIL import Image

from .base import BaseModel

_VLLM_PLUGIN_INSTALL_MSG = (
    'VisionPsy with use_vllm=True needs the VisionPsy vLLM plugin, which '
    'registers the architecture with vLLM:\n'
    '  pip install "git+https://github.com/tether-ai-research/'
    'qvac-visionpsy-nano#subdirectory=vllm-inference"'
)


class VisionPsy(BaseModel):
    """VisionPsy-Nano, a ~460M-parameter vision-language model for edge devices.

    The checkpoints are Hub-packaged: they bundle their own processor and
    modeling code, so the default backend is plain `transformers` with
    `trust_remote_code=True` and needs no extra install. Passing
    `use_vllm=True` serves the same checkpoint through an in-process vLLM
    engine instead (requires the VisionPsy vLLM plugin).

    Two preprocessing variants exist and are resolved from the checkpoint's own
    config: the base model resizes to the max side, the Flash variant keeps the
    native resolution with a min-side clamp.
    """

    INSTALL_REQ = False
    INTERLEAVE = True

    _MODEL_TYPES = ('visionpsynano', 'visionpsy')

    def __init__(self, model_path='qvac/VisionPsy-Nano-460M', use_vllm=False, **kwargs):
        super().__init__()
        self.use_vllm = use_vllm

        raw_cfg = self._read_raw_config(model_path)
        if raw_cfg is None:
            raise RuntimeError(f'cannot read config.json for {model_path}')
        if raw_cfg.get('model_type') not in self._MODEL_TYPES:
            raise ValueError(
                f'{model_path} is not a VisionPsy checkpoint '
                f'(config model_type: {raw_cfg.get("model_type")!r}).'
            )

        kwargs_default = {'max_new_tokens': 2048}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'VisionPsy kwargs: {self.kwargs}')

        if use_vllm:
            self._init_vllm(model_path, raw_cfg)
        else:
            self._init_transformers(model_path)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------
    def _init_transformers(self, model_path):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.float32,
        ).to('cuda').eval()
        if hasattr(self.vlm, 'apply_eager_profile'):
            self.vlm.apply_eager_profile()
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
        )
        torch.cuda.empty_cache()

    def _init_vllm(self, model_path, raw_cfg):
        # Build the vLLM engine FIRST: HF tokenizers and torch spawn worker
        # threads, and vLLM forks its engine-core subprocess -- forking a
        # threaded parent deadlocks the child. Only config reads happen before
        # the engine is up.
        self.llm = self._build_engine(
            model_path, raw_cfg.get('lm_max_position_embeddings', 8192),
        )

        self._ensure_reference_importable()
        from dataclasses import fields as dc_fields

        from data.processors import apply_model_preprocess, get_image_processor, get_tokenizer
        from models.config import VLMConfig

        valid = {fld.name for fld in dc_fields(VLMConfig)}
        self.cfg = VLMConfig(**{k: v for k, v in raw_cfg.items() if k in valid})
        self.tokenizer = get_tokenizer(
            self.cfg.lm_tokenizer,
            getattr(self.cfg, 'vlm_extra_tokens', None),
            getattr(self.cfg, 'lm_chat_template', None),
        )
        # Resolves the base/Flash resize policy from the checkpoint config.
        apply_model_preprocess(self.cfg)
        max_img = (getattr(self.cfg, 'inference_max_img_size', None)
                   or getattr(self.cfg, 'max_img_size', self.cfg.vit_img_size))
        self.image_processor = get_image_processor(
            max_img,
            self.cfg.vit_img_size,
            getattr(self.cfg, 'resize_to_max_side_len', False),
            getattr(self.cfg, 'resize_min_side_len', None),
        )

    @staticmethod
    def _build_engine(model_path, max_model_len):
        try:
            import visionpsy_vllm
        except ImportError:
            raise ImportError(_VLLM_PLUGIN_INSTALL_MSG)
        visionpsy_vllm.register()
        from vllm import LLM

        return LLM(
            model=model_path,
            dtype='float32',
            max_model_len=max_model_len,
            limit_mm_per_prompt={'image': 128},
            enforce_eager=True,
        )

    @staticmethod
    def _ensure_reference_importable():
        """Expose the reference preprocessing bundled with the vLLM plugin."""
        try:
            import visionpsy_vllm
        except ImportError:
            raise ImportError(_VLLM_PLUGIN_INSTALL_MSG)
        ref = os.path.join(os.path.dirname(visionpsy_vllm.__file__), 'reference')
        if os.path.isdir(ref) and ref not in sys.path:
            sys.path.append(ref)

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
            'InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'POPE',
            'BLINK',
        ):
            prompt, images = self._build_prompt_default(message, add_brief=True)
        elif dataset == 'HallusionBench':
            prompt, images = self._build_prompt_default(message, add_yes_or_no=True)
        elif dataset in (
            'MMStar', 'SEEDBench_IMG', 'AI2D_TEST',
            'ScienceQA_VAL', 'ScienceQA_TEST', 'RealWorldQA',
        ):
            prompt, images = self._build_prompt_puremcq(message)
        else:
            prompt, images = self._build_prompt_default(message)

        if self.use_vllm:
            return self._run_generation_vllm(prompt, images)
        return self._run_generation(prompt, images)

    def _run_generation(self, prompt, pil_images):
        """Greedy generation through the Hub-packaged model.

        The bundled processor reproduces the reference preprocessing (tiling,
        image-token string, chat template), so the adapter only supplies the
        per-dataset prompt text and the raw images.
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

        The image is tiled client-side and the prompt carries the global/tile
        position tokens with ONE image placeholder per tile; the plugin's
        multimodal processor expands each placeholder to mp_image_token_length
        image tokens inside vLLM.
        """
        from data.processors import get_image_string
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
    # Per-dataset prompt builders
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

    def _build_prompt_default(self, message, add_brief=False, add_yes_or_no=False,
                              add_direct=False):
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
