from __future__ import annotations
import json
import logging
import os
import sys
import warnings
from abc import abstractmethod

from huggingface_hub import snapshot_download

from vlmeval.smp import get_cache_path
from ..base import BaseModel


class RBLNVLMBase(BaseModel):
    """Shared infrastructure for VLMEvalKit wrappers backed by optimum-rbln.

    Subclasses select the optimum-rbln model class for a particular VLM
    family and implement ``_load_rbln_model_and_processor`` +
    ``generate_inner``. Prompt construction is NOT handled here — each
    family wrapper composes whichever ``*PromptMixin`` its upstream
    counterpart uses (or none, matching upstream models that inherit
    ``BaseModel`` directly).
    """

    is_api = False
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    RBLN_MODEL_CLASS = None

    # Decode-time knobs for the default ``generate_inner`` impls. Override
    # in subclasses that need different post-processing (e.g. BLIP-2 wants
    # the full sequence + a trailing ``.strip()``).
    _DECODE_TRIM: bool = True
    _DECODE_STRIP: bool = False

    def __init__(
        self,
        model_path: str,
        rbln_config: dict | None = None,
        rbln_runtime_config: dict | None = None,
        rbln_export: bool | None = None,
        max_new_tokens: int = 2048,
        do_sample: bool = False,
        temperature: float | None = 0.0,
        top_p: float | None = 1.0,
        top_k: int | None = 1,
        repetition_penalty: float = 1.0,
        num_beams: int | None = None,
        system_prompt: str | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        # BaseModel.__init__ takes no arguments and VLMEvalKit's dispatch
        # injects framework-wide kwargs (use_vllm, retry, timeout, ...)
        # into every wrapper partial. Absorb them here without forwarding.
        super().__init__()

        if model_path is None:
            raise ValueError("model_path is required")

        # Keep the caller's original model_path: ``_save_dir`` derives the
        # cache location from its basename, mirroring rbln-model-zoo's
        # ``model.save_pretrained(os.path.basename(model_id))`` convention.
        self._input_model_path = model_path
        self.model_path = self._resolve_model_path(model_path)
        self.rbln_config = rbln_config or {}
        self.rbln_runtime_config = rbln_runtime_config or {}
        self.rbln_export = rbln_export
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.max_new_tokens = max_new_tokens
        # Drop None-valued sampling params so HF generate falls back to
        # its own defaults (e.g. upstream LLaVA wrappers pass ``top_p=None``).
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_beams=num_beams,
        )
        self.generate_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        self.model, self.processor = self._load_rbln_model_and_processor()
        self._maybe_save_compiled_artifact()
        self._check_compiled_max_seq_len()

    @staticmethod
    def _is_compiled_dir(path: str) -> bool:
        if not os.path.isdir(path):
            return False
        try:
            return any(n.endswith('.rbln') for n in os.listdir(path))
        except OSError:
            return False

    def _save_dir(self) -> str:
        """``<cwd>/<basename(model_path)>`` — rbln-model-zoo convention."""
        basename = os.path.basename(self._input_model_path.rstrip('/'))
        return os.path.abspath(basename) if basename else ''

    def _resolve_model_path(self, model_path: str) -> str:
        """Pick the directory optimum-rbln should load from.

        Order:
          1. ``./<basename(model_path)>/`` if it already contains a
             compiled RBLN artifact (rbln-model-zoo convention).
          2. ``model_path`` itself if it is a compiled-artifact directory.
          3. ``model_path`` if it is an existing local directory.
          4. Otherwise treat ``model_path`` as an HF id and resolve it to
             the HF cache (downloading the snapshot if necessary).
        """
        basename = os.path.basename(model_path.rstrip('/'))
        if basename:
            cached = os.path.abspath(basename)
            if self._is_compiled_dir(cached):
                return cached
        if self._is_compiled_dir(model_path):
            return model_path
        if os.path.exists(model_path):
            return model_path
        cache_path = get_cache_path(model_path, repo_type='models')
        if cache_path is None:
            snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(model_path, repo_type='models')
        return cache_path

    def _resolve_rbln_class(self):
        """Pick the optimum-rbln model class.

        Resolution order:
          1. ``self.RBLN_MODEL_CLASS`` if a subclass sets it.
          2. Subclass override of ``_pick_rbln_class()`` (for config-driven dispatch).
          3. ``RBLNAutoModelForVision2Seq`` as the default.
        """
        if self.RBLN_MODEL_CLASS is not None:
            return self.RBLN_MODEL_CLASS
        picked = self._pick_rbln_class()
        if picked is not None:
            return picked
        from optimum.rbln import RBLNAutoModelForVision2Seq
        return RBLNAutoModelForVision2Seq

    def _pick_rbln_class(self):
        return None

    def _read_architectures(self) -> str:
        cfg_path = os.path.join(self.model_path, 'config.json')
        if not os.path.isfile(cfg_path):
            return ''
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                arch = json.load(f).get('architectures', '')
        except (OSError, json.JSONDecodeError):
            return ''
        if isinstance(arch, list):
            return ' '.join(str(a) for a in arch).lower()
        return str(arch).lower()

    def _from_pretrained(self, rbln_cls=None):
        """Load an optimum-rbln model class with the configured rbln_config.

        ``rbln_export`` semantics:
          * ``None``  -> let optimum-rbln auto-detect (load if compiled
            artifact exists in ``model_path``, else compile).
          * ``True``  -> force re-compile. Only ``rbln_config`` is passed.
          * ``False`` -> require pre-compiled artifact; ``rbln_config`` and
            ``rbln_runtime_config`` are merged for runtime hints.

        Mirrors the rbln-model-zoo convention where compile.py supplies
        e.g. ``{"visual": {"max_seq_lens": 6400}, "tensor_parallel_size":
        8, "max_seq_len": 114_688}`` while inference.py later overlays
        device placement like ``{"visual": {"device": 0}, "device":
        [0,1,2,3,4,5,6,7]}``. ``_merge_rbln_config`` does a one-level
        nested merge so per-submodule overrides do not clobber the
        compile-time keys.
        """
        if rbln_cls is None:
            rbln_cls = self._resolve_rbln_class()

        # When loading from an already-compiled directory, the compile-time
        # rbln_config (tensor_parallel_size, max_seq_len, ...) is baked into
        # the saved rbln_config.json. Passing it again triggers
        # ``ValueError: Cannot set the following arguments: [...] Since the
        # value is already set to None``. Only runtime overrides
        # (device placement) remain applicable.
        is_compiled = self._is_compiled_dir(self.model_path)

        kwargs = {}
        if self.rbln_export is not None:
            kwargs['export'] = self.rbln_export
        elif is_compiled:
            kwargs['export'] = False

        if is_compiled:
            if self.rbln_runtime_config:
                kwargs['rbln_config'] = dict(self.rbln_runtime_config)
        else:
            merged = self._merge_rbln_config()
            if merged:
                kwargs['rbln_config'] = merged

        if self.verbose:
            logging.info(
                f"Loading {rbln_cls.__name__} from {self.model_path} "
                f"(export={kwargs.get('export')}, "
                f"rbln_config={kwargs.get('rbln_config')})"
            )
        return rbln_cls.from_pretrained(self.model_path, **kwargs)

    def _merge_rbln_config(self) -> dict:
        if self.rbln_export is True or not self.rbln_runtime_config:
            return dict(self.rbln_config)
        merged = dict(self.rbln_config)
        for k, v in self.rbln_runtime_config.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                sub = dict(merged[k])
                sub.update(v)
                merged[k] = sub
            else:
                merged[k] = v
        return merged

    @staticmethod
    def _load_image(path_or_url):
        from transformers.image_utils import load_image
        return load_image(path_or_url)

    @staticmethod
    def _ensure_media_url(value: str, kind: str = 'image') -> str:
        prefixes = ('http://', 'https://', 'file://', f'data:{kind};')
        if value.startswith(prefixes):
            return value
        if os.path.exists(value):
            return 'file://' + value
        raise ValueError(f'Invalid {kind} path/URL: {value}')

    def _decode_generated(self, inputs, generated_ids, *, trim: bool = True,
                          skip_special_tokens: bool = True,
                          clean_up_tokenization_spaces: bool = False):
        if trim and hasattr(inputs, 'input_ids'):
            input_ids = inputs.input_ids
            trimmed = [
                out[len(inp):] for inp, out in zip(input_ids, generated_ids)
            ]
        else:
            trimmed = generated_ids

        decoded = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return decoded[0] if decoded else ''

    def _debug_log(self, obj, color: str = 'red'):
        if not self.verbose:
            return
        logging.debug('[%s] %s', self.__class__.__name__, obj)
        if sys.stdout.isatty():
            ansi = {'red': 31, 'green': 32, 'yellow': 33, 'blue': 34}.get(color, 0)
            print(f'\033[{ansi}m{obj}\033[0m')

    def _warn_once(self, key: str, message: str) -> None:
        flag = f'_warned_{key}'
        if getattr(self, flag, False):
            return
        warnings.warn(message)
        setattr(self, flag, True)

    def _maybe_save_compiled_artifact(self) -> None:
        """Persist the freshly compiled RBLN artifact + processor assets to
        ``<cwd>/<basename(model_path)>/`` (rbln-model-zoo convention) so
        subsequent runs can skip recompilation.

        Skipped when the target already contains an ``*.rbln`` artifact
        (it was either loaded from there or saved by a previous run) or
        when save fails for any reason — failure is logged and the
        in-memory model continues to serve this run.
        """
        path = self._save_dir()
        if not path or self._is_compiled_dir(path):
            return
        try:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
            if hasattr(self.processor, 'save_pretrained'):
                self.processor.save_pretrained(path)
            logging.info(f'Saved compiled RBLN artifact to {path}')
        except Exception as e:
            logging.warning(f'Failed to save compiled RBLN artifact to {path}: {e}')

    def _check_compiled_max_seq_len(self) -> None:
        """Warn when the compiled RBLN artifact's ``max_seq_len`` is
        shorter than what upstream HF transformers would consider the
        model's effective context length.

        Why this matters for evaluation parity: upstream/GPU runs use
        the tokenizer's ``model_max_length`` (or ``config.json``'s
        ``max_position_embeddings``) as the upper bound on input + output
        tokens. RBLN bakes ``max_seq_len`` into the compiled artifact at
        compile time, so a shorter compiled value truncates long inputs
        silently on RBLN while GPU continues to process them — producing
        different predictions for the same item. Common trigger: MMMU /
        DocVQA samples with multiple high-resolution images.

        The check is best-effort: we recursively scan the saved
        ``rbln_config.json`` for any ``max_seq_len`` field (the exact
        nesting path varies per family — top-level for Qwen-VL,
        ``language_model.max_seq_len`` for Gemma3 / LLaVA-Next,
        ``text_model.max_seq_len`` for Idefics3, etc.), pick the smallest,
        and compare with what the loaded HF processor/config advertises.
        """
        compiled = self._extract_compiled_max_seq_len()
        if compiled is None:
            return
        expected = self._extract_upstream_max_seq_len()
        if expected is None or expected >= 10**9:
            # Tokenizers sometimes set model_max_length to a sentinel
            # like ``int(1e30)`` when the model has no explicit cap.
            return
        if compiled >= expected:
            return
        warnings.warn(
            f'{type(self).__name__}: compiled RBLN max_seq_len={compiled} is '
            f'shorter than the upstream HF context length={expected}. Long '
            'inputs (multi-image MMMU/DocVQA, long-prompt VQA) may be '
            'truncated on RBLN but not on GPU — evaluation parity will '
            'break for those items.',
            stacklevel=2,
        )

    def _extract_compiled_max_seq_len(self) -> int | None:
        """Walk the saved ``rbln_config.json`` and return the smallest
        ``max_seq_len`` value, or ``None`` if no such field exists.
        """
        cfg_path = os.path.join(self.model_path, 'rbln_config.json')
        if not os.path.isfile(cfg_path):
            return None
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        found: list[int] = []

        def _walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k == 'max_seq_len' and isinstance(v, int):
                        found.append(v)
                    else:
                        _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        _walk(cfg)
        return min(found) if found else None

    # Tokenizers often advertise ``model_max_length`` as a sentinel like
    # ``int(1e30)`` when no real cap is set. Above this threshold we fall
    # through to the model's ``config.json`` instead.
    _MAX_SEQ_LEN_SENTINEL: int = 10**9

    def _extract_upstream_max_seq_len(self) -> int | None:
        """Best-effort guess at what HF transformers would consider the
        model's max context length, using (in order):

        1. ``processor.tokenizer.model_max_length`` if it is a sane int
           (i.e. below :pyattr:`_MAX_SEQ_LEN_SENTINEL`).
        2. ``config.json``'s ``max_position_embeddings``.
        3. ``config.json``'s ``text_config.max_position_embeddings`` /
           ``language_model.max_position_embeddings`` for nested VLM
           configs.

        Returns ``None`` if nothing usable is found.
        """
        tokenizer = getattr(self.processor, 'tokenizer', None)
        mml = getattr(tokenizer, 'model_max_length', None)
        if isinstance(mml, int) and mml < self._MAX_SEQ_LEN_SENTINEL:
            return mml

        cfg_path = os.path.join(self.model_path, 'config.json')
        if not os.path.isfile(cfg_path):
            return None
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        for key in ('max_position_embeddings',):
            v = cfg.get(key)
            if isinstance(v, int):
                return v
        for parent in ('text_config', 'language_model_config', 'language_model'):
            sub = cfg.get(parent)
            if isinstance(sub, dict):
                v = sub.get('max_position_embeddings')
                if isinstance(v, int):
                    return v
        return None

    @abstractmethod
    def _load_rbln_model_and_processor(self):
        raise NotImplementedError

    def _finalize_response(self, inputs, generated_ids) -> str:
        out = self._decode_generated(inputs, generated_ids, trim=self._DECODE_TRIM)
        return out.strip() if self._DECODE_STRIP else out

    def generate_inner(self, message, dataset=None):
        """Default for raw-prompt single-image VLMs (PaliGemma, PaliGemma2,
        BLIP-2). Subclasses can override for special pre-processing.
        """
        prompt, image = self.message_to_promptimg(message, dataset=dataset)
        if image is None:
            raise ValueError(
                f"{type(self).__name__} requires an image input."
            )
        inputs = self.processor(
            text=prompt,
            images=self._load_image(image),
            return_tensors='pt',
        )
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        return self._finalize_response(inputs, generated_ids)


class RBLNChatVLMBase(RBLNVLMBase):
    """Base for VLMs that use ``processor.apply_chat_template`` for input
    construction (Qwen2-VL family, Qwen3-VL, LLaVA, LLaVA-Next, Idefics3,
    Gemma3, Pixtral, Cosmos-Reason1).

    Provides a default chat-template ``generate_inner`` and inherits
    loading/decoding helpers from ``RBLNVLMBase``. Does **not** bake in
    any prompt mixin: each family wrapper composes whichever
    ``*PromptMixin`` matches its upstream counterpart so RBLN and the
    upstream CUDA wrapper send the model byte-identical prompts.

    Upstream mapping for the wrappers in this package::

        upstream                       RBLN
        Qwen2VLChat (Qwen2VLPromptMixin)  -> RBLNQwen2VL (Qwen2VLPromptMixin, RBLNChatVLMBase)
        Qwen3VLChat (Qwen3VLPromptMixin)  -> RBLNQwen3VL (Qwen3VLPromptMixin, RBLNQwen2VL)
        LLaVA / LLaVA_Next (no mixin)     -> RBLNLlava / RBLNLlavaNext (RBLNChatVLMBase)
        IDEFICS3        (no mixin)        -> RBLNIdefics3 (RBLNChatVLMBase)
        Gemma3          (no mixin)        -> RBLNGemma3 (RBLNChatVLMBase)
        Pixtral         (no mixin)        -> RBLNPixtral (RBLNChatVLMBase)
    """

    def generate_inner(self, message, dataset=None):
        """Default for chat-template VLMs with interleaved image / text
        content. Qwen2-VL family overrides this to thread per-image
        ``min_pixels`` / ``max_pixels`` and ``video_kwargs`` through.
        """
        images, content = [], []
        for s in message:
            if s['type'] == 'image':
                images.append(self._load_image(s['value']))
                content.append({'type': 'image'})
            elif s['type'] == 'text':
                content.append({'type': 'text', 'text': s['value']})
            else:
                raise ValueError(
                    f"{type(self).__name__} does not support type: {s['type']}"
                )

        chat = []
        if self.system_prompt is not None:
            chat.append({'role': 'system', 'content': self.system_prompt})
        chat.append({'role': 'user', 'content': content})

        prompt = self.processor.apply_chat_template(
            chat, add_generation_prompt=True, tokenize=False,
        )
        inputs = self.processor(
            text=prompt,
            images=images if images else None,
            return_tensors='pt',
        )
        generated_ids = self.model.generate(**inputs, **self.generate_kwargs)
        return self._finalize_response(inputs, generated_ids)
