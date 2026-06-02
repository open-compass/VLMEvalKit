"""Auto-dispatch helpers for the RBLN backend.

These helpers let ``run.py --device rbln --model <X>`` accept either an HF
model id or a local compiled directory: the RBLN wrapper class and its
compile defaults are resolved here, with no static model registry.

The wrapper class is chosen from ``config.json``'s ``architectures`` field
(with a small ``"cosmos"`` substring override for Cosmos-Reason1, which
shares the Qwen2.5-VL architecture but ships different video defaults).
``rbln_export`` is chosen from the on-disk artifact state. ``rbln_config``
compile defaults mirror the values used by rbln-model-zoo's
``compile.py`` for each family — necessary because optimum-rbln requires
keys like ``visual.max_seq_lens`` or ``language_model.max_seq_len`` to be
present at compile time and otherwise raises ``ValueError``.
"""

from __future__ import annotations
import json
import os
from functools import partial

from huggingface_hub import hf_hub_download

# Per-family table: (architectures-substring token, wrapper class name,
# compile-time rbln_config defaults).
#
# The compile defaults mirror rbln-model-zoo's
# huggingface/transformers/image-text-to-text/<family>/compile.py so a
# first-time auto-compile reproduces the same artifact the model-zoo
# scripts produce. The ``create_runtimes: False`` marker that some of
# those scripts use is intentionally omitted — that flag is for
# compile-only flows and would prevent inference right after auto-compile.
#
# Order matters: more specific tokens first (qwen3 before qwen2,
# paligemma2 before paligemma, llavanext before llavafor, blip_2 before
# blip2).
_ARCH_TABLE: tuple[tuple[str, str, dict], ...] = (
    ('qwen3vl', 'RBLNQwen3VL', {
        'visual': {'max_seq_lens': 16384, 'tensor_parallel_size': 8},
        'tensor_parallel_size': 8,
        'kvcache_partition_len': 16384,
        'max_seq_len': 262144,
    }),
    ('qwen2_5_vl', 'RBLNQwen2VL', {
        'visual': {'max_seq_lens': 6400},
        'tensor_parallel_size': 8,
        'kvcache_partition_len': 16384,
        'max_seq_len': 114688,
    }),
    ('qwen2vl', 'RBLNQwen2VL', {
        'visual': {'max_seq_lens': 6400},
        'tensor_parallel_size': 8,
        'max_seq_len': 32768,
    }),
    ('llavanext', 'RBLNLlavaNext', {
        'language_model': {'tensor_parallel_size': 4, 'use_inputs_embeds': True},
    }),
    ('llavafor', 'RBLNLlava', {
        'vision_tower': {'output_hidden_states': True},
        'language_model': {'tensor_parallel_size': 4, 'use_inputs_embeds': True},
    }),
    ('idefics3', 'RBLNIdefics3', {
        'text_model': {
            'batch_size': 1,
            'max_seq_len': 131072,
            'tensor_parallel_size': 8,
            'use_inputs_embeds': True,
            'attn_impl': 'flash_attn',
            'kvcache_partition_len': 16384,
        },
    }),
    # Matches rbln-model-zoo/.../gemma3/gemma-3-4b/compile.py verbatim.
    # The 4b recipe is chosen as the default because it has the broadest
    # NPU footprint (tp=4 fits on every supported board). Larger Gemma3
    # variants use different tensor_parallel_size in rbln-model-zoo
    # (gemma-3-12b=8, gemma-3-27b=16); callers can override via
    # ``--rbln-kwargs '{"rbln_config": {...}}'``.
    # ``max_seq_len`` is intentionally absent — rbln-model-zoo lets
    # optimum-rbln fall back to the model's
    # ``config.json:max_position_embeddings``.
    ('gemma3', 'RBLNGemma3', {
        'language_model': {
            'tensor_parallel_size': 4,
            'kvcache_partition_len': 16384,
            'use_inputs_embeds': True,
        },
    }),
    ('paligemma2', 'RBLNPaliGemma2', {
        'language_model': {
            'batch_size': 1,
            'max_seq_len': 8192,
            'tensor_parallel_size': 4,
            'prefill_chunk_size': 8192,
        },
    }),
    ('paligemma', 'RBLNPaliGemma', {
        'language_model': {
            'batch_size': 1,
            'max_seq_len': 8192,
            'tensor_parallel_size': 4,
            'prefill_chunk_size': 8192,
        },
    }),
    ('blip2', 'RBLNBlip2', {
        'language_model': {
            'batch_size': 1,
            'max_seq_len': 2048,
            'tensor_parallel_size': 1,
            'use_inputs_embeds': True,
        },
    }),
)

# Cosmos-Reason1 specific compile defaults — shares Qwen2.5-VL architecture
# but uses a different visual.max_seq_lens budget per rbln-model-zoo
# cosmos-reason1/compile.py.
_COSMOS_RBLN_CONFIG: dict = {
    'visual': {'max_seq_lens': 8192},
    'tensor_parallel_size': 8,
    'kvcache_partition_len': 16384,
    'max_seq_len': 114688,
}

# Pixtral compile defaults. Pixtral ships with
# architectures=["LlavaForConditionalGeneration"] and model_type="llava" —
# byte-for-byte the same as LLaVA-1.5 — so it CANNOT be told apart from
# LLaVA by the architectures string. It is therefore routed by a "pixtral"
# model-path marker (mirroring the cosmos override) instead of _ARCH_TABLE,
# and its defaults live here rather than in that table. ``kvcache_partition_len``
# makes optimum-rbln switch to flash attention automatically, which the
# 1M-token Pixtral context length requires (eager attention caps at 32768).
_PIXTRAL_RBLN_CONFIG: dict = {
    'vision_tower': {'batch_size': 1, 'output_hidden_states': True},
    'language_model': {
        'tensor_parallel_size': 8,
        'use_inputs_embeds': True,
        'batch_size': 1,
        'max_seq_len': 131072,
        'kvcache_partition_len': 16384,
    },
}

# Wrapper-specific __init__ kwarg defaults (separate from rbln_config),
# e.g. Qwen's min/max pixels.
_WRAPPER_KWARG_DEFAULTS: dict[str, dict] = {
    'RBLNQwen2VL': {
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1280 * 28 * 28,
    },
    'RBLNQwen3VL': {
        'min_pixels': 256 * 28 * 28,
        'max_pixels': 1280 * 28 * 28,
    },
}


def _read_architectures_field(config_json_path: str) -> str:
    with open(config_json_path, 'r', encoding='utf-8') as f:
        arch = json.load(f).get('architectures', '')
    if isinstance(arch, list):
        return ' '.join(str(a) for a in arch).lower()
    return str(arch).lower()


def _fetch_architectures(model_path: str) -> str:
    """Return the architectures string for a local dir or HF model id."""
    if os.path.isdir(model_path):
        cfg = os.path.join(model_path, 'config.json')
        if not os.path.isfile(cfg):
            raise ValueError(
                f"No config.json found in directory {model_path!r}; "
                "cannot auto-detect RBLN wrapper class."
            )
        return _read_architectures_field(cfg)

    # Treat as HF id; download only config.json so we don't pull the full snapshot.
    cfg = hf_hub_download(repo_id=model_path, filename='config.json')
    return _read_architectures_field(cfg)


def auto_select_wrapper(model_path: str) -> tuple[type, dict]:
    """Resolve a model path to ``(RBLNWrapperCls, default_init_kwargs)``.

    ``model_path`` may be either a local compiled directory or an HF
    model id. The choice is driven by ``config.json``'s ``architectures``
    except for one substring override: anything containing ``"cosmos"``
    (case-insensitive) picks ``RBLNCosmosReason1`` even though it shares
    the Qwen2.5-VL architecture.

    The returned dict contains both wrapper ``__init__`` kwargs (e.g.
    ``min_pixels``) and a ``rbln_config`` entry seeded with the
    rbln-model-zoo compile defaults for that family. Caller (typically
    ``register_rbln_auto``) can overlay user-supplied overrides on top.
    """
    # Lazy import: avoid pulling every wrapper at module load time, and
    # avoid a circular import when this module is loaded from rbln/__init__.
    from . import (RBLNBlip2, RBLNCosmosReason1, RBLNGemma3, RBLNIdefics3, RBLNLlava,
                   RBLNLlavaNext, RBLNPaliGemma, RBLNPaliGemma2, RBLNPixtral, RBLNQwen2VL,
                   RBLNQwen3VL)

    name_to_cls: dict[str, type] = {
        'RBLNQwen2VL': RBLNQwen2VL,
        'RBLNQwen3VL': RBLNQwen3VL,
        'RBLNLlavaNext': RBLNLlavaNext,
        'RBLNLlava': RBLNLlava,
        'RBLNIdefics3': RBLNIdefics3,
        'RBLNGemma3': RBLNGemma3,
        'RBLNPixtral': RBLNPixtral,
        'RBLNPaliGemma2': RBLNPaliGemma2,
        'RBLNPaliGemma': RBLNPaliGemma,
        'RBLNBlip2': RBLNBlip2,
        'RBLNCosmosReason1': RBLNCosmosReason1,
    }

    if 'cosmos' in model_path.lower():
        defaults = {
            **_WRAPPER_KWARG_DEFAULTS.get('RBLNQwen2VL', {}),
            'rbln_config': dict(_COSMOS_RBLN_CONFIG),
        }
        return RBLNCosmosReason1, defaults

    # Pixtral shares LLaVA-1.5's architectures string
    # ("LlavaForConditionalGeneration", model_type="llava"), so the
    # _ARCH_TABLE scan below would match "llavafor" -> RBLNLlava and compile
    # with the wrong (LLaVA) defaults — which omit max_seq_len /
    # kvcache_partition_len and blow past eager attention's 32768 cap on
    # Pixtral's 1M context. Route by the "pixtral" path marker first.
    if 'pixtral' in model_path.lower():
        defaults = {
            **_WRAPPER_KWARG_DEFAULTS.get('RBLNPixtral', {}),
            'rbln_config': dict(_PIXTRAL_RBLN_CONFIG),
        }
        return RBLNPixtral, defaults

    archs = _fetch_architectures(model_path)
    for token, cls_name, compile_defaults in _ARCH_TABLE:
        if token in archs:
            cls = name_to_cls[cls_name]
            defaults = {
                **_WRAPPER_KWARG_DEFAULTS.get(cls_name, {}),
                'rbln_config': dict(compile_defaults),
            }
            return cls, defaults

    raise ValueError(
        f"No RBLN wrapper matches architectures={archs!r} for "
        f"model_path={model_path!r}. Add the architecture token to "
        "_ARCH_TABLE in vlmeval/vlm/rbln/auto.py."
    )


def register_rbln_auto(
    model_name: str,
    supported_vlm: dict,
    extra_kwargs: dict | None = None,
    register_as: str | None = None,
) -> str:
    """Add an entry to ``supported_vlm`` for ``model_name``.

    ``register_as`` controls the dict key (defaulting to ``model_name``);
    callers pass ``os.path.basename(model_name)`` when the model name
    contains ``/`` so VLMEvalKit's output paths stay flat.

    The wrapper itself (``RBLNVLMBase``) decides whether to load an
    already-compiled artifact at ``./<basename(model_name)>/`` or to
    compile fresh from ``model_name`` and save the result there — see
    ``_resolve_model_path`` and ``_maybe_save_compiled_artifact`` in
    ``base.py``. This function therefore only routes the wrapper class
    and seeded ``rbln_config`` compile defaults into the partial.

    Idempotent — does nothing when the resolved key is already present,
    so pre-registered ``*-RBLN`` names always win.

    Returns the dict key used.
    """
    key = register_as or model_name
    if key in supported_vlm:
        return key

    cls, defaults = auto_select_wrapper(model_name)
    kwargs: dict = {**defaults, 'model_path': model_name}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    supported_vlm[key] = partial(cls, **kwargs)
    return key
