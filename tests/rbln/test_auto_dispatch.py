"""US-002 — auto.py architecture dispatch.

``auto_select_wrapper`` maps a model path to ``(RBLNWrapperCls,
default_kwargs)`` using ``config.json``'s ``architectures`` field, with a
``"cosmos"`` substring override. These tests build throwaway local
directories with a synthetic ``config.json`` so no HF download or NPU is
involved, and assert:

* each ``_ARCH_TABLE`` token resolves to the documented wrapper class,
* ordering hazards are respected (paligemma2 before paligemma, the qwen
  variants, blip2),
* the ``"cosmos"`` path override wins without reading config,
* compile-time ``rbln_config`` defaults are seeded into the kwargs,
* an unknown architecture raises ``ValueError``.
"""

from __future__ import annotations
import json

import pytest

from vlmeval.vlm.rbln import (RBLNBlip2, RBLNCosmosReason1, RBLNGemma3, RBLNIdefics3, RBLNLlava,
                              RBLNLlavaNext, RBLNPaliGemma, RBLNPaliGemma2, RBLNPixtral,
                              RBLNQwen2VL, RBLNQwen3VL)
from vlmeval.vlm.rbln.auto import auto_select_wrapper


def _make_model_dir(tmp_path, name: str, architectures):
    d = tmp_path / name
    d.mkdir()
    (d / 'config.json').write_text(
        json.dumps({'architectures': architectures}), encoding='utf-8'
    )
    return str(d)


# (subdir name, architectures list, expected wrapper class)
_CASES = [
    ('qwen3', ['Qwen3VLForConditionalGeneration'], RBLNQwen3VL),
    ('qwen25', ['Qwen2_5_VLForConditionalGeneration'], RBLNQwen2VL),
    ('qwen2', ['Qwen2VLForConditionalGeneration'], RBLNQwen2VL),
    ('llavanext', ['LlavaNextForConditionalGeneration'], RBLNLlavaNext),
    ('llava', ['LlavaForConditionalGeneration'], RBLNLlava),
    ('idefics3', ['Idefics3ForConditionalGeneration'], RBLNIdefics3),
    ('gemma3', ['Gemma3ForConditionalGeneration'], RBLNGemma3),
    # NOTE: pixtral is NOT here — it ships as LlavaForConditionalGeneration
    # (model_type llava), indistinguishable from LLaVA-1.5 by architectures,
    # so it is routed by a path marker. See test_pixtral_* below.
    ('paligemma2', ['PaliGemma2ForConditionalGeneration'], RBLNPaliGemma2),
    ('paligemma', ['PaliGemmaForConditionalGeneration'], RBLNPaliGemma),
    ('blip2', ['Blip2ForConditionalGeneration'], RBLNBlip2),
]


@pytest.mark.parametrize('name,archs,expected', _CASES,
                         ids=[c[0] for c in _CASES])
def test_arch_token_resolves_to_wrapper(tmp_path, name, archs, expected):
    path = _make_model_dir(tmp_path, name, archs)
    cls, defaults = auto_select_wrapper(path)
    assert cls is expected
    # Compile defaults are seeded into the returned kwargs.
    assert 'rbln_config' in defaults
    assert isinstance(defaults['rbln_config'], dict)


def test_paligemma2_wins_over_paligemma(tmp_path):
    """paligemma2 token precedes paligemma in _ARCH_TABLE — the more
    specific token must win even though 'paligemma2' contains 'paligemma'.
    """
    path = _make_model_dir(tmp_path, 'pg2', ['PaliGemma2ForConditionalGeneration'])
    cls, _ = auto_select_wrapper(path)
    assert cls is RBLNPaliGemma2
    assert cls is not RBLNPaliGemma


def test_qwen3_does_not_collapse_to_qwen2(tmp_path):
    path = _make_model_dir(tmp_path, 'q3', ['Qwen3VLForConditionalGeneration'])
    cls, _ = auto_select_wrapper(path)
    assert cls is RBLNQwen3VL


def test_cosmos_path_override_skips_config(tmp_path):
    """Anything with 'cosmos' in the path resolves to RBLNCosmosReason1
    even though it shares the Qwen2.5-VL architecture — and even without a
    config.json present (the override returns before reading it).
    """
    d = tmp_path / 'Cosmos-Reason1-7B'
    d.mkdir()  # intentionally no config.json
    cls, defaults = auto_select_wrapper(str(d))
    assert cls is RBLNCosmosReason1
    # Cosmos seeds its own visual.max_seq_lens budget (8192, vs Qwen 6400).
    assert defaults['rbln_config']['visual']['max_seq_lens'] == 8192


def test_pixtral_routed_by_path_despite_llava_arch(tmp_path):
    """Real Pixtral ships as architectures=['LlavaForConditionalGeneration']
    (model_type llava) — identical to LLaVA-1.5. It must still resolve to
    RBLNPixtral via the 'pixtral' path marker, with the Pixtral compile
    defaults (max_seq_len + kvcache_partition_len, which the LLaVA defaults
    lack and whose absence caused the 1M-context eager-attention failure).
    """
    path = _make_model_dir(tmp_path, 'pixtral-12b', ['LlavaForConditionalGeneration'])
    cls, defaults = auto_select_wrapper(path)
    assert cls is RBLNPixtral
    lm = defaults['rbln_config']['language_model']
    assert lm['max_seq_len'] == 131072
    assert lm['kvcache_partition_len'] == 16384


def test_llava_not_misrouted_to_pixtral(tmp_path):
    """Same architectures as Pixtral, but no 'pixtral' path marker -> stays
    RBLNLlava (guards against the path override over-matching)."""
    path = _make_model_dir(tmp_path, 'llava-1.5-7b-hf', ['LlavaForConditionalGeneration'])
    cls, _ = auto_select_wrapper(path)
    assert cls is RBLNLlava


def test_qwen_seeds_pixel_kwargs(tmp_path):
    path = _make_model_dir(tmp_path, 'q2', ['Qwen2VLForConditionalGeneration'])
    _, defaults = auto_select_wrapper(path)
    # _WRAPPER_KWARG_DEFAULTS seeds min/max pixels for Qwen families.
    assert 'min_pixels' in defaults and 'max_pixels' in defaults


def test_unknown_architecture_raises(tmp_path):
    path = _make_model_dir(tmp_path, 'mystery', ['TotallyUnknownArch'])
    with pytest.raises(ValueError):
        auto_select_wrapper(path)


def test_missing_config_json_raises(tmp_path):
    d = tmp_path / 'nocfg'
    d.mkdir()
    with pytest.raises(ValueError):
        auto_select_wrapper(str(d))
