"""US-005 — prompt parity, organised into four buckets.

The core invariant (AGENTS.md) is that RBLN wrappers feed the model
byte-identical prompts to their upstream CUDA counterparts. How that's
verified differs by family, so the tests are bucketed:

* REUSE      — Qwen2/Qwen3 reuse the upstream *PromptMixin verbatim;
               assert method identity (catches an accidental override).
* REIMPLEMENT — LLaVA / LLaVA-Next re-implement build_prompt; assert the
               exact message it produces for a known MCQ line.
* EMBEDDED   — Idefics3 has no model-level build_prompt; its prompt logic
               lives in _format_for_dataset. Assert that directly.
* DELEGATE   — Cosmos/Pixtral/Gemma3/PaliGemma(2) delegate to the
               dataset's own build_prompt (use_custom_prompt is False).

Plus a BLIP-2 decode-trim assertion (its distinguishing post-processing).

None of these load a model or touch the NPU.
"""

from __future__ import annotations

import pytest

from vlmeval.vlm.base import BaseModel
from vlmeval.vlm.qwen2_vl.prompt import Qwen2VLPromptMixin
from vlmeval.vlm.qwen3_vl.prompt import Qwen3VLPromptMixin
from vlmeval.vlm.rbln import (RBLNBlip2, RBLNCosmosReason1, RBLNGemma3, RBLNIdefics3, RBLNLlava,
                              RBLNLlavaNext, RBLNPaliGemma, RBLNPaliGemma2, RBLNPixtral,
                              RBLNQwen2VL, RBLNQwen3VL)


def test_qwen2_reuses_qwen2_mixin():  # Bucket 1: REUSE — method identity vs upstream mixin
    assert RBLNQwen2VL.build_prompt is Qwen2VLPromptMixin.build_prompt
    assert RBLNQwen2VL.use_custom_prompt is Qwen2VLPromptMixin.use_custom_prompt


def test_qwen3_reuses_qwen3_mixin_not_qwen2():
    assert RBLNQwen3VL.build_prompt is Qwen3VLPromptMixin.build_prompt
    assert RBLNQwen3VL.use_custom_prompt is Qwen3VLPromptMixin.use_custom_prompt
    # Qwen3 must NOT collapse to the Qwen2 mixin.
    assert RBLNQwen3VL.build_prompt is not Qwen2VLPromptMixin.build_prompt


# ----------------------------------------------------------------------
# Bucket 2: REIMPLEMENT — golden message for a known MCQ line
# ----------------------------------------------------------------------

class _PromptStub:
    """Minimal stand-in for build_prompt's `self` dependencies."""

    def use_custom_prompt(self, dataset):
        return True

    def dump_image(self, line, dataset):
        return '/tmp/img.png'


_MCQ_LINE = {'question': 'What animal?', 'A': 'cat', 'B': 'dog'}

_EXPECTED_MCQ_TEXT = (
    'What animal?\nA. cat\nB. dog\n'
    "Answer with the option's letter from the given choices directly."
)


@pytest.mark.parametrize('cls', [RBLNLlava, RBLNLlavaNext])
def test_llava_family_build_prompt_golden(cls):
    msg = cls.build_prompt(_PromptStub(), _MCQ_LINE, 'MMBench_DEV_EN')
    assert msg == [
        {'type': 'image', 'value': '/tmp/img.png'},
        {'type': 'text', 'value': _EXPECTED_MCQ_TEXT},
    ]


@pytest.mark.parametrize('cls', [RBLNLlava, RBLNLlavaNext])
def test_llava_family_use_custom_prompt_mcq(cls):
    # MCQ dataset -> custom prompt; non-MCQ -> dataset's own.
    assert cls.use_custom_prompt(None, 'MMBench_DEV_EN') is True


# ----------------------------------------------------------------------
# Bucket 3: EMBEDDED — Idefics3 prompt logic lives in _format_for_dataset
# ----------------------------------------------------------------------

def test_idefics3_has_no_model_level_build_prompt():
    # Inherits the abstract BaseModel.build_prompt (raises NotImplementedError).
    assert RBLNIdefics3.build_prompt is BaseModel.build_prompt


def test_idefics3_puremcq_format(monkeypatch):
    monkeypatch.setattr(
        'vlmeval.vlm.rbln.idefics3.load_image', lambda v: f'IMG:{v}'
    )
    obj = RBLNIdefics3.__new__(RBLNIdefics3)
    message = [
        {'type': 'image', 'value': 'x'},
        {'type': 'text', 'value': 'Question: foo\nOptions:\nA. a'},
    ]
    # MMStar -> _build_prompt_puremcq: '\nOptions:' becomes '\nChoices:'.
    prompt, images = obj._format_for_dataset(message, 'MMStar')
    assert prompt == (
        'User:<image>Question: foo\nChoices:\nA. a'
        '<end_of_utterance>\nAssistant: Answer:'
    )
    assert images == ['IMG:x']


def test_idefics3_default_format(monkeypatch):
    monkeypatch.setattr(
        'vlmeval.vlm.rbln.idefics3.load_image', lambda v: f'IMG:{v}'
    )
    obj = RBLNIdefics3.__new__(RBLNIdefics3)
    message = [
        {'type': 'image', 'value': 'y'},
        {'type': 'text', 'value': 'plain question'},
    ]
    # Unknown dataset -> _build_prompt_default (no add_brief/yes_no).
    prompt, images = obj._format_for_dataset(message, 'SomeUnknownDataset')
    assert prompt == 'User:<image>plain question<end_of_utterance>\nAssistant:'
    assert images == ['IMG:y']


# ----------------------------------------------------------------------
# Bucket 4: DELEGATE — use_custom_prompt is False (dataset builds the prompt)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('cls', [RBLNPixtral, RBLNGemma3, RBLNPaliGemma, RBLNPaliGemma2])
def test_delegate_inherits_basemodel_use_custom_prompt(cls):
    # No override -> inherits BaseModel.use_custom_prompt (returns False).
    assert cls.use_custom_prompt is BaseModel.use_custom_prompt


def test_cosmos_overrides_use_custom_prompt_to_false():
    # Cosmos overrides it explicitly (different object) but still returns False.
    assert RBLNCosmosReason1.use_custom_prompt is not BaseModel.use_custom_prompt
    assert RBLNCosmosReason1.use_custom_prompt(None, 'MMBench_DEV_EN') is False
    assert RBLNCosmosReason1.use_custom_prompt(None, 'Video-MME') is False


# ----------------------------------------------------------------------
# BLIP-2 decode-trim differentiator
# ----------------------------------------------------------------------

def test_blip2_decode_flags():
    # optimum-rbln BLIP-2 generate returns the full sequence incl. the
    # prompt, so trim stays ON (inherited True); strip drops trailing ws.
    assert RBLNBlip2._DECODE_TRIM is True
    assert RBLNBlip2._DECODE_STRIP is True
    assert RBLNBlip2.INTERLEAVE is False


# ----------------------------------------------------------------------
# Family flags (migrated from test_registry_integrity.py)
# ----------------------------------------------------------------------

def test_family_flags():
    # Video-capable families
    assert RBLNQwen2VL.VIDEO_LLM is True
    assert RBLNCosmosReason1.VIDEO_LLM is True  # inherits RBLNQwen2VL
    # Single-image families
    for cls in (RBLNPaliGemma, RBLNPaliGemma2, RBLNBlip2):
        assert cls.INTERLEAVE is False
    # Interleave-capable families default to True (inherited from RBLNVLMBase)
    for cls in (RBLNQwen2VL, RBLNLlava, RBLNLlavaNext, RBLNIdefics3,
                RBLNGemma3, RBLNPixtral):
        assert cls.INTERLEAVE is True
    # BLIP-2: generate returns the full sequence incl. prompt, so trim
    # stays ON (inherited); strip drops trailing whitespace.
    assert RBLNBlip2._DECODE_TRIM is True
    assert RBLNBlip2._DECODE_STRIP is True
