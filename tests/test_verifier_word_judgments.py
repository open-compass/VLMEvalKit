"""Offline regressions for the local verifier's judgment normalisation.

QUESTION_QUALITY_PROMPT_EN_* sanctions two reply shapes:

    'Just return the letters "A", "B", or "C", with no text around it.'
    'Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID.'

_process_judgment handled only the letter shape. Its last-resort
``re.findall(r"([A-C])", ...)`` reduced ``INVALID`` to its ``A`` and both
``CORRECT`` and ``INCORRECT`` to a ``C``, so a judge that obeyed the second
sentence had ``INVALID`` scored correct and ``CORRECT`` scored incorrect.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_verifier_module():
    pkg = sys.modules.setdefault('vlmeval', types.ModuleType('vlmeval'))
    pkg.__path__ = []
    smp = sys.modules.setdefault('vlmeval.smp', types.ModuleType('vlmeval.smp'))
    smp.__path__ = []
    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = lambda *a, **k: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None
    )
    sys.modules['vlmeval.smp.log'] = log

    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/utils/verifier.py'
    spec = importlib.util.spec_from_file_location('verifier_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Inputs(dict):
    """Stands in for a BatchEncoding: mapping for **kwargs, plus input_ids."""

    def __init__(self, ids):
        super().__init__(input_ids=ids)
        self.input_ids = ids

    def to(self, _device):
        return self


class _FakeTokenizer:
    def __init__(self, reply):
        self._reply = reply

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]['content']

    def __call__(self, _texts, **kwargs):
        return _Inputs(np.array([[0]]))

    def decode(self, _ids, **kwargs):
        return self._reply


class _FakeModel:
    device = 'cpu'

    def generate(self, **kwargs):
        # One prompt token echoed back plus one generated token.
        return np.array([[0, 1]])


@pytest.fixture(scope='module')
def verifier_module():
    return _load_verifier_module()


def _verifier(verifier_module, judge_reply, use_cot=False):
    """Build a Verifier_Model without loading a checkpoint."""
    model = object.__new__(verifier_module.Verifier_Model)
    model.tokenizer = _FakeTokenizer(judge_reply)
    model.model = _FakeModel()
    model.use_vllm = False
    model.use_cot = use_cot
    return model


# (judge reply, expected letter, expected verdict from evaluate())
WORD_REPLIES = [
    ('INVALID', 'C', False),
    ('CORRECT', 'A', True),
    ('INCORRECT', 'B', False),
    ('invalid', 'C', False),
    ('correct', 'A', True),
    ('incorrect', 'B', False),
    ('Final Judgment: CORRECT', 'A', True),
    ('Final Judgment: INVALID', 'C', False),
]


@pytest.mark.parametrize('reply, letter, _verdict', WORD_REPLIES)
def test_word_replies_normalise_to_the_matching_letter(verifier_module, reply, letter, _verdict):
    assert verifier_module.Verifier_Model._process_judgment(None, reply) == letter


@pytest.mark.parametrize('reply, _letter, verdict', WORD_REPLIES)
def test_word_replies_produce_the_matching_verdict(verifier_module, reply, _letter, verdict):
    """Drives evaluate() end to end, so the caller's own test is exercised."""
    model = _verifier(verifier_module, reply)
    assert model.evaluate('q', 'pred', 'gold') is verdict


# The letter shape, which already worked, must keep working.
LETTER_REPLIES = [
    ('A', 'A', True),
    ('B', 'B', False),
    ('C', 'C', False),
    ('\\boxed{A}', 'A', True),
    ('\\boxed{B}', 'B', False),
    ('Final Judgment: A', 'A', True),
    ('Final Judgment: B', 'B', False),
    ('Final Judgment: (C)', 'C', False),
]


@pytest.mark.parametrize('reply, letter, verdict', LETTER_REPLIES)
def test_letter_replies_keep_working(verifier_module, reply, letter, verdict):
    assert verifier_module.Verifier_Model._process_judgment(None, reply) == letter
    assert _verifier(verifier_module, reply).evaluate('q', 'pred', 'gold') is verdict


def test_negated_prose_still_defers_to_the_final_judgment(verifier_module):
    """A rationale containing the word 'correct' must not override the verdict."""
    reply = "The candidate's answer is not correct.\nFinal Judgment: B"
    assert verifier_module.Verifier_Model._process_judgment(None, reply) == 'B'
    assert _verifier(verifier_module, reply).evaluate('q', 'pred', 'gold') is False


def test_unparseable_reply_is_not_scored_correct(verifier_module):
    assert _verifier(verifier_module, 'hmm').evaluate('q', 'pred', 'gold') is False
