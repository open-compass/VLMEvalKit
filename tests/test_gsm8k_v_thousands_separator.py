"""Offline regressions for GSM8K-V numeric answer extraction with thousands separators.

The GSM8K-V prompt asks the model to emit ``FINAL ANSWER: <number>`` and
``_clean_extracted_number`` already strips ``,`` before validating, so a
US-formatted integer such as ``1,234`` is a well-formed answer that must
round-trip to ``1234.0``.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _install_smp_stub():
    """Stub the three vlmeval.smp helpers gsm8k_v imports at module scope.

    The helpers are only used by evaluate_gsm8k_v (file IO), never by the
    extractor under test, so stubbing keeps this regression offline.
    """
    pkg = sys.modules.setdefault('vlmeval', types.ModuleType('vlmeval'))
    pkg.__path__ = []
    smp = types.ModuleType('vlmeval.smp')
    smp.dump = lambda *a, **k: None
    smp.load = lambda *a, **k: None
    smp.get_intermediate_file_path = lambda *a, **k: ''
    sys.modules['vlmeval.smp'] = smp
    pkg.smp = smp


@pytest.fixture
def gsm8k_v():
    _install_smp_stub()
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/utils/gsm8k_v.py'
    spec = importlib.util.spec_from_file_location('gsm8k_v_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('response, expected', [
    ('FINAL ANSWER: 1,234', 1234.0),
    ('FINAL ANSWER: 2,500', 2500.0),
    ('Working it out step by step.\n\nFINAL ANSWER: 3,600', 3600.0),
    ('FINAL ANSWER: 1,234,567', 1234567.0),
    ('FINAL ANSWER: -1,500', -1500.0),
    ('FINAL ANSWER: 1,234.50', 1234.5),
    ('The answer is 12,000', 12000.0),
    ('#### 7,250', 7250.0),
])
def test_thousands_separators_are_not_truncated(gsm8k_v, response, expected):
    assert gsm8k_v.extract_numerical_answer(response) == expected


@pytest.mark.parametrize('response, expected', [
    ('FINAL ANSWER: 18', 18.0),
    ('FINAL ANSWER: 72.5', 72.5),
    ('FINAL ANSWER: -4', -4.0),
    ('FINAL ANSWER: 3/4', 0.75),
    ('FINAL ANSWER: $42', 42.0),
    ('The answer is 9', 9.0),
    ('#### 120', 120.0),
])
def test_plain_numbers_keep_working(gsm8k_v, response, expected):
    assert gsm8k_v.extract_numerical_answer(response) == expected


@pytest.mark.parametrize('response, expected', [
    # Bare year followed by a comma must not absorb the comma, and the
    # last number on the line stays the answer.
    ('In 2020, the shop sold 15 crates.', 15.0),
    ('She paid 1,250 dollars and got 50 back.', 50.0),
])
def test_trailing_commas_do_not_swallow_the_separator(gsm8k_v, response, expected):
    assert gsm8k_v.extract_numerical_answer(response) == expected
