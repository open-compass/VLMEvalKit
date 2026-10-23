"""Offline regressions for Yes/No extraction from markdown-emphasised answers.

The Yes/No datasets append 'Please answer yes or no.' to every prompt and, with
the default --judge exact_matching, YOrN_Extraction's verdict is final:
ImageYORNDataset.evaluate scores `data['answer'] == data['extracted']`, so an
extraction of 'Unknown' is a zero.

process_punctuation already strips the other markdown decorators (`_` and
backticks) but not `*`, so '**Yes**' survived as one token and extracted as
'Unknown'.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_real_process_punctuation():
    """Load the genuine helper from vlmeval/smp/misc.py, not a stub.

    The behaviour under test is the interaction between YOrN_Extraction and
    this function, so substituting it would prove nothing.
    """
    spec = importlib.util.spec_from_file_location(
        'yorn_test_smp_misc', REPO / 'vlmeval/smp/misc.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.process_punctuation


@pytest.fixture(scope='module')
def yorn_extraction():
    pkg = sys.modules.setdefault('vlmeval', types.ModuleType('vlmeval'))
    pkg.__path__ = []
    smp = types.ModuleType('vlmeval.smp')
    smp.process_punctuation = _load_real_process_punctuation()
    smp.d2df = lambda *a, **k: None      # only used by the rating helpers
    smp.load = lambda *a, **k: None      # only used by the rating helpers
    sys.modules['vlmeval.smp'] = smp

    spec = importlib.util.spec_from_file_location(
        'yorn_under_test', REPO / 'vlmeval/dataset/utils/yorn.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.YOrN_Extraction


@pytest.mark.parametrize('prediction, expected', [
    ('**Yes**', 'Yes'),
    ('*Yes*', 'Yes'),
    ('**No**', 'No'),
    ('*No*', 'No'),
    ('**Yes**, the image shows a dog.', 'Yes'),
    ('Answer: **No**', 'No'),
    ('**Yes**.', 'Yes'),
    ('The answer is **no**.', 'No'),
])
def test_markdown_emphasis_does_not_hide_the_verdict(yorn_extraction, prediction, expected):
    assert yorn_extraction(prediction) == expected


@pytest.mark.parametrize('prediction, expected', [
    # The decorators process_punctuation already handled must keep working.
    ('Yes', 'Yes'),
    ('Yes.', 'Yes'),
    ('yes', 'Yes'),
    ('_Yes_', 'Yes'),
    ('`Yes`', 'Yes'),
    ('No', 'No'),
    ('No.', 'No'),
    ('Yes, there is a dog.', 'Yes'),
    ('No, there is no dog in this image.', 'No'),
])
def test_existing_shapes_keep_working(yorn_extraction, prediction, expected):
    assert yorn_extraction(prediction) == expected


@pytest.mark.parametrize('prediction', [
    'I cannot tell from this image.',
    'Yes and no.',
    '**Yes** or **no**, depending on the lighting.',
    'Maybe.',
])
def test_ambiguous_or_absent_verdicts_stay_unknown(yorn_extraction, prediction):
    """Both-or-neither must remain Unknown so the judge fallback still fires."""
    assert yorn_extraction(prediction) == 'Unknown'
