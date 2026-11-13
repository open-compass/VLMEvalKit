"""Offline regressions for UniSVG category-accuracy matching.

The category branch of _evaluate_row matched the raw model answer against two
hand-rolled casings of the ground truth:

    correct_lower = set(c.lower() for c in cats_gt)
    correct_cap   = set(c.capitalize() for c in cats_gt)
    matched = any(c in model_answer for c in correct_lower) or ...

str.capitalize() also lowercases the remainder, so a title-cased multi-word
category ("Red Panda") could never match, and an upper-cased answer never
could. The colour branch twenty lines above lowercases model_answer instead,
which is the comparison this file's other branch already uses.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def category_accuracy():
    """Return a helper that scores one category row.

    _evaluate_row carries @timeout(60., use_signals=False), which runs the
    call in a separate process; __wrapped__ is the same function object
    without that wrapper, so the real branch logic runs in-process.
    """
    for name in ('vlmeval', 'vlmeval.smp', 'vlmeval.utils'):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    smp = sys.modules['vlmeval.smp']
    defaults = (
        ('LMUDataRoot', lambda *a, **k: '/tmp'),
        ('dump', lambda *a, **k: None),
        ('load', lambda *a, **k: None),
        ('get_intermediate_file_path', lambda *a, **k: ''),
    )
    for attr, value in defaults:
        if not hasattr(smp, attr):
            setattr(smp, attr, value)
    if not hasattr(smp, 'get_logger'):
        smp.get_logger = lambda *a, **k: types.SimpleNamespace(
            info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None
        )
    if 'vlmeval.utils.mp_util' not in sys.modules:
        mp = types.ModuleType('vlmeval.utils.mp_util')
        mp.track_progress_rich = lambda *a, **k: []
        sys.modules['vlmeval.utils.mp_util'] = mp
    # Rendering and embedding backends are only reached by the generation
    # branches, never by the category branch under test.
    for heavy in ('torch', 'sentence_transformers'):
        if heavy not in sys.modules:
            mod = types.ModuleType(heavy)
            if heavy == 'sentence_transformers':
                mod.util = types.SimpleNamespace(cos_sim=lambda *a, **k: 0)
            sys.modules[heavy] = mod

    spec = importlib.util.spec_from_file_location(
        'uni_svg_under_test', REPO / 'vlmeval/dataset/utils/uni_svg.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['uni_svg_under_test'] = module
    spec.loader.exec_module(module)
    body = module._evaluate_row.__wrapped__

    def _accuracy(categories, model_answer, q_type='CSVGUN_category'):
        _idx, metrics = body(
            0, model_answer, '', q_type,
            {'keywords': {'category': categories}}, '/tmp', 'cpu',
        )
        return metrics['accuracy']

    return _accuracy


CASING_VARIANTS = [
    (['butterfly'], 'This SVG depicts a BUTTERFLY.'),
    (['red panda'], 'This is a Red Panda.'),
    (['red panda'], 'This is a RED PANDA.'),
    (['sea turtle'], 'The image shows a Sea Turtle swimming.'),
]


@pytest.mark.parametrize('q_type', ['CSVGUN_category', 'ISVGUN_category'])
@pytest.mark.parametrize('categories, model_answer', CASING_VARIANTS)
def test_casing_does_not_hide_a_correct_category(category_accuracy, q_type, categories, model_answer):
    assert category_accuracy(categories, model_answer, q_type) == 1


@pytest.mark.parametrize('categories, model_answer', [
    # Shapes that already matched must keep matching.
    (['butterfly'], 'This SVG depicts a butterfly.'),
    (['butterfly'], 'This SVG depicts a Butterfly.'),
    (['red panda'], 'This is a red panda.'),
    (['Butterfly'], 'This is a butterfly.'),
])
def test_shapes_that_already_matched_keep_matching(category_accuracy, categories, model_answer):
    assert category_accuracy(categories, model_answer) == 1


@pytest.mark.parametrize('categories, model_answer', [
    (['butterfly'], 'This is a dog.'),
    (['red panda'], 'This is a giant panda.'),
    (['sea turtle'], 'I cannot identify the subject.'),
])
def test_wrong_categories_still_score_zero(category_accuracy, categories, model_answer):
    assert category_accuracy(categories, model_answer) == 0


def test_any_of_several_ground_truth_categories_counts(category_accuracy):
    assert category_accuracy(['red panda', 'raccoon'], 'Looks like a Raccoon to me.') == 1


def test_empty_category_list_scores_zero(category_accuracy):
    assert category_accuracy([], 'This is a butterfly.') == 0
