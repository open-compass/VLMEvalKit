"""Offline regressions for the shared MCQ answer-prefix strip lists.

Five copies of ``extract_characters_regex`` list the prefixes they strip from a
model response. Two entries in each list are missing their trailing comma, so
Python concatenates the adjacent strings and the four intended prefixes --
'The best option is', 'The correct option is', 'Best answer:', 'Best option:' --
are never stripped. A response of ``Best answer: C`` then matches the ``B`` of
``Best`` and is scored as option B.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# module dotted name -> (file path, option letters)
TARGETS = {
    'vlmeval.dataset.utils.videomme': ('vlmeval/dataset/utils/videomme.py', 'ABCD'),
    'vlmeval.dataset.utils.worldsense': ('vlmeval/dataset/utils/worldsense.py', 'ABCD'),
    'vlmeval.dataset.utils.videott': ('vlmeval/dataset/utils/videott.py', 'ABCD'),
    'vlmeval.dataset.utils.longvideobench': ('vlmeval/dataset/utils/longvideobench.py', 'ABCDE'),
    'vlmeval.dataset.EgoExoBench.utils': ('vlmeval/dataset/EgoExoBench/utils.py', 'ABCD'),
}


def _stub_environment():
    """Register the parent packages and the few leaf imports these modules need.

    Only file-IO helpers and a judge-model entry point are stubbed; none is
    reachable from extract_characters_regex, which is pure string handling.
    """
    for name in ('vlmeval', 'vlmeval.dataset', 'vlmeval.dataset.utils',
                 'vlmeval.dataset.EgoExoBench', 'vlmeval.smp'):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
    sys.modules['vlmeval.smp'].load = lambda *a, **k: None
    sys.modules['vlmeval.smp'].dump = lambda *a, **k: None

    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.load = lambda *a, **k: None
    sys.modules['vlmeval.smp.file'] = smp_file

    mc = types.ModuleType('vlmeval.dataset.utils.multiple_choice')
    mc.extract_answer_from_item = lambda *a, **k: {'opt': ''}
    sys.modules['vlmeval.dataset.utils.multiple_choice'] = mc

    for heavy in ('torch', 'torchvision'):
        if heavy not in sys.modules:
            sys.modules[heavy] = types.ModuleType(heavy)


def _load(dotted):
    _stub_environment()
    rel, _letters = TARGETS[dotted]
    spec = importlib.util.spec_from_file_location(dotted, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=sorted(TARGETS))
def extractor(request):
    module = _load(request.param)
    return request.param, module.extract_characters_regex, TARGETS[request.param][1]


def test_best_answer_prefix_is_stripped(extractor):
    """'Best answer: C' must yield C, not the B of 'Best'."""
    name, extract, letters = extractor
    for letter in letters:
        assert extract('Best answer: {}'.format(letter)) == letter, name


def test_best_option_prefix_is_stripped(extractor):
    name, extract, letters = extractor
    for letter in letters:
        assert extract('Best option: {}'.format(letter)) == letter, name


def test_option_wording_prefixes_are_stripped(extractor):
    name, extract, letters = extractor
    for letter in letters:
        assert extract('The best option is {}'.format(letter)) == letter, name
        assert extract('The correct option is {}'.format(letter)) == letter, name


def test_existing_prefixes_keep_working(extractor):
    """Prefixes that already had their comma must be unaffected."""
    name, extract, letters = extractor
    for letter in letters:
        for prefix in ('The best answer is', 'The correct answer is', 'The answer is',
                       'Answer:', 'Option:'):
            assert extract('{} {}'.format(prefix, letter)) == letter, (name, prefix)


def test_bare_letter_and_no_match_unchanged(extractor):
    name, extract, letters = extractor
    assert extract(letters[0]) == letters[0], name
    assert extract('I cannot tell from this video.') == '', name
