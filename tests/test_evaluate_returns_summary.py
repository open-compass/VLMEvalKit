"""`dataset.evaluate()` must return its metrics.

`run.py` only records a score for a dataset when `dataset.evaluate()` returns a
``dict`` or a ``pandas.DataFrame`` -- an implicit ``None`` is reported as
``skip_reason='evaluate_returned_none'`` in ``status.json``, so a benchmark that
ran to completion silently reports nothing (issue #1690).

The judge/OCR datasets below used to discard their summary, so these tests pin
the return contract. Like ``tests/test_logicvista.py``, they load the dataset
modules from file with their heavy imports stubbed out, and call ``evaluate``
unbound (no real dataset/network access needed).
"""

import importlib.util
import sys
import types
import unittest
from unittest import mock

SUMMARY = {'overall': 42.0, 'doc_en': 40.0}


class FakeLine(dict):
    pass


class FakeDF(dict):
    """Minimal stand-in for the prediction table returned by ``load()``."""

    def __init__(self, rows):
        super().__init__()
        self.iloc = [FakeLine(row) for row in rows]

    def __len__(self):
        return len(self.iloc)


def _pkg(name):
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _stub_modules():
    modules = {}

    for name in ('vlmeval', 'vlmeval.dataset', 'vlmeval.dataset.olmOCRBench'):
        modules[name] = _pkg(name)
    ds_utils = _pkg('vlmeval.dataset.utils')
    ds_utils.build_judge = lambda *a, **k: None
    ds_utils.levenshtein_distance = lambda a, b: 0
    modules['vlmeval.dataset.utils'] = ds_utils

    smp = _pkg('vlmeval.smp')
    modules['vlmeval.smp'] = smp
    for attr in ('dump', 'load', 'md5', 'read_ok', 'toliststr', 'download_file',
                 'decode_base64_to_image_file', 'encode_image_to_base64', 'LMUDataRoot',
                 'listinstr'):
        setattr(smp, attr, lambda *a, **k: None)
    smp.get_logger = lambda name: __import__('logging').getLogger(name)
    smp.dump = lambda *a, **k: None
    smp.load = lambda eval_file: FakeDF([{'index': 0}])

    image_base = _pkg('vlmeval.dataset.image_base')
    image_base.ImageBaseDataset = type('ImageBaseDataset', (), {})
    modules['vlmeval.dataset.image_base'] = image_base

    judge_cache = _pkg('vlmeval.dataset.utils.judge_cache')
    judge_cache.get_judge_detail_file = lambda ef, tag, judge: 'detail.csv'
    judge_cache.get_judge_cache_file = lambda ef, tag, judge: 'cache.json'
    judge_cache.get_judge_named_legacy_cache_file = lambda ef, judge: 'legacy.json'
    judge_cache.get_judge_score_file = lambda ef, judge, ext: 'score.csv'
    judge_cache.load_judge_cache = lambda *a, **k: {0: {'res': 'True'}}
    judge_cache.run_cached_tasks = lambda *a, **k: {}
    modules['vlmeval.dataset.utils.judge_cache'] = judge_cache

    judge_util = _pkg('vlmeval.dataset.utils.judge_util')
    judge_util.build_judge = lambda *a, **k: None
    modules['vlmeval.dataset.utils.judge_util'] = judge_util

    doc = _pkg('vlmeval.dataset.mmlongbenchdoc')
    doc.MMLongBench_auxeval = lambda *a, **k: None
    doc.MMLongBench_judge_failed = lambda record: False
    doc.anls_compute = lambda answer, pred: 1.0
    doc.concat_images = lambda *a, **k: None
    modules['vlmeval.dataset.mmlongbenchdoc'] = doc

    # pandas / PIL / torchvision are only imported for their side effects here
    modules['pandas'] = types.ModuleType('pandas')
    pil = _pkg('PIL')
    pil.Image = types.ModuleType('PIL.Image')
    pil.ImageDraw = types.ModuleType('PIL.ImageDraw')
    pil.ImageFont = types.ModuleType('PIL.ImageFont')
    modules['PIL'] = pil
    modules['PIL.Image'] = pil.Image
    modules['torchvision'] = _pkg('torchvision')
    modules['torchvision.transforms'] = types.ModuleType('torchvision.transforms')
    return modules


def _load_dataset(path, name, modules):
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        return module


class TestJudgeDatasetEvaluateReturnsSummary(unittest.TestCase):
    """DUDE / SlideVQA / MMLongBench-Doc share the same judge-then-score body."""

    def _check(self, dataset_file, module_name, acc_name):
        modules = _stub_modules()
        module = _load_dataset(dataset_file, module_name, modules)
        # the aggregate helper is what the dataset reports; make it observable
        module.__dict__[acc_name] = lambda storage: SUMMARY
        dataset_cls = next(
            obj for obj in vars(module).values()
            if isinstance(obj, type) and hasattr(obj, 'evaluate')
        )
        dataset_obj = types.SimpleNamespace(DEFAULT_JUDGE_MODEL='judge', data_path='data.tsv')
        # evaluate() is a classmethod on these datasets
        result = dataset_cls.evaluate('pred.xlsx')
        self.assertEqual(result, SUMMARY, f'{dataset_cls.__name__}.evaluate() dropped its score')

    def test_dude(self):
        self._check('vlmeval/dataset/dude.py', 'vlmeval.dataset.dude', 'DUDE_acc')

    def test_slidevqa(self):
        self._check('vlmeval/dataset/slidevqa.py', 'vlmeval.dataset.slidevqa', 'SlideVQA_acc')

    def test_mmlongbenchdoc(self):
        self._check('vlmeval/dataset/mmlongbenchdoc.py', 'vlmeval.dataset.mmlongbenchdoc',
                    'MMLongBench_acc')


class TestOlmOCRBenchEvaluateReturnsSummary(unittest.TestCase):

    def test_olmocrbench(self):
        modules = _stub_modules()
        module = _load_dataset('vlmeval/dataset/olmOCRBench/olmocrbench.py',
                               'vlmeval.dataset.olmOCRBench.olmocrbench', modules)

        calls = []

        def fake_evaluator(tsv_path, eval_file):
            calls.append((tsv_path, eval_file))
            return SUMMARY

        evaluator_module = types.ModuleType('vlmeval.dataset.olmOCRBench.evaluator')
        evaluator_module.evaluator = fake_evaluator
        modules['vlmeval.dataset.olmOCRBench.evaluator'] = evaluator_module

        dataset_cls = next(
            obj for obj in vars(module).values()
            if isinstance(obj, type) and hasattr(obj, 'evaluate')
        )
        with mock.patch.dict(sys.modules, modules):
            if isinstance(dataset_cls.__dict__.get('evaluate'), classmethod):
                result = dataset_cls.evaluate('pred.xlsx')
            else:
                result = dataset_cls.evaluate(
                    types.SimpleNamespace(data_path='data.tsv'), 'pred.xlsx')

        self.assertEqual(calls, [('data.tsv', 'pred.xlsx')])
        self.assertEqual(result, SUMMARY, 'olmOCRBench.evaluate() dropped the summary')


if __name__ == '__main__':
    unittest.main()
