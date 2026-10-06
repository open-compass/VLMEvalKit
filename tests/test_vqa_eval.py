import importlib.util
import sys
import types
import unittest
from unittest import mock

import numpy  # noqa: F401  (imported before patching sys.modules so it stays loaded)


def _load_vqa_eval():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    smp = types.ModuleType('vlmeval.smp')
    smp.istype = lambda *args, **kwargs: False
    smp.listinstr = lambda lst, s: any(x in s for x in lst)
    smp.process_punctuation = lambda s: s

    modules = {'vlmeval': vlmeval, 'vlmeval.smp': smp}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.utils.vqa_eval',
            'vlmeval/dataset/utils/vqa_eval.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestAnlsCompute(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vqa_eval = _load_vqa_eval()

    def test_whitespace_does_not_lower_distance(self):
        anls_compute = self.vqa_eval.anls_compute
        padded = ' ' * 100 + 'London\n\n'
        self.assertEqual(anls_compute('Paris', 'London'), anls_compute('Paris', padded))
        self.assertEqual(anls_compute('Paris', '  paris  '), 0.0)

    def test_empty(self):
        self.assertEqual(self.vqa_eval.anls_compute('', '   '), 0.0)


if __name__ == '__main__':
    unittest.main()
