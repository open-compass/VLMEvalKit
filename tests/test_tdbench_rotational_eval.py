import importlib.util
import logging
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import pandas as pd


def _load_file_module():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = logging.getLogger
    misc = types.ModuleType('vlmeval.smp.misc')
    misc.toliststr = lambda value: value if isinstance(value, list) else [value]
    vlm = types.ModuleType('vlmeval.smp.vlm')
    vlm.decode_base64_to_image_file = mock.MagicMock()
    validators = types.ModuleType('validators')
    validators.url = lambda value: False

    modules = {
        'validators': validators,
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.log': log,
        'vlmeval.smp.misc': misc,
        'vlmeval.smp.vlm': vlm,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.smp.file', 'vlmeval/smp/file.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules['vlmeval.smp.file'] = module
        spec.loader.exec_module(module)
        sys.modules.pop('vlmeval.smp.file', None)
        return module


def _load_tdbench(file_module):
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp.load = file_module.load

    with mock.patch.dict(sys.modules, {'vlmeval': vlmeval, 'vlmeval.smp': smp}):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.utils.tdbench',
            'vlmeval/dataset/utils/tdbench.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestTDBenchRotationalEval(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.file_module = _load_file_module()
        cls.tdbench = _load_tdbench(cls.file_module)

    def test_reads_judged_results_from_xlsx(self):
        # TDBench.do_evaluate always writes the judged table as xlsx, independent of PRED_FORMAT,
        # and rotational_eval reads the same file for every rotation.
        hits = {'rot0': [1, 1], 'rot90': [1, 0], 'rot180': [0, 0], 'rot270': [1, 0]}
        with tempfile.TemporaryDirectory() as work_dir:
            for rot, hit in hits.items():
                frame = pd.DataFrame({'index': [1, 2], 'category': ['obj', 'obj'], 'hit': hit})
                path = os.path.join(work_dir, f'Model_tdbench_{rot}_exact_matching_result.xlsx')
                self.file_module.dump(frame, path)

            acc = self.tdbench.rotational_eval(os.path.join(work_dir, 'Model_tdbench_rot0_exact_matching_result.xlsx'))

            self.assertEqual(acc.loc['obj', 'total_questions'], 2)
            self.assertEqual(acc.loc['obj', 'hit_all'], 0.0)
            self.assertEqual(acc.loc['obj', 'miss_all'], 0.0)
            self.assertEqual(acc.loc['obj', 'rot0'], 1.0)
            self.assertEqual(acc.loc['obj', 'rot90'], 0.5)


if __name__ == '__main__':
    unittest.main()
