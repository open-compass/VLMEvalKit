import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def load_crpe():
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    base = types.ModuleType('vlmeval.dataset.image_base')
    base.ImageBaseDataset = type('ImageBaseDataset', (), {})
    smp = types.ModuleType('vlmeval.smp')
    smp.__getattr__ = lambda name: mock.Mock()
    utils = types.ModuleType('vlmeval.utils')
    utils.track_progress_rich = mock.Mock()
    dataset_utils = types.ModuleType('vlmeval.dataset.utils')
    dataset_utils.__path__ = ['vlmeval/dataset/utils']
    dataset_utils.DEBUG_MESSAGE = ''
    dataset_utils.build_judge = mock.Mock()
    cache = types.ModuleType('vlmeval.dataset.utils.judge_cache')
    cache.__getattr__ = lambda name: mock.Mock()
    vqa = types.ModuleType('vlmeval.dataset.utils.vqa_eval')
    vqa.istype = mock.Mock()
    modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.dataset.image_base': base,
               'vlmeval.smp': smp, 'vlmeval.utils': utils, 'vlmeval.dataset.utils': dataset_utils,
               'vlmeval.dataset.utils.judge_cache': cache, 'vlmeval.dataset.utils.vqa_eval': vqa}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.image_vqa', 'vlmeval/dataset/image_vqa.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        crpe = importlib.import_module('vlmeval.dataset.utils.crpe')
    return module, crpe


class TestCRPEEmptyResponse(unittest.TestCase):

    def test_empty_response_marks_the_four_question_group_incorrect(self):
        module, crpe = load_crpe()
        data = pd.DataFrame({'prediction': ['A', '', 'A', 'A', 'A', 'A', 'A', 'A'],
                             'answer': ['A. cat'] * 8, 'category': ['exist'] * 8})
        with mock.patch.object(module, 'load', return_value=data), \
                mock.patch.dict(sys.modules, {'vlmeval.dataset.utils.crpe': crpe}):
            result = module.CRPE.evaluate('predictions.xlsx')
        self.assertEqual(result['total'], 0.5)
        self.assertEqual(result['exist'], 0.5)

    def test_valid_letter_and_text_answers_keep_their_behavior(self):
        _, crpe = load_crpe()
        self.assertTrue(crpe.is_correct('A. cat', 'A'))
        self.assertTrue(crpe.is_correct('A. cat', 'The cat is visible'))
        self.assertFalse(crpe.is_correct('A. cat', 'B'))
        self.assertFalse(crpe.is_correct('A. cat', ''))


if __name__ == '__main__':
    unittest.main()
