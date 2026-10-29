import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def _load_util(name):
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp.load = lambda data: data
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.load = lambda data: data
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    utils = types.ModuleType('vlmeval.dataset.utils')
    utils.__path__ = ['vlmeval/dataset/utils']
    multiple_choice = types.ModuleType('vlmeval.dataset.utils.multiple_choice')
    multiple_choice.extract_answer_from_item = None

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.file': smp_file,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.utils': utils,
        'vlmeval.dataset.utils.multiple_choice': multiple_choice,
        'torch': types.ModuleType('torch'),
        'torchvision': types.ModuleType('torchvision'),
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            f'vlmeval.dataset.utils.{name}',
            f'vlmeval/dataset/utils/{name}.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestVideoMCQDimensionRating(unittest.TestCase):

    def test_failed_api_answers_are_not_counted_as_correct(self):
        # MVBench / MVBench_MP4 / MVTamperBench evaluate() write score -1 when the
        # prediction is 'Failed to obtain answer via API.'.
        data = pd.DataFrame({
            'task_type': ['Action Sequence'] * 4,
            'tamper_type': ['dropping'] * 4,
            'score': [1, 0, -1, -1],
        })

        for name in ['mvbench', 'tamperbench']:
            with self.subTest(module=name):
                rating = _load_util(name).get_dimension_rating(data)

                self.assertEqual(rating['Action Sequence'], [1, 4, '25.00%'])
                self.assertEqual(rating['overall'], [1, 4, '25.00%'])

    def test_tamperbench_rating_by_other_category(self):
        data = pd.DataFrame({
            'task_type': ['Action Sequence'] * 2,
            'tamper_type': ['dropping', 'masking'],
            'score': [1, -1],
        })

        rating = _load_util('tamperbench').get_dimension_rating(data, 'tamper_type')

        self.assertEqual(rating['dropping'], [1, 1, '100.00%'])
        self.assertEqual(rating['masking'], [0, 1, '0.00%'])


if __name__ == '__main__':
    unittest.main()
