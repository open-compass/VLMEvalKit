import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def load_omtg_module():
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    base = types.ModuleType('vlmeval.dataset.video_base')
    base.VideoBaseDataset = type('VideoBaseDataset', (), {})
    smp = types.ModuleType('vlmeval.smp')
    for name in ['LMUDataRoot', 'dump', 'get_cache_path', 'load']:
        setattr(smp, name, mock.Mock())
    modules = {
        'vlmeval': package,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.video_base': base,
        'vlmeval.smp': smp,
    }
    path = Path(__file__).resolve().parents[1] / 'vlmeval/dataset/omtgbench.py'
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.omtgbench', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestOMTGTimeIntervals(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_omtg_module()

    def test_natural_language_preserves_complete_end_times(self):
        for template in ['It starts at {} and ends at {} seconds.',
                         'The start is at {} seconds and the end is at {} seconds.']:
            for start, end, expected in [
                ('12', '25', [[12.0, 25.0]]),
                ('1.5', '8.75', [[1.5, 8.75]]),
                ('00:12', '00:25', [[12.0, 25.0]]),
                ('01:02:03', '01:02:25.5', [[3723.0, 3745.5]]),
            ]:
                text = template.format(start, end)
                with self.subTest(text=text):
                    self.assertEqual(self.module.parse_time_intervals(text), expected)

    def test_existing_formats_and_invalid_intervals(self):
        for text, expected in [
            ('It starts at 1 and ends at 8 seconds', [[1.0, 8.0]]),
            ('It starts at 12 and ends at 25', [[12.0, 25.0]]),
            ('<time>12-25 seconds</time>', [[12.0, 25.0]]),
            ('12 - 25 seconds', [[12.0, 25.0]]),
            ('[[12, 25], [30, 42]]', [[12.0, 25.0], [30.0, 42.0]]),
            ('It starts at 25 and ends at 12 seconds', []),
            ('It starts at 12 and ends at 12 seconds', []),
            ('No matching event.', []),
        ]:
            with self.subTest(text=text):
                self.assertEqual(self.module.parse_time_intervals(text), expected)

    def test_evaluate_scores_equivalent_time_formats_as_perfect_matches(self):
        module = self.module
        dataset = module.OMTGBench.__new__(module.OMTGBench)
        dataset.dataset_name = 'OMTGBench'
        data = pd.DataFrame({
            'prediction': [
                'It starts at 12 and ends at 25 seconds.',
                'The start is at 1.5 and the end is at 8.75 seconds.',
            ],
            'answer': ['<time>12-25 seconds</time>', '<time>1.5-8.75 seconds</time>'],
        })
        with mock.patch.object(module, 'load', return_value=data), mock.patch.object(module, 'dump'):
            result = dataset.evaluate('predictions.xlsx')

        self.assertTrue(result)
        for metric, value in result.items():
            with self.subTest(metric=metric):
                self.assertAlmostEqual(value, 100.0)


if __name__ == '__main__':
    unittest.main()
