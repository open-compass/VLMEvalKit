"""Offline regression for VQA answer normalization of periods after digits."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVQAPeriodStrip(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the leaf modules without importing optional model backends.
        cls.misc = load('vlmeval_smp_misc_under_test', ROOT / 'vlmeval/smp/misc.py')
        smp = types.ModuleType('vlmeval.smp')
        smp.istype = cls.misc.istype
        smp.listinstr = cls.misc.listinstr
        smp.process_punctuation = cls.misc.process_punctuation
        with mock.patch.dict(sys.modules, {'vlmeval': types.ModuleType('vlmeval'), 'vlmeval.smp': smp}):
            cls.vqa_eval = load('vlmeval_vqa_eval_under_test', ROOT / 'vlmeval/dataset/utils/vqa_eval.py')

    def test_matches_official_vqa_period_strip(self):
        # Official VQA / TextVQA evaluators use periodStrip = '(?!<=\d)(\.)(?!\d)',
        # which also removes a period that follows a digit.
        cases = {
            '22.': '22',
            '27. April': '27 April',
            'NT$31,888.': 'NT$31888',
            '3.5': '3.5',
            'a.m.': 'am',
            'yes.': 'yes',
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.misc.process_punctuation(text), expected)

    def test_textvqa_credits_number_with_trailing_period(self):
        line = {'answer': str(['22'] * 10), 'prediction': '22.'}
        res = self.vqa_eval.process_line(line, method='vqa_score')
        self.assertEqual(self.vqa_eval.hit_calculate([res], 'TextVQA_VAL'), [1.0])


if __name__ == '__main__':
    unittest.main()
