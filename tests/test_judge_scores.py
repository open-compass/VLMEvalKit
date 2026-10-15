import importlib.util
import sys
import types
import unittest
from unittest import mock

import numpy as np


def load_score_module(name):
    smp = types.ModuleType('vlmeval.smp')
    smp.load = mock.Mock()
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.load = mock.Mock()
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    with mock.patch.dict(sys.modules, {'vlmeval': package, 'vlmeval.smp': smp, 'vlmeval.smp.file': smp_file}):
        spec = importlib.util.spec_from_file_location(
            f'vlmeval.dataset.utils.{name}', f'vlmeval/dataset/utils/{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestFiniteJudgeScores(unittest.TestCase):

    def test_mmvet_retries_nan_before_reporting_a_score(self):
        module = load_score_module('mmvet')
        judge = mock.Mock()
        judge.generate.side_effect = ['nan', '0.5']
        result = module.MMVet_auxeval(judge, {'question': 'What color?', 'answer': 'blue', 'prediction': 'blue'})
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(judge.generate.call_count, 2)
        self.assertIn('invalid score', result['log'])

    def test_mmvet_exhausted_nonfinite_outputs_report_failure_score(self):
        module = load_score_module('mmvet')
        judge = mock.Mock()
        judge.generate.return_value = 'nan'
        result = module.MMVet_auxeval(judge, {'question': 'What color?', 'answer': 'blue', 'prediction': 'blue'})
        self.assertEqual(result['score'], 0.0)
        self.assertIn('All 5 retries failed', result['log'])

    def test_mmoral_retries_nonfinite_scores(self):
        module = load_score_module('mmoral_opg')
        judge = mock.Mock()
        judge.generate.side_effect = ['nan', '0.5']
        line = {'question': 'How many teeth?', 'answer': '30', 'prediction': '30'}
        result = module.MMOral_opg_auxeval(judge, line)
        self.assertEqual(result['score'], 0.5)
        self.assertEqual(judge.generate.call_count, 2)
        self.assertIn('invalid score', result['log'])

    def test_valid_boundary_scores_remain_valid(self):
        module = load_score_module('mmvet')
        judge = mock.Mock()
        for value in ['0.0', '1.0']:
            judge.generate.return_value = value
            result = module.MMVet_auxeval(judge, {'question': 'What color?', 'answer': 'blue', 'prediction': 'blue'})
            self.assertEqual(result['score'], float(value))
            self.assertTrue(np.isfinite(result['score']))


if __name__ == '__main__':
    unittest.main()
