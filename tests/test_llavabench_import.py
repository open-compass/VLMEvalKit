import importlib.util
import sys
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd


class TestLLaVABenchImport(unittest.TestCase):

    def test_evaluation_module_uses_the_top_level_logger(self):
        package = types.ModuleType('vlmeval')
        package.__path__ = ['vlmeval']
        dataset = types.ModuleType('vlmeval.dataset')
        dataset.__path__ = ['vlmeval/dataset']
        utils = types.ModuleType('vlmeval.dataset.utils')
        utils.__path__ = ['vlmeval/dataset/utils']
        smp = types.ModuleType('vlmeval.smp')
        smp.__path__ = ['vlmeval/smp']
        modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.dataset.utils': utils,
                   'vlmeval.smp': smp}
        with mock.patch.dict(sys.modules, modules):
            log_spec = importlib.util.spec_from_file_location('vlmeval.smp.log', 'vlmeval/smp/log.py')
            log = importlib.util.module_from_spec(log_spec)
            sys.modules[log_spec.name] = log
            log_spec.loader.exec_module(log)
            spec = importlib.util.spec_from_file_location(
                'vlmeval.dataset.utils.llavabench', 'vlmeval/dataset/utils/llavabench.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.assertIs(module.logger, log.get_logger(module.__name__))
        judge = mock.Mock()
        judge.generate.return_value = '8, 9\nExplanation'
        self.assertEqual(module.LLaVABench_atomeval(judge, 'Judge this pair'), [8.0, 9.0])
        for reply in ['8  9', '8\t9', ' 8,  9 ']:
            self.assertEqual(module.parse_score(reply), [8.0, 9.0])
        for reply in ['nan 9', '8 inf']:
            self.assertEqual(module.parse_score(reply), [-1, -1])
        for reply in ['bad', 'eight nine']:
            with self.assertLogs(module.logger, level='ERROR') as logs:
                self.assertEqual(module.parse_score(reply), [-1, -1])
            self.assertIn(reply, logs.output[0])
        data = pd.DataFrame([{'category': 'conv', 'gpt4_score': 8.0, 'score': 9.0}])
        result = module.LLaVABench_score(data)
        np.testing.assert_allclose(result['Relative Score (main)'], [112.5, 112.5])


if __name__ == '__main__':
    unittest.main()
