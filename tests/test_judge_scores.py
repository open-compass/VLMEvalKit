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


def load_zerobench_evaluator():
    import ast
    import re
    from pathlib import Path

    import pandas as pd

    source = Path('vlmeval/dataset/image_vqa.py')
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'ZEROBench')
    namespace = {'ImageVQADataset': object, 'pd': pd, 're': re,
                 'load': lambda filename: pd.read_csv(filename, keep_default_na=False)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['ZEROBench']()


class TestZeroBenchAnswerProtocol(unittest.TestCase):

    def score(self, answer, prediction):
        import tempfile
        from pathlib import Path

        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'predictions.csv'
            row = {'index': 1, 'answers': answer, 'prediction': prediction}
            pd.DataFrame([row]).to_csv(table, index=False)
            return load_zerobench_evaluator().evaluate(str(table))['accuracy']

    def test_larger_number_containing_gold_is_not_correct(self):
        self.assertEqual(self.score('2', '{12}'), 0.0)

    def test_word_containing_gold_is_not_correct(self):
        self.assertEqual(self.score('red', '{redwood}'), 0.0)

    def test_explanatory_brace_content_is_not_an_exact_answer(self):
        self.assertEqual(self.score('2', '{2 or 3}'), 0.0)

    def test_exact_answer_case_normalization_and_last_brace_match(self):
        self.assertEqual(self.score('RED', '{blue} then {red}'), 1.0)

    def test_answer_without_braces_remains_incorrect(self):
        self.assertEqual(self.score('2', 'The answer is 2'), 0.0)

    def test_wrong_final_brace_answer_remains_incorrect(self):
        self.assertEqual(self.score('2', '{2} then {3}'), 0.0)


if __name__ == '__main__':
    unittest.main()
