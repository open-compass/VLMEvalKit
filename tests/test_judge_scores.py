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


def load_zerobench_class(image_vqa_dataset=object):
    import ast
    import re
    from pathlib import Path

    import pandas as pd

    source = Path('vlmeval/dataset/image_vqa.py')
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'ZEROBench')
    namespace = {'ImageVQADataset': image_vqa_dataset, 'pd': pd, 're': re,
                 'load': lambda filename: pd.read_csv(
                     filename, keep_default_na=False,
                     sep='\t' if filename.endswith('.tsv') else ','),
                 'd2df': lambda result: pd.DataFrame({key: [value] for key, value in result.items()}),
                 'get_intermediate_file_path':
                 lambda eval_file, suffix, target_format: eval_file.replace(
                     f'.{eval_file.split(".")[-1]}', f'{suffix}.{target_format}'),
                 'dump': lambda data, filename: data.to_csv(filename, index=False)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['ZEROBench']


def load_zerobench_evaluator(k=5):
    cls = load_zerobench_class()
    evaluator = cls.__new__(cls)
    evaluator.k = k
    return evaluator


class TestZeroBenchAnswerProtocol(unittest.TestCase):

    def test_init_sets_k_before_calling_base_init(self):
        observed_k = []

        class ImageVQADataset:

            def __init__(self, dataset, skip_noimg):
                observed_k.append(self.k)

        cls = load_zerobench_class(ImageVQADataset)

        evaluator = cls(dataset='ZEROBench', k=3)

        self.assertEqual(observed_k, [3])
        self.assertEqual(evaluator.k, 3)

    def score(self, answer, prediction):
        import tempfile
        from pathlib import Path

        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'predictions.csv'
            row = {'index': 1, 'orig_index': 1, 'answers': answer, 'prediction': prediction}
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

    def test_accuracy_is_dumped_to_acc_csv(self):
        import tempfile
        from pathlib import Path

        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'predictions.tsv'
            row = {'index': 1, 'orig_index': 1, 'answers': '2', 'prediction': '{2}'}
            pd.DataFrame([row]).to_csv(table, index=False, sep='\t')

            load_zerobench_evaluator(k=1).evaluate(str(table))

            score_file = Path(directory) / 'predictions_acc.csv'
            self.assertTrue(score_file.exists())
            self.assertEqual(pd.read_csv(score_file).iloc[0]['accuracy'], 1.0)

    def test_post_build_repeats_each_case_k_times(self):
        import pandas as pd

        evaluator = load_zerobench_evaluator(k=3)
        evaluator.data = pd.DataFrame({
            'index': ['a', 'b'],
            'question': ['first', 'second'],
        })

        evaluator.post_build('ZEROBench')

        self.assertEqual(evaluator.data['index'].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(evaluator.data['orig_index'].tolist(), ['a', 'a', 'a', 'b', 'b', 'b'])
        self.assertEqual(evaluator.data['question'].tolist(),
                         ['first', 'first', 'first', 'second', 'second', 'second'])

    def test_pass_metrics_are_grouped_by_orig_index(self):
        import tempfile
        from pathlib import Path

        import pandas as pd

        rows = [
            {'index': 0, 'orig_index': 10, 'answers': '1', 'prediction': '{1}'},
            {'index': 1, 'orig_index': 10, 'answers': '1', 'prediction': '{2}'},
            {'index': 2, 'orig_index': 10, 'answers': '1', 'prediction': '{1}'},
            {'index': 3, 'orig_index': 20, 'answers': '2', 'prediction': '{3}'},
            {'index': 4, 'orig_index': 20, 'answers': '2', 'prediction': '{4}'},
            {'index': 5, 'orig_index': 20, 'answers': '2', 'prediction': '{5}'},
            {'index': 6, 'orig_index': 30, 'answers': '6', 'prediction': '{6}'},
            {'index': 7, 'orig_index': 30, 'answers': '6', 'prediction': '{6}'},
            {'index': 8, 'orig_index': 30, 'answers': '6', 'prediction': '{6}'},
        ]

        with tempfile.TemporaryDirectory() as directory:
            table = Path(directory) / 'predictions.tsv'
            pd.DataFrame(rows).to_csv(table, index=False, sep='\t')

            result = load_zerobench_evaluator(k=3).evaluate(str(table))

            self.assertAlmostEqual(result['accuracy'], 5 / 9)
            self.assertAlmostEqual(result['pass@3'], 2 / 3)
            self.assertAlmostEqual(result['pass^3'], 1 / 3)

            score = pd.read_csv(Path(directory) / 'predictions_acc.csv').iloc[0]
            self.assertAlmostEqual(score['accuracy'], 5 / 9)
            self.assertAlmostEqual(score['pass@3'], 2 / 3)
            self.assertAlmostEqual(score['pass^3'], 1 / 3)

            details = pd.read_csv(Path(directory) / 'predictions_results.csv')
            self.assertEqual(len(details), 9)
            self.assertEqual(details.columns.tolist(),
                             ['Question_ID', 'Ground_Truth', 'Model_Output', 'Correct?', 'orig_index'])
            self.assertEqual(details['orig_index'].tolist(), [10] * 3 + [20] * 3 + [30] * 3)


if __name__ == '__main__':
    unittest.main()
