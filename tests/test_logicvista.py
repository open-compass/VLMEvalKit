import importlib.util
import sys
import types
import unittest
from unittest import mock


def _load_logicvista():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.load = lambda path: path

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.file': smp_file,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.utils.logicvista',
            'vlmeval/dataset/utils/logicvista.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class FakeJudge:

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, temperature):
        self.calls.append((prompt, temperature))
        return next(self.responses)


class TestLogicVistaAnswerExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.logicvista = _load_logicvista()

    def test_accepts_numeric_answer(self):
        judge = FakeJudge(['3'])
        line = {
            'question': 'Which answer is correct? Select from 1-5',
            'prediction': '<answer>3</answer>',
            'answer': '3',
        }

        result = self.logicvista.LogicVista_auxeval(judge, line)

        self.assertEqual(result, {'log': 'Succeed', 'res': '3', 'hit': 1})
        self.assertEqual(len(judge.calls), 1)
        self.assertIn('option labels chosen (letters or numbers)', judge.calls[0][0])

    def test_normalizes_case_order_and_separators(self):
        judge = FakeJudge(['c, a'])
        line = {
            'question': 'Select all correct answers from A-D',
            'prediction': 'A and C',
            'answer': 'A, C',
        }

        result = self.logicvista.LogicVista_auxeval(judge, line)

        self.assertEqual(result['hit'], 1)
        self.assertEqual(result['res'], 'c, a')

    def test_retries_explanatory_output_and_logs_actual_response(self):
        judge = FakeJudge(['I choose 3', '3'])
        line = {
            'question': 'Which answer is correct? Select from 1-5',
            'prediction': '<answer>3</answer>',
            'answer': '3',
        }

        result = self.logicvista.LogicVista_auxeval(judge, line)

        self.assertEqual(result['hit'], 1)
        self.assertEqual(len(judge.calls), 2)
        self.assertIn('output is I choose 3, failed to parse.', result['log'])

    def test_valid_wrong_choice_is_scored_incorrect(self):
        judge = FakeJudge(['D'])
        line = {
            'question': 'Which answer is correct? Select from A-D',
            'prediction': '<answer>D</answer>',
            'answer': 'A',
        }

        result = self.logicvista.LogicVista_auxeval(judge, line)

        self.assertEqual(result, {'log': 'Succeed', 'res': 'D', 'hit': 0})
        self.assertEqual(len(judge.calls), 1)


if __name__ == '__main__':
    unittest.main()
