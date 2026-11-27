import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def load_matching_module():
    # Load the real leaf module and logger without importing optional model backends.
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('vlmeval')
    package.__path__ = [str(root / 'vlmeval')]
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = [str(root / 'vlmeval' / 'smp')]
    with mock.patch.dict(sys.modules, {'vlmeval': package, 'vlmeval.smp': smp}):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.utils.matching_util', root / 'vlmeval' / 'utils' / 'matching_util.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


CHOICES = {'A': '$171,900', 'B': '$7', 'C': '$8', 'D': '$77,490'}


class TestRuleBasedMCQ(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.m = load_matching_module()

    def test_answer_statements_from_mmmu_smoke(self):
        # Tails of real MMMU_DEV_VAL responses whose full text can_infer could not read.
        cases = [
            ('Gains = 39,690 + 37,800 = **77,490**\n\n---\n\n### ✅ Final Answer: **D. $77,490**', 'D'),
            ('Project B has a positive NPV.\n\n---\n\n### **Answer:**\n**B. Project B**', 'B'),
            ('= $6 + $2 = $8\n\n### Final Answer:\n**C. $8**', 'C'),
            ('so we go with **B**.\n\n---\n\n✅ Final Answer: **B**', 'B'),
        ]
        for response, want in cases:
            self.assertEqual(self.m.rule_based_mcq_answer(response, CHOICES), (want, 'answer statement'))

    def test_last_answer_statement_wins(self):
        response = 'At first the answer is (A). Wait, that is wrong.\nThe correct answer is (C).'
        self.assertEqual(self.m.parse_answer_statement(response, CHOICES), 'C')

    def test_articles_and_words_are_not_options(self):
        for response in ['The answer is a horse.', 'Answer: A company records revenue.',
                         'Answer: I think so.', 'The answer is To be determined.']:
            self.assertIsNone(self.m.parse_answer_statement(response, CHOICES), response)

    def test_option_outside_choices_is_ignored(self):
        self.assertIsNone(self.m.parse_answer_statement('Final answer: E', CHOICES))

    def test_mmmu_parser_brackets_and_content(self):
        self.assertEqual(self.m.parse_mmmu_choice('I pick (B) over (A), finally (D)', CHOICES), 'D')
        self.assertEqual(
            self.m.rule_based_mcq_answer('So the ending balance comes to $171,900 in total here', CHOICES),
            ('A', 'MMMU parser'))

    def test_no_match_is_none_not_random(self):
        self.assertEqual(self.m.rule_based_mcq_answer('I cannot tell from this image.', CHOICES), (None, None))


if __name__ == '__main__':
    unittest.main()
