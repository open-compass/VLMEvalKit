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


class TestVerboseOptionBoundaries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matching = load_matching_module()

    def choices(self):
        return {'A': 'cat', 'B': 'fish', 'C': 'horse', 'D': 'rabbit'}

    def test_option_text_reaches_text_matching(self):
        for answer in ['The answer is cat.', 'The correct answer is **cat**.', 'The answer is Cat.']:
            with self.subTest(answer=answer):
                self.assertIs(self.matching.can_infer_option(answer, self.choices()), False)
                self.assertEqual(self.matching.can_infer(answer, self.choices()), 'A')

    def test_unrecognized_words_do_not_become_options(self):
        for word in ['dog', 'bird', 'B2', 'b2', 'Café', 'café']:
            answer = f'The answer is {word}.'
            with self.subTest(answer=answer):
                self.assertIs(self.matching.can_infer_option(answer, self.choices()), False)
                self.assertIs(self.matching.can_infer(answer, self.choices()), False)

    def test_verbose_labels_remain_available_when_other_options_are_mentioned(self):
        for prefix in ['The correct answer is', 'THE ANSWER IS', 'Answer is']:
            for option in 'ABCDabcd':
                answer = f'{prefix} **{option}**. Considering A, B, C and D, that is my pick.'
                with self.subTest(answer=answer):
                    self.assertEqual(self.matching.can_infer_option(answer, self.choices()), option.upper())

    def test_plain_lowercase_option_labels_are_available(self):
        for option in 'abcd':
            for answer in [f'The answer is {option}.', f'The correct answer is **{option}**.', f'Answer is {option}']:
                with self.subTest(answer=answer):
                    self.assertEqual(self.matching.can_infer_option(answer, self.choices()), option.upper())
                    self.assertEqual(self.matching.can_infer(answer, self.choices()), option.upper())

    def test_plain_uppercase_option_labels_are_unchanged(self):
        for answer, expected in [('B', 'B'), ('The answer is B.', 'B'), ('Answer is D', 'D'),
                                 ('The answer is C: horse', 'C'),
                                 ('The answer is A because it looks like a horse.', 'A')]:
            with self.subTest(answer=answer):
                self.assertEqual(self.matching.can_infer_option(answer, self.choices()), expected)

    def test_lowercase_article_a_is_not_option_a(self):
        # "a <word>" is the English article. Option text falls through to can_infer_text;
        # a word that is not an option stays unresolved.
        for answer, expected in [('The answer is a horse.', 'C'), ('The correct answer is a rabbit.', 'D')]:
            with self.subTest(answer=answer):
                self.assertIs(self.matching.can_infer_option(answer, self.choices()), False)
                self.assertEqual(self.matching.can_infer(answer, self.choices()), expected)

        for answer in ['The answer is a dog.', 'The answer is a bit unclear, but it looks like a horse.']:
            with self.subTest(answer=answer):
                self.assertIs(self.matching.can_infer_option(answer, self.choices()), False)
                self.assertIs(self.matching.can_infer(answer, self.choices()), False)


if __name__ == '__main__':
    unittest.main()
