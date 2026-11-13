import ast
import re
import unittest
from pathlib import Path


def load_extractor():
    source = Path(__file__).resolve().parents[1] / 'vlmeval' / 'dataset' / 'utils' / 'multiple_choice.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'extract_characters_regex')
    namespace = {'re': re}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['extract_characters_regex']


class TestMissingMultipleChoicePrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.extract = load_extractor()

    def test_empty_predictions_stay_unresolved(self):
        for prediction in ['', '  ', '\t\n', {}]:
            with self.subTest(prediction=prediction):
                self.assertEqual(type(self).extract(prediction), '')

    def test_existing_labels_and_custom_choice_text_remain_supported(self):
        self.assertEqual(type(self).extract('The answer is D.'), 'D')
        choices = ['(F)', '(G)']
        self.assertEqual(type(self).extract('F', choices), 'F')
        self.assertEqual(type(self).extract('The choice is (G).', choices), 'G')


if __name__ == '__main__':
    unittest.main()
