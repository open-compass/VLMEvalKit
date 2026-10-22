import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def load_matching():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('vlmeval')
    package.__path__ = [str(root / 'vlmeval')]
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = [str(root / 'vlmeval/smp')]
    with mock.patch.dict(sys.modules, {'vlmeval': package, 'vlmeval.smp': smp}):
        spec = importlib.util.spec_from_file_location('matching_under_test', root / 'vlmeval/utils/matching_util.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestMissingPredictions(unittest.TestCase):

    def test_missing_prediction_from_tsv_does_not_become_a_letter_answer(self):
        matching = load_matching()
        row = pd.read_csv(io.StringIO('prediction\tanswer\n\tD\n'), sep='\t').iloc[0]
        self.assertTrue(pd.isna(row['prediction']))
        for prediction in [row['prediction'], None, pd.NA]:
            with self.subTest(prediction=prediction):
                self.assertFalse(matching.can_infer(prediction, {'A': 'D', 'B': 'B', 'C': 'C', 'D': 'A'}))
                self.assertFalse(matching.can_infer_lego(prediction, 'multiple-choice', {'A': 'None', 'B': 'nan'}))

    def test_short_option_text_does_not_match_inside_words(self):
        matching = load_matching()
        for answer in ['I cannot answer this question.', 'The animal is a caterpillar.', 'I do not know.']:
            with self.subTest(answer=answer):
                choices = {'A': 'cat', 'B': 'No', 'C': 'Cannot be determined', 'D': 'Not enough information'}
                self.assertFalse(matching.can_infer(answer, choices))

    def test_valid_labels_text_and_non_latin_text_are_preserved(self):
        matching = load_matching()
        for answer, choices, expected in [
            ('B', {'A': 'Yes', 'B': 'No'}, 'B'),
            ('No', {'A': 'Yes', 'B': 'No'}, 'B'),
            ('cat.', {'A': 'cat', 'B': 'dog'}, 'A'),
            ('猫。', {'A': '猫', 'B': '狗'}, 'A'),
            ('nan', {'A': 'nan', 'B': 'value'}, 'A'),
            ('42', {'A': '42', 'B': '43'}, 'A'),
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(matching.can_infer(answer, choices), expected)


if __name__ == '__main__':
    unittest.main()
