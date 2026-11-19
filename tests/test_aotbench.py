import os.path as osp
import tempfile
import unittest

import pandas as pd

from vlmeval.dataset import AoTBench
from vlmeval.dataset.video_dataset_config import PREDEFINED_DATASET_SPECS


class TestAoTBench(unittest.TestCase):

    def evaluate(self, rows):
        with tempfile.TemporaryDirectory() as tmp_dir:
            eval_file = osp.join(tmp_dir, 'model_AoTBench_QA.xlsx')
            pd.DataFrame(rows).to_excel(eval_file, index=False)
            return AoTBench.evaluate(eval_file)

    def test_first_character_must_match_answer(self):
        predictions = {
            'A': 1,
            'A. The first segment': 1,
            '(A)': 0,
            ' A': 0,
            'a': 0,
            'The answer is A': 0,
            None: 0,
        }
        for prediction, expected in predictions.items():
            with self.subTest(prediction=prediction):
                rating = self.evaluate([{'index': 0, 'prediction': prediction, 'answer': 'A'}])
                self.assertEqual(rating['correct'], expected)

    def test_reports_each_question_type(self):
        rating = self.evaluate([
            {'index': 0, 'prediction': 'A', 'answer': 'A', 'question_type': 'nextqa'},
            {'index': 1, 'prediction': 'B', 'answer': 'A', 'question_type': 'nextqa'},
            {'index': 2, 'prediction': 'C', 'answer': 'C', 'question_type': 'TVBench'},
        ])
        self.assertAlmostEqual(rating['overall'], 200 / 3)
        self.assertEqual(rating['nextqa'], 50)
        self.assertEqual(rating['TVBench'], 100)

    def test_sample_without_video_track_counts_as_wrong(self):
        rating = self.evaluate([
            {'index': 0, 'prediction': 'A', 'answer': 'A', 'has_video': True},
            {'index': 1, 'prediction': 'A', 'answer': 'A', 'has_video': False},
        ])
        self.assertEqual(rating['correct'], 1)
        self.assertEqual(rating['total'], 2)

    def test_each_subset_has_a_16_frame_alias(self):
        for name in AoTBench.supported_datasets():
            with self.subTest(name=name):
                spec = PREDEFINED_DATASET_SPECS[f'{name}_16frame']
                self.assertEqual(spec.dataset_name, name)
                self.assertEqual(spec.build_config['nframe'], 16)


if __name__ == '__main__':
    unittest.main()
