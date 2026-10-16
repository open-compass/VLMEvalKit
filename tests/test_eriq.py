import unittest

import pandas as pd

from vlmeval.dataset.eriq import _build_eriq_score_fn, _extract_yes_no
from vlmeval.dataset.utils.spatial_bench.cal_scores import build_mcq_score_fn


class TestERIQScoring(unittest.TestCase):

    def test_extract_yes_no(self):
        cases = {
            'Yes': 'yes',
            'No, the robot did not make a mistake.': 'no',
            'The final answer is YES.': 'yes',
            '<think>It may be no.</think><answer>yes</answer>': 'yes',
            'Please answer yes or no.': None,
            'Cannot determine.': None,
        }
        for response, expected in cases.items():
            with self.subTest(response=response):
                self.assertEqual(_extract_yes_no(response), expected)

    def test_mistake_existence_is_scored_as_yes_no(self):
        data = pd.DataFrame([
            {
                'index': 0,
                'task_type': 'QA_MISTAKE_EXISTENCE',
                'prediction': 'Yes, the robot made a mistake.',
                'answer': 'Yes',
            },
            {
                'index': 1,
                'task_type': 'QA_MISTAKE_EXISTENCE',
                'prediction': 'The final answer is no.',
                'answer': 'No',
            },
            {
                'index': 2,
                'task_type': 'QA_MISTAKE_EXISTENCE',
                'prediction': 'No',
                'answer': 'Yes',
            },
            {
                'index': 3,
                'task_type': 'QA_MISTAKE_EXISTENCE',
                'prediction': 'Cannot determine.',
                'answer': 'No',
            },
            {
                'index': 4,
                'task_type': 'QA_ACTION_UNDERSTANDING',
                'prediction': 'The answer is B.',
                'answer': 'B',
            },
        ])
        score_fn = _build_eriq_score_fn(build_mcq_score_fn())

        scored = score_fn(data)

        self.assertEqual(scored['hit'].tolist(), [1., 1., 0., 0., 1.])
        self.assertEqual(scored['pred_extracted'].tolist()[:3], ['yes', 'no', 'no'])
        self.assertTrue(pd.isna(scored.loc[3, 'pred_extracted']))
        self.assertEqual(scored.loc[4, 'pred_extracted'], 'B')
        self.assertEqual(scored['prediction'].tolist(), data['prediction'].tolist())
        self.assertEqual(scored['answer'].tolist(), data['answer'].tolist())

    def test_wrapper_preserves_judge_metadata(self):
        base_score_fn = build_mcq_score_fn()
        score_fn = _build_eriq_score_fn(base_score_fn)

        self.assertEqual(score_fn.judge_mode, base_score_fn.judge_mode)
        self.assertEqual(score_fn.judge_model, base_score_fn.judge_model)

    def test_llm_judge_path_is_unchanged(self):
        data = pd.DataFrame([{
            'index': 0,
            'task_type': 'QA_MISTAKE_EXISTENCE',
            'prediction': 'Yes',
            'answer': 'Yes',
        }])
        delegated = []

        def llm_score_fn(received):
            delegated.append(received.copy())
            scored = received.copy()
            scored['judge_grade'] = 'A'
            scored['pred_extracted'] = 'Yes'
            scored['hit'] = 1
            return scored

        llm_score_fn.judge_mode = 'llm'
        llm_score_fn.judge_model = 'test-judge'

        scored = _build_eriq_score_fn(llm_score_fn)(data)

        self.assertEqual(len(delegated), 1)
        pd.testing.assert_frame_equal(delegated[0], data)
        self.assertEqual(scored.loc[0, 'judge_grade'], 'A')
        self.assertEqual(scored.loc[0, 'hit'], 1)


if __name__ == '__main__':
    unittest.main()
