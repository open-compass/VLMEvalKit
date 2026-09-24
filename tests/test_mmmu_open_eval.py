import importlib.util
import logging
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


def _load_pickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def _dump_pickle(value, path):
    with open(path, 'wb') as file:
        pickle.dump(value, file)


def _load_multiple_choice():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    utils_package = types.ModuleType('vlmeval.dataset.utils')
    utils_package.__path__ = ['vlmeval/dataset/utils']

    smp = types.ModuleType('vlmeval.smp')
    smp.cn_string = lambda value: False
    smp.d2df = lambda value: value
    smp.dump = _dump_pickle
    smp.get_logger = logging.getLogger
    smp.get_pred_file_format = lambda: 'pkl'
    smp.istype = lambda value, expected: isinstance(value, expected)
    smp.load = _load_pickle
    smp.timestr = lambda: 'test'

    smp_vlm = types.ModuleType('vlmeval.smp.vlm')
    smp_vlm.build_option_str = lambda choices: str(choices)

    vlmeval_utils = types.ModuleType('vlmeval.utils')
    vlmeval_utils.can_infer = lambda prediction, choices: prediction if prediction in choices else False
    vlmeval_utils.can_infer_lego = vlmeval_utils.can_infer

    def track_progress(function, tasks, save, keys, **kwargs):
        del kwargs
        results = [function(**task) for task in tasks]
        _dump_pickle(dict(zip(keys, results)), save)
        return results

    vlmeval_utils.track_progress_rich = track_progress

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.utils': utils_package,
        'vlmeval.smp': smp,
        'vlmeval.smp.vlm': smp_vlm,
        'vlmeval.utils': vlmeval_utils,
    }
    with mock.patch.dict(sys.modules, modules):
        mmmu_spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.utils.mmmu',
            'vlmeval/dataset/utils/mmmu.py',
        )
        mmmu = importlib.util.module_from_spec(mmmu_spec)
        sys.modules['vlmeval.dataset.utils.mmmu'] = mmmu
        mmmu_spec.loader.exec_module(mmmu)

        mcq_spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.utils.multiple_choice',
            'vlmeval/dataset/utils/multiple_choice.py',
        )
        multiple_choice = importlib.util.module_from_spec(mcq_spec)
        mcq_spec.loader.exec_module(multiple_choice)
        return mmmu, multiple_choice


class TestMMMUOpenEvaluation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mmmu, cls.multiple_choice = _load_multiple_choice()

    def test_open_match_handles_numeric_and_text_answers(self):
        numeric = self.mmmu.parse_open_response('Therefore, the answer is 1,250.0')
        text = self.mmmu.parse_open_response('The final answer is mitosis.')

        self.assertTrue(self.mmmu.eval_open('1250', numeric))
        self.assertTrue(self.mmmu.eval_open('mitosis', text))
        self.assertFalse(self.mmmu.eval_open('meiosis', text))

    def test_one_character_answer_does_not_match_inside_word(self):
        prediction = self.mmmu.parse_open_response('The area is large')

        self.assertFalse(self.mmmu.eval_open('a', prediction))

    def test_mixed_mmmu_uses_content_for_open_and_choice_for_mcq(self):
        meta = pd.DataFrame({
            'index': [1, 2],
            'answer': ['42', 'B'],
            'question_type': ['open', 'multiple-choice'],
            'A': [float('nan'), 'wrong'],
            'B': [float('nan'), 'right'],
        })
        data = pd.DataFrame({
            'index': [1, 2],
            'question': ['What is six times seven?', 'Pick the correct option'],
            'prediction': ['Work shown. Answer: 42', 'B'],
            'A': [float('nan'), 'wrong'],
            'B': [float('nan'), 'right'],
        })

        with tempfile.TemporaryDirectory() as directory:
            result_file = str(Path(directory) / 'results.pkl')
            evaluated = self.multiple_choice.mcq_vanilla_eval(
                None, data, meta, 1, result_file, 'MMMU_DEV_VAL')

        self.assertEqual(list(evaluated['hit']), [1, 1])
        self.assertTrue(pd.isna(evaluated.iloc[0]['A']))
        self.assertIn('MMMU open-ended match', evaluated.iloc[0]['log'])

    def test_open_result_overrides_stale_pseudo_mcq_cache(self):
        meta = pd.DataFrame({
            'index': [1],
            'answer': ['42'],
            'question_type': ['open'],
            'A': [float('nan')],
        })
        data = pd.DataFrame({
            'index': [1],
            'question': ['What is six times seven?'],
            'prediction': ['42'],
            'A': [float('nan')],
        })

        with tempfile.TemporaryDirectory() as directory:
            result_file = str(Path(directory) / 'results.pkl')
            _dump_pickle({1: {'hit': 0, 'log': 'stale pseudo-MCQ result'}}, result_file)
            evaluated = self.multiple_choice.mcq_vanilla_eval(
                None, data, meta, 1, result_file, 'MMMU_DEV_VAL')

        self.assertEqual(evaluated.iloc[0]['hit'], 1)
        self.assertNotIn('stale', evaluated.iloc[0]['log'])


if __name__ == '__main__':
    unittest.main()
