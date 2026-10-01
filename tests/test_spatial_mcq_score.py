import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def _load_matching_func():
    spec = importlib.util.spec_from_file_location(
        'matching_func',
        'vlmeval/dataset/utils/spatial_bench/matching_func.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_cal_scores():
    """Load ``cal_scores`` in isolation.

    Only ``compute_mcq_score`` is exercised here. It needs ``can_match_option``
    from ``matching_func``; the judge / LLM extraction imports are irrelevant to
    that function and are replaced with placeholders so the module can be
    imported without the rest of the package's heavy dependencies.
    """
    matching = _load_matching_func()

    pkg = 'vlmeval.dataset.utils.spatial_bench'
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    utils = types.ModuleType('vlmeval.dataset.utils')
    utils.__path__ = ['vlmeval/dataset/utils']
    spatial = types.ModuleType(pkg)
    spatial.__path__ = ['vlmeval/dataset/utils/spatial_bench']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.get_intermediate_file_path = lambda *a, **k: ''
    smp_vlm = types.ModuleType('vlmeval.smp.vlm')
    smp_vlm.gpt_key_set = lambda: False
    judge_util = types.ModuleType('vlmeval.dataset.utils.judge_util')
    judge_util.build_judge = lambda *a, **k: None
    llm_extract = types.ModuleType(pkg + '.llm_extract')
    llm_extract.parallel_llm_extract = lambda *a, **k: None
    matching_mod = types.ModuleType(pkg + '.matching_func')
    matching_mod.can_match_option = matching.can_match_option
    matching_mod.can_match_na = matching.can_match_na
    tools = types.ModuleType(pkg + '.tools')
    tools.__path__ = []
    tools_files = types.ModuleType(pkg + '.tools.files')
    tools_files.build_eval_paths = lambda *a, **k: (None, None)
    tools_files.get_judge_tag_from_score_fn = lambda *a, **k: None

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.utils': utils,
        'vlmeval.dataset.utils.spatial_bench': spatial,
        'vlmeval.smp': smp,
        'vlmeval.smp.file': smp_file,
        'vlmeval.smp.vlm': smp_vlm,
        'vlmeval.dataset.utils.judge_util': judge_util,
        pkg + '.llm_extract': llm_extract,
        pkg + '.matching_func': matching_mod,
        pkg + '.tools': tools,
        pkg + '.tools.files': tools_files,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            pkg + '.cal_scores',
            'vlmeval/dataset/utils/spatial_bench/cal_scores.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestComputeMcqScore(unittest.TestCase):

    def setUp(self):
        self.cal_scores = _load_cal_scores()

    def test_failed_prediction_is_not_a_hit(self):
        # ``can_match_option`` returns ``False`` when the prediction cannot be
        # reduced to an option letter. Stringifying ``False`` on both sides
        # would make ``"false" == "false"`` and reward the sample.
        df = pd.DataFrame({
            'prediction': ['', 'nonsense'],
            'answer': ['B', 'B'],
        })
        out = self.cal_scores.compute_mcq_score(df)
        self.assertEqual(out['pred_extracted'].tolist(), [False, False])
        self.assertEqual(out['hit'].tolist(), [0., 0.])

    def test_failed_ground_truth_is_not_a_hit(self):
        # An unparseable ground truth can never be answered correctly, even if
        # the prediction is also unparseable.
        df = pd.DataFrame({
            'prediction': ['', 'A'],
            'answer': ['not a choice', 'not a choice'],
        })
        out = self.cal_scores.compute_mcq_score(df)
        self.assertEqual(out['hit'].tolist(), [0., 0.])

    def test_correct_answer_still_scores(self):
        df = pd.DataFrame({
            'prediction': ['the answer is (B)', 'A'],
            'answer': ['B', 'not a choice'],
        })
        out = self.cal_scores.compute_mcq_score(df)
        self.assertEqual(out['pred_extracted'].tolist(), ['B', 'A'])
        self.assertEqual(out['hit'].tolist(), [1., 0.])


if __name__ == '__main__':
    unittest.main()
