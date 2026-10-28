"""Offline regression for MME-Reasoning score aggregation."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_mme_reasoning():
    # Load the leaf module without importing optional model backends.
    smp = types.ModuleType('vlmeval.smp')
    smp.load = pd.read_csv
    with mock.patch.dict(sys.modules, {'vlmeval': types.ModuleType('vlmeval'), 'vlmeval.smp': smp}):
        spec = importlib.util.spec_from_file_location(
            'vlmeval_mme_reasoning_under_test', ROOT / 'vlmeval/dataset/utils/mme_reasoning.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestMMEReasoningAcc(unittest.TestCase):

    def test_acc_aggregates_scores(self):
        data = pd.DataFrame({
            'index': [0, 1, 2, 3],
            'capability': ['calculation', 'calculation, pattern analysis', 'spatial-temporal', 'pattern analysis'],
            'reasoning_type': ['deductive', 'inductive', 'abductive', 'deductive'],
            'score': [True, False, True, True],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.csv'
            data.to_csv(path, index=False)
            res = load_mme_reasoning().MMEReasoning_acc(str(path))
        self.assertEqual(res['Overall'][0], 75.0)
        self.assertEqual(res['calculation'][0], 50.0)
        self.assertEqual(res['pattern analysis'][0], 50.0)
        self.assertEqual(res['deductive'][0], 100.0)


if __name__ == '__main__':
    unittest.main()
