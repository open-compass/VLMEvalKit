"""Offline regression for MathVerse score aggregation."""
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_mathverse():
    # Load the leaf module without importing optional model backends.
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.load = pd.read_csv
    modules = {
        'vlmeval': types.ModuleType('vlmeval'),
        'vlmeval.smp': types.ModuleType('vlmeval.smp'),
        'vlmeval.smp.file': smp_file,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval_mathverse_under_test', ROOT / 'vlmeval/dataset/utils/mathverse.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestMathVerseAcc(unittest.TestCase):

    def test_reports_overall_accuracy_across_problem_versions(self):
        meta = "{'split': 'testmini', 'source': 'GeoQA', 'subject': 'Plane Geometry', 'subfield': 'Angle'}"
        data = pd.DataFrame({
            'index': [1, 2, 3, 4],
            'problem_version': ['Text Dominant', 'Text Dominant', 'Vision Only', 'Vision Only'],
            'metadata': [meta] * 4,
            'score': [True, True, True, False],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'score.csv'
            data.to_csv(path, index=False)
            res = load_mathverse().MathVerse_acc(str(path)).set_index('split')
        self.assertEqual(res.loc['Text Dominant', 'Overall'], 100.0)
        self.assertEqual(res.loc['Vision Only', 'Overall'], 50.0)
        self.assertEqual(res.loc['Overall', 'Overall'], 75.0)


if __name__ == '__main__':
    unittest.main()
