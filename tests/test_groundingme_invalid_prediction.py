import importlib.util
import sys
import types
from pathlib import Path
from unittest import TestCase, mock

import pandas as pd


def load_groundingme(raw_data, predictions):
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('vlmeval')
    package.__path__ = [str(root / 'vlmeval')]
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = [str(root / 'vlmeval/dataset')]
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = [str(root / 'vlmeval/smp')]
    smp.load = lambda _: predictions
    dumped = {}
    smp.dump = lambda value, path: dumped.setdefault(path, value)
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.get_intermediate_file_path = lambda path, suffix, extension: f'{path}{suffix}.{extension}'

    image_base = types.ModuleType('vlmeval.dataset.image_base')

    class ImageBaseDataset:
        def __init__(self, _):
            self.data = raw_data

    image_base.ImageBaseDataset = ImageBaseDataset
    modules = {
        'vlmeval': package,
        'vlmeval.dataset': dataset,
        'vlmeval.smp': smp,
        'vlmeval.smp.file': smp_file,
        'vlmeval.dataset.image_base': image_base,
    }
    path = root / 'vlmeval/dataset/groundingme.py'
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.groundingme', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, dumped


class TestGroundingMEInvalidPrediction(TestCase):

    def evaluate_rejection(self, prediction):
        raw_data = pd.DataFrame([{
            'answer': '[0, 0, 0, 0]',
            'width': 100,
            'height': 100,
            'subtask_l1': 'Rejection',
            'subtask_l2': '',
        }])
        predictions = pd.DataFrame([{'index': 0, 'prediction': prediction}])
        module, dumped = load_groundingme(raw_data, predictions)
        scores = module.GroundingME.evaluate('predictions.tsv')
        return scores.iloc[0]['ACC@0.5'], dumped

    def test_unparseable_response_does_not_score_as_a_correct_rejection(self):
        score, _ = self.evaluate_rejection('I could not identify the object.')
        self.assertEqual(score, 0)

    def test_explicit_null_bbox_scores_as_a_correct_rejection(self):
        score, _ = self.evaluate_rejection('{"bbox_2d": null}')
        self.assertEqual(score, 1)


if __name__ == '__main__':
    import unittest

    unittest.main()
