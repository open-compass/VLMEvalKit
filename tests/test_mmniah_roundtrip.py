import importlib.util
import io
import sys
import types
import unittest
from unittest import mock

import pandas as pd
from tqdm import tqdm


def load_mmniah():
    tqdm.get_lock()
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    base = types.ModuleType('vlmeval.dataset.image_base')
    base.ImageBaseDataset = type('ImageBaseDataset', (), {})
    smp = types.ModuleType('vlmeval.smp')
    smp.__getattr__ = lambda name: mock.Mock()
    utils = types.ModuleType('vlmeval.utils')
    utils.track_progress_rich = mock.Mock()
    dataset_utils = types.ModuleType('vlmeval.dataset.utils')
    dataset_utils.__path__ = ['vlmeval/dataset/utils']
    dataset_utils.DEBUG_MESSAGE = ''
    dataset_utils.build_judge = mock.Mock()
    cache = types.ModuleType('vlmeval.dataset.utils.judge_cache')
    cache.__getattr__ = lambda name: mock.Mock()
    vqa = types.ModuleType('vlmeval.dataset.utils.vqa_eval')
    vqa.istype = mock.Mock()
    modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.dataset.image_base': base,
               'vlmeval.smp': smp, 'vlmeval.utils': utils, 'vlmeval.dataset.utils': dataset_utils,
               'vlmeval.dataset.utils.judge_cache': cache, 'vlmeval.dataset.utils.vqa_eval': vqa}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.image_vqa', 'vlmeval/dataset/image_vqa.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        helper = importlib.import_module('vlmeval.dataset.utils.mmniah')
    return module, helper


class TestMMNIAHAnswerListRoundtrip(unittest.TestCase):

    def test_count_image_list_matching_survives_tsv_roundtrip(self):
        module, helper = load_mmniah()
        native = pd.DataFrame({'category': ['count-image'] * 3, 'answer': [[1, 2]] * 3,
                               'prediction': ['[1, 2]', '[1, 3]', '[3, 4]']})
        restored = pd.read_csv(io.StringIO(native.to_csv(sep='\t', index=False)), sep='\t')
        results = []
        for data in [native, restored]:
            with mock.patch.object(module, 'load', return_value=data), \
                    mock.patch.dict(sys.modules, {'vlmeval.dataset.utils.mmniah': helper}):
                results.append(module.MMNIAH.evaluate('predictions.tsv'))
        self.assertEqual(results[0]['count-image'], 0.5)
        self.assertEqual(results[1]['count-image'], results[0]['count-image'])
        self.assertEqual(results[1]['total'], results[0]['total'])

    def test_count_text_list_matching_survives_tsv_roundtrip(self):
        module, helper = load_mmniah()
        native = pd.DataFrame({'category': ['count-text'] * 3, 'answer': [[1, 2]] * 3,
                               'prediction': ['[1, 2]', '[1, 3]', '[3, 4]']})
        restored = pd.read_csv(io.StringIO(native.to_csv(sep='\t', index=False)), sep='\t')
        results = []
        for data in [native, restored]:
            with mock.patch.object(module, 'load', return_value=data), \
                    mock.patch.dict(sys.modules, {'vlmeval.dataset.utils.mmniah': helper}):
                results.append(module.MMNIAH.evaluate('predictions.tsv'))
        self.assertEqual(results[0]['count-text'], 0.5)
        self.assertEqual(results[1]['count-text'], results[0]['count-text'])
        self.assertEqual(results[1]['total'], results[0]['total'])

    def test_non_counting_answer_categories_preserve_existing_behavior(self):
        module, helper = load_mmniah()
        data = pd.DataFrame({'category': ['find-image', 'find-text'], 'answer': ['0', 'target'],
                             'prediction': ['A', 'target']})
        with mock.patch.object(module, 'load', return_value=data), \
                mock.patch.dict(sys.modules, {'vlmeval.dataset.utils.mmniah': helper}):
            result = module.MMNIAH.evaluate('predictions.tsv')
        self.assertEqual(result['total'], 1.0)


if __name__ == '__main__':
    unittest.main()
