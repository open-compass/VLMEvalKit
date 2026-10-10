import importlib.util
import sys
import types
import unittest
from unittest import mock


def load_dynamath():
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
        spec = importlib.util.spec_from_file_location('vlmeval.dataset.dynamath', 'vlmeval/dataset/dynamath.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestDynaMathJSONNumbers(unittest.TestCase):

    def test_numeric_short_answer_does_not_call_extraction_judge(self):
        module = load_dynamath()
        model = mock.Mock()
        model.generate.side_effect = AssertionError('A valid JSON number should not require a judge')
        for answer in [3, 3.5, -1.25, 0]:
            with self.subTest(answer=answer):
                line = {'prediction': '{"short answer": ' + str(answer) + '}',
                        'answer_type': 'float', 'answer': str(answer)}
                result = module.DynaMath_auxeval(model, line)
                self.assertTrue(result['parse'])
                self.assertTrue(result['correct'])
                self.assertEqual(result['extracted'], float(answer))
        model.generate.assert_not_called()

    def test_existing_string_and_pi_answers_still_parse(self):
        module = load_dynamath()
        self.assertEqual(module.parse_answer('3.5', 'float'), (True, 3.5))
        self.assertEqual(module.parse_answer(float('nan'), 'float'), (False, None))
        self.assertEqual(module.parse_answer(float('inf'), 'float'), (False, None))
        self.assertEqual(module.parse_answer('A', 'multiple choice'), (True, 'A'))
        self.assertEqual(module.parse_answer('not a number', 'float'), (False, None))


if __name__ == '__main__':
    unittest.main()
