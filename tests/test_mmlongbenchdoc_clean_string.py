import importlib.util
import sys
import types
import unittest
from unittest import mock


def _load_mmlongbenchdoc():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    utils = types.ModuleType('vlmeval.dataset.utils')
    utils.__path__ = ['vlmeval/dataset/utils']
    utils.build_judge = mock.MagicMock()
    utils.levenshtein_distance = mock.MagicMock()
    judge_cache = types.ModuleType('vlmeval.dataset.utils.judge_cache')
    for name in [
            'get_judge_cache_file', 'get_judge_detail_file', 'get_judge_named_legacy_cache_file',
            'get_judge_score_file', 'load_judge_cache', 'run_cached_tasks'
    ]:
        setattr(judge_cache, name, mock.MagicMock())
    smp = types.ModuleType('vlmeval.smp')
    for name in [
            'decode_base64_to_image_file', 'dump', 'encode_image_to_base64', 'listinstr', 'load', 'read_ok', 'toliststr'
    ]:
        setattr(smp, name, mock.MagicMock())
    smp.get_logger = lambda name: mock.MagicMock()
    image_base = types.ModuleType('vlmeval.dataset.image_base')
    image_base.ImageBaseDataset = type('ImageBaseDataset', (), {})
    torchvision = types.ModuleType('torchvision')
    torchvision.__path__ = []
    transforms = types.ModuleType('torchvision.transforms')
    torchvision.transforms = transforms
    pil = types.ModuleType('PIL')
    for name in ['Image', 'ImageDraw', 'ImageFont']:
        setattr(pil, name, mock.MagicMock())

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.utils': utils,
        'vlmeval.dataset.utils.judge_cache': judge_cache,
        'vlmeval.dataset.image_base': image_base,
        'vlmeval.smp': smp,
        'torchvision': torchvision,
        'torchvision.transforms': transforms,
        'PIL': pil,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.dataset.mmlongbenchdoc',
            'vlmeval/dataset/mmlongbenchdoc.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class MMLongBenchDocCleanStringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = _load_mmlongbenchdoc()

    def test_unit_suffixes_are_removed(self):
        get_clean_string = self.module.get_clean_string
        self.assertEqual(get_clean_string('12 miles'), '12')
        self.assertEqual(get_clean_string('1 mile'), '1')
        self.assertEqual(get_clean_string('3.5 Million'), '3.5')

    def test_other_strings_are_unchanged(self):
        get_clean_string = self.module.get_clean_string
        self.assertEqual(get_clean_string('Smile'), 'smile')
        self.assertEqual(get_clean_string('Two miles'), 'two miles')
        self.assertEqual(get_clean_string('$4.2'), '4.2')
        self.assertEqual(get_clean_string('25%'), '25')
        self.assertEqual(get_clean_string('Paris (France)'), 'paris')

    def test_float_answers_with_units_match(self):
        eval_score = self.module.eval_score
        self.assertEqual(eval_score('12 miles', '12', 'Float'), 1.0)
        self.assertEqual(eval_score('12', '12 miles', 'Float'), 1.0)
        self.assertEqual(eval_score('2.5 million', '2.5', 'Float'), 1.0)
        self.assertEqual(eval_score('12 miles', '13', 'Float'), 0.0)


if __name__ == '__main__':
    unittest.main()
