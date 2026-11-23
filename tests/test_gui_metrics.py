import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def load_gui_module(name):
    package = types.ModuleType('vlmeval')
    package.__path__ = ['vlmeval']
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.__path__ = ['vlmeval/dataset']
    base = types.ModuleType('vlmeval.dataset.image_base')
    base.ImageBaseDataset = type('ImageBaseDataset', (), {})
    smp = types.ModuleType('vlmeval.smp')
    for attr in ['LMUDataRoot', 'dump', 'get_intermediate_file_path', 'load', 'toliststr']:
        setattr(smp, attr, mock.Mock())
    smp.get_logger = lambda _name: mock.Mock()
    modules = {'vlmeval': package, 'vlmeval.dataset': dataset, 'vlmeval.dataset.image_base': base,
               'vlmeval.smp': smp}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            f'vlmeval.dataset.GUI.{name}', f'vlmeval/dataset/GUI/{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestGUIMetrics(unittest.TestCase):

    def evaluate(self, module_name, class_name, records):
        module = load_gui_module(module_name)
        dataset = getattr(module, class_name).__new__(getattr(module, class_name))
        dataset.img_root = '/unused'
        dataset.parse_response_func = module.parse_bbox_aguvis
        data = pd.DataFrame(records)
        image = mock.Mock(size=(100, 100))
        method = dataset.evaluate if module_name == 'venusbench' else dataset.evaluate_point
        with mock.patch.object(module, 'load', return_value=data), \
                mock.patch.object(module.Image, 'open', return_value=image), \
                mock.patch.dict('os.environ', {}, clear=True):
            return method('predictions.xlsx')

    def records(self, module_name):
        records = []
        groups = [('small', 'text', 1, True), ('large', 'text', 3, False),
                  ('small', 'icon', 2, True), ('large', 'icon', 1, False)]
        for category, ui_type, count, correct in groups:
            for _ in range(count):
                point = 0.25 if correct else 0.9
                if module_name == 'osworld_g':
                    point *= 1000
                records.append({'bbox': '[10, 10, 40, 40]', 'image_path': 'image.png', 'category': category,
                                'data_type': ui_type, 'ui_type': ui_type, 'question': 'Click the target',
                                'data_source': 'test', 'application': 'test',
                                'prediction': f'pyautogui.click(x={point}, y={point})'})
        return records

    def test_point_metrics_count_each_sample_once(self):
        for name, class_name in [('screenspot', 'ScreenSpot'), ('screenspot_pro', 'ScreenSpot_Pro'),
                                 ('osworld_g', 'OSWorld_G'), ('vbgd', 'VBGD')]:
            with self.subTest(dataset=name):
                scores = self.evaluate(name, class_name, self.records(name))
                self.assertAlmostEqual(scores['Overall_Accuracy'], 300 / 7)
                self.assertEqual(scores['Text_Accuracy'], 25)
                self.assertAlmostEqual(scores['Icon_Accuracy'], 200 / 3)
                self.assertEqual(scores['small_Accuracy'], 100)
                self.assertEqual(scores['large_Accuracy'], 0)

    def test_venusbench_category_metrics_are_scalar_sample_means(self):
        scores = self.evaluate('venusbench', 'VenusBench_GD', self.records('venusbench'))
        self.assertAlmostEqual(scores['Overall_Accuracy'], 300 / 7)
        self.assertEqual(scores['small_Accuracy'], 100)
        self.assertEqual(scores['large_Accuracy'], 0)


if __name__ == '__main__':
    unittest.main()
