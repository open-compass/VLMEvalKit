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


class TestGUIInvalidClick(unittest.TestCase):

    def test_malformed_clicks_are_not_replaced_with_the_origin(self):
        for name in ['screenspot', 'screenspot_pro', 'osworld_g', 'vbgd', 'venusbench', 'screenspot_v2']:
            module = load_gui_module(name)
            for response in ['', 'Failed to obtain answer via API.', 'pyautogui.click(x=0)']:
                with self.subTest(dataset=name, response=response):
                    with self.assertRaises(ValueError):
                        module.parse_bbox_aguvis(response)
            self.assertEqual(module.parse_bbox_aguvis('pyautogui.click(x=0, y=0)'), [0.0, 0.0])

    def test_screenspot_marks_unparseable_origin_targets_as_format_errors(self):
        module = load_gui_module('screenspot')
        dataset = module.ScreenSpot.__new__(module.ScreenSpot)
        dataset.img_root = '/unused'
        data = pd.DataFrame({'bbox': ['[0, 0, 20, 20]'] * 4, 'prediction': ['', 'pyautogui.click(x=0, y=0)'] * 2,
                             'image_path': ['test.png'] * 4, 'question': ['Click target'] * 4,
                             'data_type': ['text'] * 2 + ['icon'] * 2, 'data_source': ['test'] * 4})
        with mock.patch.object(module, 'load', return_value=data), \
                mock.patch.object(module.Image, 'open', return_value=mock.Mock(size=(100, 100))), \
                mock.patch.dict('os.environ', {}, clear=True):
            score = dataset.evaluate_point('predictions.xlsx')
        self.assertEqual(score['Overall_Accuracy'], 50)
        self.assertEqual(score['Format_Err_Rate'], 50)


if __name__ == '__main__':
    unittest.main()
