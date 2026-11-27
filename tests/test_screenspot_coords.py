import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def norm_xywh(bbox, image_size):
    """Inclusive xywh box in the same [0, 1] xyxy space ScreenSpot scores in."""
    x, y, w, h = bbox
    width, height = image_size
    return [x / width, y / height, (x + w - 1) / width, (y + h - 1) / height]


def load_click_coords():
    spec = importlib.util.spec_from_file_location('screenspot_click_coords',
                                                  'vlmeval/dataset/GUI/click_coords.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    modules = {
        'vlmeval': package,
        'vlmeval.dataset': dataset,
        'vlmeval.dataset.image_base': base,
        'vlmeval.smp': smp,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(f'vlmeval.dataset.GUI.{name}',
                                                      f'vlmeval/dataset/GUI/{name}.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class TestClickScale(unittest.TestCase):

    def setUp(self):
        self.coords = load_click_coords()

    def test_fractions_stay_fractions_on_a_large_screenshot(self):
        scale, reason = self.coords.infer_coord_scale([[0.22, 0.228]], [(1920, 1080)])
        self.assertEqual(scale, self.coords.NORM_1)
        self.assertIsNone(reason)
        point, out_of_range = self.coords.normalize_click_point([0.22, 0.228], (1920, 1080), scale)
        self.assertEqual(point, [0.22, 0.228])
        self.assertFalse(out_of_range)

    def test_qwen_grid_on_a_larger_screenshot_is_divided_by_1000(self):
        # Without a box the same numbers are also legal pixels, so auto keeps them.
        # The annotated box contains only the 0-1000 reading, which is the Qwen case.
        image = (1920, 1080)
        box = norm_xywh([400, 200, 80, 80], image)
        bare, bare_reason = self.coords.infer_coord_scale([[220, 228]], [image])
        self.assertEqual(bare, self.coords.PIXEL)
        self.assertIn('Keeping absolute pixels', bare_reason)
        scale, reason = self.coords.infer_coord_scale([[220, 228]], [image], boxes=[box])
        self.assertEqual(scale, self.coords.NORM_1000)
        self.assertIn('0-1000', reason)
        point, out_of_range = self.coords.normalize_click_point([220, 228], image, scale)
        self.assertEqual(point, [0.22, 0.228])
        self.assertFalse(out_of_range)

    def test_fraction_in_the_same_file_as_a_qwen_click_is_not_rescaled(self):
        image = (1920, 1080)
        box = norm_xywh([400, 200, 80, 80], image)
        scale, _reason = self.coords.infer_coord_scale(
            [[0.22, 0.228], [220, 228]], [image, image], boxes=[box, box])
        self.assertEqual(scale, self.coords.NORM_1000)
        fraction, _ = self.coords.normalize_click_point([0.22, 0.228], (1920, 1080), scale)
        grid, _ = self.coords.normalize_click_point([220, 228], (1920, 1080), scale)
        self.assertEqual(fraction, [0.22, 0.228])
        self.assertEqual(grid, [0.22, 0.228])

    def test_value_above_1000_keeps_absolute_pixels_for_the_whole_file(self):
        points = [[800, 400], [1500, 200]]
        sizes = [(1920, 1080), (1920, 1080)]
        scale, reason = self.coords.infer_coord_scale(points, sizes)
        self.assertEqual(scale, self.coords.PIXEL)
        self.assertIsNone(reason)
        left, _ = self.coords.normalize_click_point(points[0], sizes[0], scale)
        right, _ = self.coords.normalize_click_point(points[1], sizes[1], scale)
        self.assertAlmostEqual(left[0], 800 / 1920)
        self.assertAlmostEqual(left[1], 400 / 1080)
        self.assertAlmostEqual(right[0], 1500 / 1920)
        self.assertAlmostEqual(right[1], 200 / 1080)

    def test_click_outside_the_image_but_inside_the_grid_uses_1000(self):
        scale, reason = self.coords.infer_coord_scale([[220, 228]], [(100, 100)])
        self.assertEqual(scale, self.coords.NORM_1000)
        self.assertIn('outside the screenshot', reason)
        point, out_of_range = self.coords.normalize_click_point([220, 228], (100, 100), scale)
        self.assertEqual(point, [0.22, 0.228])
        self.assertFalse(out_of_range)

    def test_small_screenshot_keeps_pixels_and_explains_the_ambiguity(self):
        scale, reason = self.coords.infer_coord_scale([[40, 40]], [(100, 100)])
        self.assertEqual(scale, self.coords.PIXEL)
        self.assertIn('SCREENSPOT_COORD_SCALE=norm_1000', reason)
        point, out_of_range = self.coords.normalize_click_point([40, 40], (100, 100), scale)
        self.assertEqual(point, [0.4, 0.4])
        self.assertFalse(out_of_range)

    def test_explicit_pixel_override_does_not_divide_by_1000(self):
        scale, reason = self.coords.resolve_click_scale([[220, 228]], [(1920, 1080)],
                                                        explicit='pixel')
        self.assertEqual((scale, reason), (self.coords.PIXEL, None))
        point, out_of_range = self.coords.normalize_click_point([220, 228], (1920, 1080), scale)
        self.assertAlmostEqual(point[0], 220 / 1920)
        self.assertAlmostEqual(point[1], 228 / 1080)
        self.assertFalse(out_of_range)

    def test_pixel_click_past_the_image_is_reported_out_of_range(self):
        point, out_of_range = self.coords.normalize_click_point([5000, 5000], (100, 100),
                                                                self.coords.PIXEL)
        self.assertEqual(point, [50.0, 50.0])
        self.assertTrue(out_of_range)

    def test_negative_sentinel_is_not_an_out_of_range_unit_mismatch(self):
        point, out_of_range = self.coords.normalize_click_point([-1.0, -1.0], (1920, 1080),
                                                                self.coords.NORM_1000)
        self.assertEqual(point, [-1.0, -1.0])
        self.assertFalse(out_of_range)

    def test_env_override(self):
        with mock.patch.dict('os.environ', {'SCREENSPOT_COORD_SCALE': 'norm_1000'}, clear=True):
            scale, reason = self.coords.resolve_click_scale([[1500, 200]], [(1920, 1080)])
        self.assertEqual((scale, reason), (self.coords.NORM_1000, None))

    def test_web_thumb_up_keeps_pixel_clicks_when_the_grid_also_fits(self):
        image = (2560, 1440)
        box = norm_xywh([542, 700, 107, 67], image)
        scale, reason = self.coords.infer_coord_scale([[595, 733]], [image], boxes=[box])
        self.assertEqual(scale, self.coords.PIXEL)
        self.assertIn('Keeping absolute pixels', reason)
        pixel, out_of_range = self.coords.normalize_click_point([595, 733], image, scale)
        self.assertFalse(out_of_range)
        self.assertTrue(box[0] <= pixel[0] <= box[2] and box[1] <= pixel[1] <= box[3])
        # Dividing the same numbers by 1000 is the miss around pixel (1523, 1056).
        grid, _ = self.coords.normalize_click_point([595, 733], image, self.coords.NORM_1000)
        self.assertAlmostEqual(grid[0] * image[0], 1523.2, places=1)
        self.assertAlmostEqual(grid[1] * image[1], 1055.52, places=1)
        self.assertFalse(box[0] <= grid[0] <= box[2] and box[1] <= grid[1] <= box[3])
        # (232, 509) is that click on the 0-1000 grid and still lands in the button.
        same, _ = self.coords.normalize_click_point([232, 509], image, self.coords.NORM_1000)
        self.assertTrue(box[0] <= same[0] <= box[2] and box[1] <= same[1] <= box[3])

    def test_tie_between_pixel_and_grid_keeps_pixels(self):
        image = (1920, 1080)
        scale, _reason = self.coords.infer_coord_scale(
            [[220, 228]], [image], boxes=[[0.0, 0.0, 1.0, 1.0]])
        self.assertEqual(scale, self.coords.PIXEL)


class TestScreenSpotScoring(unittest.TestCase):

    def evaluate(self, module_name, class_name, records, image_size, coord_scale=None):
        module = load_gui_module(module_name)
        dataset = getattr(module, class_name).__new__(getattr(module, class_name))
        dataset.img_root = '/unused'
        dataset.parse_response_func = module.parse_bbox_aguvis
        image = mock.Mock(size=image_size)
        kwargs = {} if coord_scale is None else {'coord_scale': coord_scale}
        with mock.patch.object(module, 'load', return_value=pd.DataFrame(records)), \
                mock.patch.object(module.Image, 'open', return_value=image), \
                mock.patch.dict('os.environ', {}, clear=True):
            score = dataset.evaluate_point('predictions.xlsx', **kwargs)
        return module, score

    def test_prompt_states_absolute_pixels(self):
        for name in ('screenspot', 'screenspot_pro', 'screenspot_v2'):
            module = load_gui_module(name)
            with self.subTest(dataset=name):
                self.assertIn('absolute pixel coordinates', module.SYSTEM_PROMPT)
                self.assertIn('not a 0-1000 grid', module.USER_INSTRUCTION)
                rendered = module.USER_INSTRUCTION.format(instruction='open the menu')
                self.assertIn('open the menu', rendered)
                self.assertIn('absolute pixel coordinates', rendered)

    def test_user_turn_receives_the_coordinate_convention(self):
        module = load_gui_module('screenspot')
        dataset = module.ScreenSpot.__new__(module.ScreenSpot)
        dataset.RE_TYPE = 'functional'
        dataset.dump_image = lambda _line: ['/tmp/screen.png']
        msgs = dataset.build_prompt(pd.Series({'question': 'open the menu'}))
        text = ' '.join(msg['value'] for msg in msgs if 'value' in msg)
        self.assertIn('absolute pixel coordinates', text)
        self.assertIn('open the menu', text)
        self.assertEqual(msgs[0]['role'], 'system')

    def test_qwen_grid_click_hits_the_normalized_box(self):
        records = [{
            'bbox': '[400, 200, 80, 80]',
            'prediction': 'pyautogui.click(x=220, y=228)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'data_type': 'text',
            'data_source': 'test',
        }, {
            'bbox': '[400, 200, 80, 80]',
            'prediction': 'pyautogui.click(x=0.22, y=0.228)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'data_type': 'icon',
            'data_source': 'test',
        }]
        _module, score = self.evaluate('screenspot', 'ScreenSpot', records, (1920, 1080))
        self.assertEqual(score['Coord_Scale'], 'norm_1000')
        self.assertEqual(score['Overall_Accuracy'], 100)
        self.assertEqual(score['Format_Err_Rate'], 0)
        self.assertEqual(score['Out_Of_Range_Rate'], 0)

    def test_pixel_override_scores_the_same_qwen_click_as_a_miss(self):
        records = [{
            'bbox': '[400, 200, 80, 80]',
            'prediction': 'pyautogui.click(x=220, y=228)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'data_type': 'text',
            'data_source': 'test',
        }]
        _module, score = self.evaluate(
            'screenspot', 'ScreenSpot', records, (1920, 1080), coord_scale='pixel')
        self.assertEqual(score['Coord_Scale'], 'pixel')
        self.assertEqual(score['Overall_Accuracy'], 0)
        self.assertEqual(score['Out_Of_Range_Rate'], 0)

    def test_absolute_pixels_still_hit_when_one_coordinate_exceeds_1000(self):
        records = [{
            'bbox': '[1480, 780, 40, 40]',
            'prediction': 'pyautogui.click(x=1500, y=800)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'data_type': 'text',
            'data_source': 'test',
        }, {
            'bbox': '[780, 380, 40, 40]',
            'prediction': 'pyautogui.click(x=800, y=400)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'data_type': 'icon',
            'data_source': 'test',
        }]
        _module, score = self.evaluate('screenspot', 'ScreenSpot', records, (1920, 1080))
        self.assertEqual(score['Coord_Scale'], 'pixel')
        self.assertEqual(score['Overall_Accuracy'], 100)
        self.assertEqual(score['Out_Of_Range_Rate'], 0)

    def test_out_of_range_pixel_is_visible_in_the_score(self):
        records = [{
            'bbox': '[0, 0, 20, 20]',
            'prediction': 'pyautogui.click(x=5000, y=5000)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'data_type': 'text',
            'data_source': 'test',
        }]
        module, score = self.evaluate('screenspot', 'ScreenSpot', records, (100, 100))
        self.assertEqual(score['Coord_Scale'], 'pixel')
        self.assertEqual(score['Overall_Accuracy'], 0)
        self.assertEqual(score['Format_Err_Rate'], 0)
        self.assertEqual(score['Out_Of_Range_Rate'], 100)
        self.assertTrue(module.logger.warning.called)

    def test_screenspot_pro_qwen_grid_uses_xyxy_boxes(self):
        records = [{
            'bbox': '[400, 200, 479, 279]',
            'prediction': 'pyautogui.click(x=220, y=228)',
            'image_path': 'screen.png',
            'question': 'open the menu',
            'category': 'os',
            'ui_type': 'text',
            'application': 'test',
            'data_type': 'text',
            'data_source': 'test',
        }]
        _module, score = self.evaluate('screenspot_pro', 'ScreenSpot_Pro', records, (1920, 1080))
        self.assertEqual(score['Coord_Scale'], 'norm_1000')
        self.assertEqual(score['Overall_Accuracy'], 100)
        self.assertEqual(score['Text_Accuracy'], 100)
        self.assertEqual(score['Out_Of_Range_Rate'], 0)

    def _thumb_up_record(self, prediction):
        # ScreenSpot Web web_b97bb6a3-999c-463a-b957-5584c14c8395.png is 2560x1440.
        # The "thumb up" button is [542, 700, 107, 67] (x, y, w, h).
        return {
            'bbox': '[542, 700, 107, 67]',
            'prediction': prediction,
            'image_path': 'web_b97bb6a3-999c-463a-b957-5584c14c8395.png',
            'question': 'thumb up',
            'data_type': 'icon',
            'data_source': 'gitlab',
        }

    def test_web_thumb_up_pixel_click_scores_under_auto_and_overrides(self):
        image = (2560, 1440)
        pixel_click = 'pyautogui.click(x=595, y=733)'
        for coord_scale in (None, 'pixel'):
            with self.subTest(coord_scale=coord_scale or 'auto'):
                _module, score = self.evaluate(
                    'screenspot', 'ScreenSpot', [self._thumb_up_record(pixel_click)], image,
                    coord_scale=coord_scale)
                self.assertEqual(score['Coord_Scale'], 'pixel')
                self.assertEqual(score['Overall_Accuracy'], 100)
                self.assertEqual(score['Icon_Accuracy'], 100)
                self.assertEqual(score['Format_Err_Rate'], 0)
                self.assertEqual(score['Out_Of_Range_Rate'], 0)

        _module, forced_grid = self.evaluate(
            'screenspot', 'ScreenSpot', [self._thumb_up_record(pixel_click)], image,
            coord_scale='norm_1000')
        self.assertEqual(forced_grid['Coord_Scale'], 'norm_1000')
        self.assertEqual(forced_grid['Overall_Accuracy'], 0)

        # The 0-1000 form of this click matches the button, so auto still divides by 1000.
        _module, auto_grid = self.evaluate(
            'screenspot',
            'ScreenSpot',
            [self._thumb_up_record('pyautogui.click(x=232, y=509)')],
            image,
        )
        self.assertEqual(auto_grid['Coord_Scale'], 'norm_1000')
        self.assertEqual(auto_grid['Overall_Accuracy'], 100)

        _module, grid_form = self.evaluate(
            'screenspot',
            'ScreenSpot',
            [self._thumb_up_record('pyautogui.click(x=232, y=509)')],
            image,
            coord_scale='norm_1000',
        )
        self.assertEqual(grid_form['Coord_Scale'], 'norm_1000')
        self.assertEqual(grid_form['Overall_Accuracy'], 100)
        self.assertEqual(grid_form['Out_Of_Range_Rate'], 0)


if __name__ == '__main__':
    unittest.main()
