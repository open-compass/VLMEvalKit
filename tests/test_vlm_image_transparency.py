import base64
import importlib.util
import io
import logging
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


def load_vlm_module():
    root = Path(__file__).resolve().parents[1]
    package = types.ModuleType('vlmeval')
    package.__path__ = [str(root / 'vlmeval')]
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = [str(root / 'vlmeval' / 'smp')]
    log = types.ModuleType('vlmeval.smp.log')
    log.get_logger = logging.getLogger
    modules = {'vlmeval': package, 'vlmeval.smp': smp, 'vlmeval.smp.log': log}
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            'vlmeval.smp.vlm', root / 'vlmeval' / 'smp' / 'vlm.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module, modules


def transparent_samples():
    rgba = Image.new('RGBA', (2, 1), (0, 0, 0, 0))
    rgba.putpixel((1, 0), (0, 0, 0, 255))

    la = Image.new('LA', (2, 1), (0, 0))
    la.putpixel((1, 0), (0, 255))

    palette = Image.new('P', (2, 1), 0)
    palette.putpalette([0, 0, 0, 0, 0, 0] + [0] * 762)
    palette.putpixel((1, 0), 1)
    palette.info['transparency'] = 0

    return {'RGBA': rgba, 'LA': la, 'P': palette}


def to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


class TestImageTransparency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vlm, cls.modules = load_vlm_module()

    def assert_white_background_and_black_line(self, image):
        image = image.convert('RGB')
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(image.getpixel((1, 0)), (0, 0, 0))

    def test_decode_flattens_alpha_and_palette_transparency_to_white(self):
        for mode, image in transparent_samples().items():
            with self.subTest(mode=mode), mock.patch.dict(sys.modules, self.modules):
                decoded = self.vlm.decode_base64_to_image(to_base64(image))
                self.assertEqual(decoded.mode, 'RGB')
                self.assert_white_background_and_black_line(decoded)

    def test_encode_flattens_alpha_and_palette_transparency_to_white(self):
        settings = {'VLMEVAL_MIN_IMAGE_EDGE': '1', 'VLMEVAL_MAX_IMAGE_SIZE': '1000000'}
        for mode, image in transparent_samples().items():
            with self.subTest(mode=mode), mock.patch.dict(sys.modules, self.modules):
                with mock.patch.dict(os.environ, settings):
                    encoded = self.vlm.encode_image_to_base64(image, fmt='PNG')
                decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
                self.assert_white_background_and_black_line(decoded)


if __name__ == '__main__':
    unittest.main()
