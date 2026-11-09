"""Transparent pixels must be composited on white, not turned black, before encoding."""

import base64
import io

from PIL import Image

from vlmeval.smp.vlm import encode_image_to_base64


def _decode(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')


def test_transparent_region_becomes_white_not_black():
    image = Image.new('RGBA', (64, 64), (255, 0, 0, 0))

    pixel = _decode(encode_image_to_base64(image)).getpixel((32, 32))

    assert all(channel > 200 for channel in pixel), pixel


def test_opaque_pixels_survive():
    image = Image.new('RGBA', (64, 64), (0, 0, 255, 255))

    pixel = _decode(encode_image_to_base64(image)).getpixel((32, 32))

    assert pixel[2] > 200 and pixel[0] < 60, pixel
