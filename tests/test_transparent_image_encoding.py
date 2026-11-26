import base64
import io

from PIL import Image, ImageDraw

from vlmeval.smp.vlm import (decode_base64_to_image, decode_base64_to_image_file,
                             encode_image_to_base64)


def _png_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def _decode_encoded(image, **kwargs):
    encoded = encode_image_to_base64(image, fmt='PNG', **kwargs)
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def _rgba_artwork():
    image = Image.new('RGBA', (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([16, 16, 112, 112], outline=(0, 0, 0, 255), width=6)
    draw.rectangle([56, 56, 72, 72], fill=(255, 0, 0, 255))
    return image


def test_decode_default_uses_legacy_rgb_conversion():
    image = decode_base64_to_image(_png_base64(_rgba_artwork())).convert('RGB')

    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((64, 16)) == (0, 0, 0)
    assert image.getpixel((64, 64)) == (255, 0, 0)


def test_decode_can_preserve_alpha_when_convert_rgb_is_false():
    image = decode_base64_to_image(_png_base64(_rgba_artwork()), convert_rgb=False)

    assert image.mode == 'RGBA'
    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert image.getpixel((64, 16)) == (0, 0, 0, 255)
    assert image.getpixel((64, 64)) == (255, 0, 0, 255)


def test_decode_file_default_and_alpha_preserving_modes(tmp_path):
    encoded = _png_base64(_rgba_artwork())
    default_path = tmp_path / 'default.png'
    alpha_path = tmp_path / 'alpha.png'

    decode_base64_to_image_file(encoded, str(default_path))
    decode_base64_to_image_file(encoded, str(alpha_path), convert_rgb=False)

    with Image.open(default_path) as image:
        assert image.mode == 'RGB'
        assert image.getpixel((0, 0)) == (0, 0, 0)
    with Image.open(alpha_path) as image:
        assert image.mode == 'RGBA'
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)


def test_encode_default_uses_legacy_rgb_conversion():
    image = _decode_encoded(_rgba_artwork()).convert('RGB')

    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((64, 16)) == (0, 0, 0)
    assert image.getpixel((64, 64)) == (255, 0, 0)


def test_encode_can_preserve_alpha_when_convert_rgb_is_false():
    image = _decode_encoded(_rgba_artwork(), convert_rgb=False)

    assert image.mode == 'RGBA'
    assert image.getpixel((0, 0)) == (0, 0, 0, 0)
    assert image.getpixel((64, 16)) == (0, 0, 0, 255)
    assert image.getpixel((64, 64)) == (255, 0, 0, 255)


def test_palette_transparency_legacy_and_preserved_modes():
    image = Image.new('P', (128, 128), 0)
    image.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    image.info['transparency'] = 0
    ImageDraw.Draw(image).rectangle([40, 40, 88, 88], fill=1)

    decoded = decode_base64_to_image(_png_base64(image)).convert('RGB')
    preserved = decode_base64_to_image(_png_base64(image), convert_rgb=False)
    encoded = _decode_encoded(image).convert('RGB')

    assert decoded.getpixel((0, 0)) == (0, 0, 0)
    assert decoded.getpixel((64, 64)) == (255, 0, 0)
    assert preserved.mode == 'P'
    assert preserved.convert('RGBA').getpixel((0, 0))[3] == 0
    assert encoded.getpixel((0, 0)) == (0, 0, 0)
    assert encoded.getpixel((64, 64)) == (255, 0, 0)


def test_opaque_rgb_image_is_unchanged():
    image = Image.new('RGB', (128, 128), (10, 20, 30))

    decoded = decode_base64_to_image(_png_base64(image)).convert('RGB')
    encoded = _decode_encoded(image).convert('RGB')

    assert decoded.getpixel((64, 64)) == (10, 20, 30)
    assert encoded.getpixel((64, 64)) == (10, 20, 30)
