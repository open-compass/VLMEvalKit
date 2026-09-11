import base64
import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError


@pytest.fixture(scope='module')
def image_codec():
    # Exercise the real codec without importing optional model/API dependencies.
    path = Path(__file__).resolve().parents[1] / 'vlmeval' / 'smp' / 'vlm.py'
    spec = importlib.util.spec_from_file_location('image_codec', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode,fmt,color,expected_mode', [
    ('CMYK', 'JPEG', (20, 80, 140, 40), 'RGB'),
    ('LAB', 'TIFF', (100, 120, 140), 'RGB'),
    ('F', 'TIFF', 120.0, 'RGB'),
    ('RGBA', 'PNG', (20, 80, 140, 100), 'RGB'),
    ('P', 'PNG', 1, 'RGB'),
    ('LA', 'PNG', (80, 100), 'RGB'),
    ('RGB', 'PNG', (20, 80, 140), 'RGB'),
    ('1', 'PNG', 1, '1'),
    ('L', 'PNG', 80, 'L'),
    ('I;16', 'PNG', 1024, 'I;16'),
])
@pytest.mark.parametrize('target_size', [-1, 16])
def test_decode_image_to_png(image_codec, tmp_path, mode, fmt, color, expected_mode, target_size):
    source = Image.new(mode, (32, 16), color)
    if mode == 'P':
        source.putpalette([0, 0, 0, 20, 80, 140] + [0] * 762)
    buffer = io.BytesIO()
    source.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')

    with Image.open(io.BytesIO(buffer.getvalue())) as original:
        expected = original.convert(expected_mode)
    if target_size > 0:
        expected.thumbnail((target_size, target_size))

    decoded = image_codec.decode_base64_to_image(encoded, target_size=target_size)
    assert decoded.mode == expected_mode
    assert decoded.size == expected.size
    assert decoded.tobytes() == expected.tobytes()

    path = tmp_path / 'images' / 'decoded.png'
    image_codec.decode_base64_to_image_file(encoded, str(path), target_size=target_size)
    with Image.open(path) as saved:
        assert saved.format == 'PNG'
        assert saved.mode == expected_mode
        assert saved.size == expected.size
        assert saved.tobytes() == expected.tobytes()


def test_decode_invalid_image_raises(image_codec, tmp_path):
    encoded = base64.b64encode(b'not an image').decode('ascii')
    with pytest.raises(UnidentifiedImageError):
        image_codec.decode_base64_to_image_file(encoded, str(tmp_path / 'invalid.png'))
