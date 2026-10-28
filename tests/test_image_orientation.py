import base64
import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest
from PIL import Image, ImageOps


@pytest.fixture
def vlm(monkeypatch):
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    root = Path(__file__).resolve().parents[1]
    original = {name: module for name, module in sys.modules.items() if name.startswith('vlmeval')}
    try:
        # Namespace only: keep actual image/log helpers and their real dependencies,
        # without importing the unrelated optional model registry.
        for name in ('vlmeval', 'vlmeval.smp'):
            package = types.ModuleType(name)
            package.__path__ = [str(root / name.replace('.', '/'))]
            sys.modules[name] = package
        for name in ('log', 'vlm'):
            full_name = f'vlmeval.smp.{name}'
            spec = importlib.util.spec_from_file_location(full_name, root / f'vlmeval/smp/{name}.py')
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        yield module
    finally:
        for name in list(sys.modules):
            if name.startswith('vlmeval'):
                del sys.modules[name]
        sys.modules.update(original)


def _camera_jpeg(path, orientation):
    image = Image.new('RGB', (64, 32), 'red')
    image.paste('blue', (32, 0, 64, 16))
    image.paste('green', (0, 16, 32, 32))
    image.paste('yellow', (32, 16, 64, 32))
    exif = image.getexif()
    exif[274] = orientation
    image.save(path, quality=100, subsampling=0, exif=exif)


def _assert_display_pixels(actual, expected):
    assert actual.size == expected.size
    width, height = expected.size
    for x in (width // 4, 3 * width // 4):
        for y in (height // 4, 3 * height // 4):
            assert all(abs(a - b) <= 8 for a, b in zip(actual.getpixel((x, y)), expected.getpixel((x, y))))
    assert actual.getexif().get(274) in (None, 1)


@pytest.mark.parametrize('orientation', range(2, 9))
def test_camera_jpeg_encoding_preserves_display_orientation(vlm, tmp_path, orientation):
    path = tmp_path / 'camera.jpg'
    _camera_jpeg(path, orientation)
    with Image.open(path) as source:
        expected = ImageOps.exif_transpose(source)
        encoded = vlm.encode_image_file_to_base64(str(path))
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as actual:
            _assert_display_pixels(actual, expected)


@pytest.mark.parametrize('orientation', (6, 8))
def test_decode_resize_and_reencode_does_not_rotate_twice(vlm, tmp_path, orientation):
    path = tmp_path / 'camera.jpg'
    _camera_jpeg(path, orientation)
    with Image.open(path) as source:
        expected = ImageOps.exif_transpose(source)
        expected.thumbnail((32, 32))
    decoded = vlm.decode_base64_to_image(base64.b64encode(path.read_bytes()).decode(), target_size=32)
    _assert_display_pixels(decoded, expected)
    encoded = vlm.encode_image_to_base64(decoded)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as actual:
        _assert_display_pixels(actual, expected)


def test_normal_image_encoding_stays_byte_identical(vlm):
    image = Image.new('RGB', (64, 32), 'red')
    expected = io.BytesIO()
    image.save(expected, format='JPEG')
    assert base64.b64decode(vlm.encode_image_to_base64(image)) == expected.getvalue()


def test_oriented_image_keeps_existing_minimum_edge_upscale(vlm, tmp_path, monkeypatch):
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '64')
    path = tmp_path / 'camera.jpg'
    _camera_jpeg(path, 6)
    result = vlm.encode_image_file_to_base64(str(path), target_size=16)
    with Image.open(io.BytesIO(base64.b64decode(result))) as actual:
        assert actual.size == (64, 128)
