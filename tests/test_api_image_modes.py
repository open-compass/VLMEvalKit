import base64
import io
from unittest.mock import Mock

import pytest
from PIL import Image

from vlmeval.api.claude_sdk import ClaudeSDKAPI
from vlmeval.api.gpt import GPT4V
from vlmeval.api.lmdeploy import LMDeployAPI
from vlmeval.api.openai_responses import OpenAIResponsesAPI
from vlmeval.smp import encode_image_to_base64


@pytest.fixture(params=[LMDeployAPI, GPT4V, OpenAIResponsesAPI, ClaudeSDKAPI])
def make_model(request, monkeypatch):
    cls = request.param
    if hasattr(cls, '_create_fresh_client'):
        monkeypatch.setattr(cls, '_create_fresh_client', lambda self: Mock())
    monkeypatch.delenv('LMDEPLOY_API_KEY', raising=False)
    monkeypatch.delenv('VLMEVAL_MAX_IMAGE_SIZE', raising=False)
    monkeypatch.setenv('VLMEVAL_MIN_IMAGE_EDGE', '1')
    monkeypatch.setenv('BOYUE', '')
    monkeypatch.setenv('VLMEVAL_LOCAL_MEDIA', '0')

    def create(**kwargs):
        return cls(model='test-model', key='test-key', api_base='https://api.invalid/v1',
                   verbose=False, **kwargs)

    return create


def decode_content(content):
    if 'source' in content:
        mime = content['source']['media_type']
        encoded = content['source']['data']
    else:
        url = content['image_url']
        if isinstance(url, dict):
            url = url['url']
        header, encoded = url.split(',', 1)
        mime = header.removeprefix('data:').removesuffix(';base64')
    image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
    image.load()
    assert mime == Image.MIME[image.format]
    return image


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P', 'RGB', 'L', '1', 'CMYK'])
def test_default_preserves_image_mode_and_pixels(make_model, tmp_path, mode):
    image = Image.new(mode, (240, 160))
    if mode == 'RGBA':
        image.paste((23, 45, 67, 128), (0, 0, 240, 160))
    elif mode == 'LA':
        image.paste((45, 128), (0, 0, 240, 160))
    elif mode in ('P', 'RGB', 'L'):
        image.info['transparency'] = (0, 0, 0) if mode == 'RGB' else 0
    if mode == 'P':
        image.putpalette([23, 45, 67] * 256)
    path = tmp_path / ('image.jpg' if mode == 'CMYK' else 'image.png')
    image.save(path)
    model = make_model()

    content = model.prepare_itlist([{'type': 'image', 'value': str(path)}])

    encoded = decode_content(content[0])
    assert encoded.mode == mode
    assert encoded.size == image.size
    assert encoded.format == ('JPEG' if mode == 'CMYK' else 'PNG')
    if mode != 'CMYK':
        assert encoded.tobytes() == image.tobytes()
        assert encoded.convert('RGBA').tobytes() == image.convert('RGBA').tobytes()


def test_explicit_rgb_conversion_restores_legacy_jpeg(make_model, tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGBA', (240, 160), (23, 45, 67, 128)).save(path)
    model = make_model(convert_rgb=True)

    content = model.prepare_itlist([{'type': 'image', 'value': str(path)}])

    encoded = decode_content(content[0])
    assert encoded.format == 'JPEG'
    assert encoded.mode == 'RGB'
    assert encoded.size == (240, 160)
    assert 'convert_rgb' not in model.default_kwargs


def test_rgba_resize_and_file_limit_keep_alpha(make_model, tmp_path):
    path = tmp_path / 'image.png'
    image = Image.effect_noise((600, 300), 100).convert('RGBA')
    image.putalpha(128)
    image.save(path)
    model = make_model(img_size=300, total_img_size=400, max_file_size=10000)

    content = model.prepare_itlist([{'type': 'image', 'value': str(path)}] * 4)

    for item in content:
        encoded = decode_content(item)
        assert encoded.mode == 'RGBA'
        assert max(encoded.size) <= 200
        assert encoded.getchannel('A').getextrema() == (128, 128)
        buffer = io.BytesIO()
        encoded.save(buffer, format='PNG')
        assert len(base64.b64encode(buffer.getvalue())) <= 10000
    with Image.open(path) as original:
        assert original.size == (600, 300)
        assert original.mode == 'RGBA'


def test_shared_encoder_keeps_legacy_default():
    image = Image.new('RGBA', (240, 160), (23, 45, 67, 128))

    encoded = encode_image_to_base64(image)

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as result:
        assert result.mode == 'RGB'
        assert result.format == 'JPEG'
