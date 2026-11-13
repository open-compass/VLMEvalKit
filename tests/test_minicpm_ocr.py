import numpy as np
import pytest
from PIL import Image

from vlmeval.vlm.minicpm_ocr import MiniCPM_OCR


def _model():
    model = MiniCPM_OCR.__new__(MiniCPM_OCR)
    model.default_prompt = 'Text Recognition:'
    return model


def _image(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (8, 8), color='white').save(path)
    return str(path)


def test_empty_prompt_uses_default(tmp_path):
    content = MiniCPM_OCR.build_content(_model(),
                                        [dict(type='image', value=_image(tmp_path)),
                                         dict(type='text', value='')])
    assert content[-1] == dict(type='text', text='Text Recognition:')


def test_whitespace_prompt_uses_default(tmp_path):
    content = MiniCPM_OCR.build_content(_model(),
                                        [dict(type='image', value=_image(tmp_path)),
                                         dict(type='text', value='  \n\t')])
    assert content[-1] == dict(type='text', text='Text Recognition:')


@pytest.mark.parametrize('value', [float('nan'), np.float32('nan')])
def test_nan_prompt_uses_default(tmp_path, value):
    content = MiniCPM_OCR.build_content(_model(),
                                        [dict(type='image', value=_image(tmp_path)),
                                         dict(type='text', value=value)])
    assert content[-1] == dict(type='text', text='Text Recognition:')


def test_normal_prompt_is_kept(tmp_path):
    content = MiniCPM_OCR.build_content(
        _model(), [dict(type='image', value=_image(tmp_path)),
                   dict(type='text', value='What is written?')])
    assert content[-1] == dict(type='text', text='What is written?')


def test_multiple_images_preserve_order(tmp_path):
    image = _image(tmp_path)
    content = MiniCPM_OCR.build_content(
        _model(),
        [dict(type='image', value=image),
         dict(type='image', value=image),
         dict(type='text', value='Compare.')],
    )
    assert [item['type'] for item in content] == ['image', 'image', 'text']


def test_video_uses_single_slice_and_no_image_id():
    model = _model()
    model.downsample_mode = '4x'
    model.max_slice_nums = 9
    model.use_image_id = True
    assert model.get_processor_kwargs('OCRBench') == dict(downsample_mode='4x', max_slice_nums=9, use_image_id=True)
    assert model.get_processor_kwargs('Video-MME') == dict(downsample_mode='4x', max_slice_nums=1, use_image_id=False)


def test_unsupported_message_type_raises():
    with pytest.raises(ValueError):
        MiniCPM_OCR.build_content(_model(),
                                  [dict(type='video', value='video.mp4'),
                                   dict(type='text', value='What happens?')])
