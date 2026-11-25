import base64
import io
import os.path as osp

import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset.dynamath import Dynamath

HOSTED_URL = 'https://opencompass.openxlab.space/utils/VLMEval/DynaMath.tsv'
HOSTED_MD5 = 'b8425ad9a7114571fc9366e013699494'
REGENERATED = 'DynaMath_598c3c'
REGENERATED_TSV = 'https://opencompass.openxlab.space/utils/VLMEval/DynaMath_598c3c.tsv'
REGENERATED_MD5 = 'bc07e9a07aea5e13218275c80fced98e'


def _transparent_png_base64():
    image = Image.new('RGBA', (2, 2), (10, 20, 30, 0))
    image.putpixel((0, 0), (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('ascii')


@pytest.fixture
def build_dynamath(monkeypatch, tmp_path):
    monkeypatch.setenv('LMUData', str(tmp_path))
    frame = pd.DataFrame([{
        'index': '1',
        'qid': '1',
        'varid': '1',
        'question': 'question',
        'image': _transparent_png_base64(),
        'answer': '1',
        'answer_type': 'float',
        'subject': 'algebra',
        'knowledge_level': 'high school',
    }])
    monkeypatch.setattr(Dynamath, 'load_data', lambda self, dataset: frame.copy())
    return lambda name: Dynamath(dataset=name)


def test_dynamath_registration_keeps_hosted_entries_and_adds_regenerated():
    supported = Dynamath.supported_datasets()

    assert {'DynaMath', 'DynaMath_noprompt', REGENERATED}.issubset(supported)
    assert Dynamath.DATASET_URL['DynaMath'] == HOSTED_URL
    assert Dynamath.DATASET_URL['DynaMath_noprompt'] == HOSTED_URL
    assert Dynamath.DATASET_URL[REGENERATED] == REGENERATED_TSV
    assert Dynamath.DATASET_MD5['DynaMath'] == HOSTED_MD5
    assert Dynamath.DATASET_MD5['DynaMath_noprompt'] == HOSTED_MD5
    assert Dynamath.DATASET_MD5[REGENERATED] == REGENERATED_MD5


@pytest.mark.parametrize('dataset', ['DynaMath', 'DynaMath_noprompt'])
def test_hosted_dynamath_keeps_rgb_conversion_and_original_image_root(
        build_dynamath, tmp_path, dataset):
    instance = build_dynamath(dataset)

    assert instance.convert_to_rgb is True
    assert instance.img_root == osp.join(str(tmp_path), 'images', dataset)


def test_regenerated_dynamath_preserves_alpha_and_uses_separate_image_root(
        build_dynamath, tmp_path):
    instance = build_dynamath(REGENERATED)

    assert instance.convert_to_rgb is False
    assert instance.img_root == osp.join(str(tmp_path), 'images', REGENERATED)
    assert instance.img_root != osp.join(str(tmp_path), 'images', 'DynaMath')
    assert instance.img_root != osp.join(str(tmp_path), 'images', 'DynaMath_noprompt')

    dumped = instance.dump_image(instance.data.iloc[0])[0]
    with Image.open(dumped) as image:
        assert image.mode == 'RGBA'
        assert image.getpixel((1, 1)) == (10, 20, 30, 0)
        assert image.getpixel((0, 0)) == (255, 0, 0, 255)
