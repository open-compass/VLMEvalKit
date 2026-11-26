import base64
import io
import os.path as osp

import numpy as np
import pytest
from PIL import Image

from vlmeval.dataset import build_dataset
from vlmeval.dataset.image_mcq import MMMUProDataset
from vlmeval.smp import LMUDataRoot

SOURCE_TAG = MMMUProDataset.SOURCE_TAG
RGBA_10C = f'MMMU_Pro_10c_{SOURCE_TAG}_rgba'


def test_mmmu_pro_registration_and_flags(monkeypatch):
    assert SOURCE_TAG == '0d7426'
    assert MMMUProDataset.DATASET_URL['MMMU_Pro_10c'].startswith('https://')
    assert MMMUProDataset.DATASET_URL['MMMU_Pro_V'].startswith('https://')
    assert MMMUProDataset.DATASET_URL[RGBA_10C] == (
        'https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c_0d7426_rgba.tsv')
    assert MMMUProDataset.DATASET_MD5[RGBA_10C] == '909b0631c0f1d80f97dd7e1add91ca8e'
    assert MMMUProDataset.supported_datasets() == [
        'MMMU_Pro_10c', 'MMMU_Pro_10c_COT', 'MMMU_Pro_V', 'MMMU_Pro_V_COT', RGBA_10C]

    import pandas as pd

    frame = pd.DataFrame({
        'index': ['1'],
        'image': ['x' * 65],
        'image_path': ['1.png'],
        'question': ['q'],
    })
    monkeypatch.setattr(MMMUProDataset, 'load_data', lambda self, dataset: frame.copy())
    for name in ('MMMU_Pro_10c', 'MMMU_Pro_10c_COT', 'MMMU_Pro_V', 'MMMU_Pro_V_COT'):
        assert MMMUProDataset(dataset=name).convert_to_rgb is True
    assert MMMUProDataset(dataset=RGBA_10C).convert_to_rgb is False


@pytest.mark.skipif(
    not osp.exists(osp.join(LMUDataRoot(), f'{RGBA_10C}.tsv')),
    reason='MMMU_Pro_10c rgba TSV (868MB) is not available locally; skip instead of downloading')
def test_mmmu_pro_10c_rgba_dump_preserves_alpha(tmp_path):
    dataset = build_dataset(RGBA_10C)
    assert dataset is not None
    assert len(dataset) == 1730
    assert dataset.convert_to_rgb is False
    assert dataset.img_root.endswith(RGBA_10C)

    dataset.img_root = str(tmp_path)
    row = dataset.data.loc[dataset.data['id'] == 'test_Music_293'].iloc[0]
    images = row['image'] if isinstance(row['image'], list) else [row['image']]
    source = Image.open(io.BytesIO(base64.b64decode(images[0])))
    source.load()
    dumped_path = dataset.dump_image(row)[0]
    dumped = Image.open(dumped_path)
    dumped.load()
    assert source.mode == 'RGBA'
    assert dumped.mode == 'RGBA'
    assert np.array_equal(np.asarray(source), np.asarray(dumped))
    assert dumped.getchannel('A').getextrema() == (0, 255)
