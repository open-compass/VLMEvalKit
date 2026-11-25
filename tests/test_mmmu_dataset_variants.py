import base64
import io

import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset.image_base import img_root_map
from vlmeval.dataset.image_mcq import MMMU_SOURCE_TAG, MMMUDataset


def test_mmmu_original_and_regenerated_dataset_entries():
    assert MMMU_SOURCE_TAG == '876ce5'
    assert MMMUDataset.DATASET_URL['MMMU_DEV_VAL'].startswith('https://')
    assert MMMUDataset.DATASET_URL['MMMU_TEST'].startswith('https://')

    dev_name = f'MMMU_DEV_VAL_{MMMU_SOURCE_TAG}'
    test_name = f'MMMU_TEST_{MMMU_SOURCE_TAG}'
    assert MMMUDataset.DATASET_URL[dev_name].endswith('MMMU_DEV_VAL_876ce5.tsv')
    assert MMMUDataset.DATASET_URL[test_name].endswith('MMMU_TEST_876ce5.tsv')
    assert MMMUDataset.DATASET_MD5[dev_name] == 'b3d313cd9dfefe438fdfcfa643daac96'
    assert MMMUDataset.DATASET_MD5[test_name] == '60e7b18a7a947272b35e9f2b05a559df'


@pytest.mark.parametrize('original', ['MMMU_TEST', 'MMMU_DEV_VAL'])
def test_regenerated_mmmu_variants_use_separate_image_roots(monkeypatch, tmp_path, original):
    regenerated = f'{original}_{MMMU_SOURCE_TAG}'
    assert img_root_map(original) == original
    assert img_root_map(regenerated) == regenerated
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('LMURoot', str(tmp_path))

    def load_data(self, dataset):
        color = (255, 0, 0, 255) if dataset == original else (0, 0, 0, 0)
        buffer = io.BytesIO()
        Image.new('RGBA', (2, 2), color).save(buffer, format='PNG')
        image = base64.b64encode(buffer.getvalue()).decode('ascii')
        return pd.DataFrame({'index': [1], 'image': [image], 'question': ['q']})

    monkeypatch.setattr(MMMUDataset, 'load_data', load_data)
    legacy = MMMUDataset(dataset=original)
    updated = MMMUDataset(dataset=regenerated)
    [legacy_path] = legacy.dump_image(legacy.data.iloc[0])
    [updated_path] = updated.dump_image(updated.data.iloc[0])

    assert legacy_path != updated_path
    with Image.open(legacy_path) as old_image, Image.open(updated_path) as new_image:
        assert old_image.mode == 'RGB'
        assert old_image.getpixel((0, 0)) == (255, 0, 0)
        assert new_image.mode == 'RGBA'
        assert new_image.getpixel((0, 0)) == (0, 0, 0, 0)


def test_mmmu_legacy_entries_keep_rgb_and_regenerated_entries_preserve_alpha(monkeypatch):
    frame = pd.DataFrame({
        'index': ['1'],
        'image': ['x' * 65],
        'question': ['q'],
    })
    monkeypatch.setattr(MMMUDataset, 'load_data', lambda self, dataset: frame.copy())

    assert MMMUDataset(dataset='MMMU_DEV_VAL').convert_to_rgb is True
    assert MMMUDataset(dataset='MMMU_TEST').convert_to_rgb is True
    assert MMMUDataset(dataset=f'MMMU_DEV_VAL_{MMMU_SOURCE_TAG}').convert_to_rgb is False
    assert MMMUDataset(dataset=f'MMMU_TEST_{MMMU_SOURCE_TAG}').convert_to_rgb is False
