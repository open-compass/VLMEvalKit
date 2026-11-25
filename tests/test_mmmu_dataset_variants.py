import pandas as pd

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


def test_regenerated_mmmu_variants_reuse_image_roots():
    assert img_root_map(f'MMMU_DEV_VAL_{MMMU_SOURCE_TAG}') == 'MMMU_DEV_VAL'
    assert img_root_map(f'MMMU_TEST_{MMMU_SOURCE_TAG}') == 'MMMU_TEST'


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
