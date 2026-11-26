from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import vlmeval.dataset.image_base as image_base
import vlmeval.dataset.image_vqa as image_vqa
from vlmeval.dataset.image_vqa import MMVET_ALPHA_DATASET, MMVET_ALPHA_MD5, MMVET_ALPHA_TSV, MMVet

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_ALPHA_TSV = Path(
    '/mnt/shared-storage-user/opencompass-shared/share/wuliwei/vlmevalkit/LMUData/MMVet_2df561.tsv')
OFFICIAL_RGBA_IMAGE = REPO_ROOT / '_internal/official/mm-vet/images/v1_52.png'


def _stub_dataset(monkeypatch, tmp_path, dataset):
    frame = pd.DataFrame([{
        'index': [1],
        'question': ['question'],
        'answer': ['answer'],
        'category': ['rec'],
    }])
    monkeypatch.setattr(MMVet, 'load_data', lambda self, name: frame.copy())
    monkeypatch.setattr(image_base, 'LMUDataRoot', lambda: str(tmp_path))
    monkeypatch.setattr(image_vqa, 'LMUDataRoot', lambda: str(tmp_path))
    return MMVet(dataset=dataset)


def test_mmvet_registration_preserves_hosted_entries():
    assert MMVet.DATASET_URL['MMVet'] == (
        'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv')
    assert MMVet.DATASET_URL['MMVet_Hard'] == (
        'https://opencompass.openxlab.space/utils/VLMEval/MMVet_Hard.tsv')
    assert MMVet.DATASET_MD5['MMVet'] == '748aa6d4aa9d4de798306a63718455e3'
    assert MMVet.DATASET_MD5['MMVet_Hard'] == '63a598819a936a2e77c410a78a21ff16'


def test_mmvet_regenerated_entry_has_url_and_md5():
    assert MMVET_ALPHA_DATASET == 'MMVet_2df561'
    assert MMVet.DATASET_URL[MMVET_ALPHA_DATASET] == MMVET_ALPHA_TSV
    assert MMVET_ALPHA_TSV == 'https://opencompass.openxlab.space/utils/VLMEval/MMVet_2df561.tsv'
    assert MMVet.DATASET_MD5[MMVET_ALPHA_DATASET] == MMVET_ALPHA_MD5
    assert MMVet.supported_datasets() == ['MMVet', 'MMVet_Hard', MMVET_ALPHA_DATASET]


def test_mmvet_only_regenerated_entry_preserves_transparency(monkeypatch, tmp_path):
    hosted = _stub_dataset(monkeypatch, tmp_path, 'MMVet')
    hard = _stub_dataset(monkeypatch, tmp_path, 'MMVet_Hard')
    regenerated = _stub_dataset(monkeypatch, tmp_path, MMVET_ALPHA_DATASET)

    assert hosted.convert_to_rgb is True
    assert hard.convert_to_rgb is True
    assert regenerated.convert_to_rgb is False


def test_mmvet_regenerated_entry_uses_separate_image_root(monkeypatch, tmp_path):
    hosted = _stub_dataset(monkeypatch, tmp_path, 'MMVet')
    regenerated = _stub_dataset(monkeypatch, tmp_path, MMVET_ALPHA_DATASET)

    assert regenerated.img_root == str(tmp_path / 'images' / MMVET_ALPHA_DATASET)
    assert regenerated.img_root != hosted.img_root


@pytest.mark.skipif(not LOCAL_ALPHA_TSV.exists(), reason='regenerated MMVet TSV is unavailable')
def test_mmvet_regenerated_dataset_loads_all_rows_and_md5():
    dataset = MMVet(dataset=MMVET_ALPHA_DATASET)

    assert len(dataset) == 218
    assert dataset.convert_to_rgb is False
    assert dataset.data['index'].tolist() == list(range(1, 219))


@pytest.mark.skipif(
    not LOCAL_ALPHA_TSV.exists() or not OFFICIAL_RGBA_IMAGE.exists(),
    reason='regenerated TSV or official RGBA sample is unavailable',
)
def test_mmvet_dump_image_preserves_official_rgba_pixels(tmp_path):
    dataset = MMVet(dataset=MMVET_ALPHA_DATASET)
    line = dict(dataset.data.iloc[52])
    dataset.img_root = str(tmp_path / 'images' / MMVET_ALPHA_DATASET)

    dumped_path = dataset.dump_image(line)[0]
    with Image.open(dumped_path) as dumped, Image.open(OFFICIAL_RGBA_IMAGE) as official:
        assert dumped.mode == 'RGBA'
        assert official.mode == 'RGBA'
        dumped_rgba = np.asarray(dumped.convert('RGBA'))
        official_rgba = np.asarray(official.convert('RGBA'))
        assert np.array_equal(dumped_rgba, official_rgba)
        assert np.any(dumped_rgba[..., 3] == 0)
