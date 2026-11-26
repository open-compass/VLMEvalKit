import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from PIL import Image

from vlmeval.dataset.image_vqa import MATHVERSE_SOURCE_COMMIT, MATHVERSE_SOURCE_TAG, MathVerse

REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PARQUET = REPO_ROOT / '_internal/official/testmini.parquet'
NEW_DATASET = f'MathVerse_{MATHVERSE_SOURCE_TAG}'
NEW_URL = f'https://opencompass.openxlab.space/utils/VLMEval/{NEW_DATASET}.tsv'
NEW_TSV = (
    '/mnt/shared-storage-user/opencompass-shared/share/wuliwei/vlmevalkit/'
    f'LMUData/{NEW_DATASET}.tsv'
)
NEW_MD5 = 'a3fc7c5db7caf86cc59e2359ec46d80f'
needs_local_tsv = pytest.mark.skipif(
    not Path(NEW_TSV).exists(), reason='regenerated MathVerse TSV is not available locally')
needs_official_parquet = pytest.mark.skipif(
    not OFFICIAL_PARQUET.exists(), reason='official MathVerse parquet is not available locally')
HOSTED_MD5 = {
    'MathVerse_MINI': '5017caca32b7fa110c350a1bea861b65',
    'MathVerse_MINI_Vision_Only': '68a11d4680014ac881fa37adeadea3a4',
    'MathVerse_MINI_Vision_Only_cot': '68a11d4680014ac881fa37adeadea3a4',
    'MathVerse_MINI_Vision_Dominant': 'b8fb63852d261ab2aaefba29cc2414d3',
    'MathVerse_MINI_Vision_Intensive': '01cbd35be202bb0c4873a4186a63bc19',
    'MathVerse_MINI_Text_Lite': '19e4b13bdd30b89a03b2e358bcfefa04',
    'MathVerse_MINI_Text_Dominant': '4f5cd2fa6630ea00bb11d6fde1f6fe6a',
}


@pytest.fixture(scope='module')
def regenerated_dataset():
    if not Path(NEW_TSV).exists():
        pytest.skip('regenerated MathVerse TSV is not available locally')
    return MathVerse(NEW_DATASET)


def test_registration_and_hashes_keep_hosted_entries_unchanged():
    assert MATHVERSE_SOURCE_COMMIT == '3bc86196678bad115a923d2851c6821dbe235939'
    assert set(HOSTED_MD5).issubset(MathVerse.supported_datasets())
    assert NEW_DATASET in MathVerse.supported_datasets()
    assert MathVerse.DATASET_URL[NEW_DATASET] == NEW_URL
    assert MathVerse.DATASET_URL['MathVerse_MINI'] == (
        'https://opencompass.openxlab.space/utils/benchmarks/MathVerse/'
        'MathVerse_MINIV.tsv'
    )
    assert MathVerse.DATASET_MD5[NEW_DATASET] == NEW_MD5
    for dataset, expected_md5 in HOSTED_MD5.items():
        assert MathVerse.DATASET_MD5[dataset] == expected_md5


@needs_local_tsv
def test_regenerated_tsv_file_and_metadata_match_registration():
    with open(NEW_TSV, 'rb') as fin:
        assert hashlib.md5(fin.read()).hexdigest() == NEW_MD5
    with open(NEW_TSV + '.meta.json', encoding='utf-8') as fin:
        metadata = json.load(fin)
    assert metadata['md5'] == NEW_MD5
    assert metadata['source_revision'] == MATHVERSE_SOURCE_COMMIT
    assert metadata['rows'] == 3940
    assert metadata['preserve_official_encoded_bytes'] is True
    assert metadata['convert_to_rgb'] is False


def test_regenerated_dataset_loads_and_has_separate_image_root(regenerated_dataset):
    dataset = regenerated_dataset
    assert len(dataset) == 3940
    assert dataset.convert_to_rgb is False
    assert Path(dataset.img_root).name == NEW_DATASET
    assert Path(dataset.img_root).name != 'MathVerse_MINI'


def test_convert_to_rgb_false_is_limited_to_regenerated_entry(monkeypatch):
    def fake_load_data(self, dataset):
        return pd.DataFrame({'index': ['1']})

    monkeypatch.setattr(MathVerse, 'load_data', fake_load_data)
    hosted = MathVerse('MathVerse_MINI')
    regenerated = MathVerse(NEW_DATASET)

    assert hosted.convert_to_rgb is True
    assert regenerated.convert_to_rgb is False
    assert Path(hosted.img_root) != Path(regenerated.img_root)


@needs_official_parquet
def test_real_transparent_image_preserves_bytes_alpha_and_pixels(
        regenerated_dataset, tmp_path):
    rows = pq.read_table(OFFICIAL_PARQUET).to_pylist()
    transparent_row = None
    for row in rows:
        image = Image.open(io.BytesIO(row['image']['bytes']))
        if image.mode == 'RGBA' and np.asarray(image)[..., 3].min() < 255:
            transparent_row = row
            break
    assert transparent_row is not None

    sample_index = str(transparent_row['sample_index'])
    official_bytes = transparent_row['image']['bytes']
    tsv_row = regenerated_dataset.data[
        regenerated_dataset.data['sample_index'].astype(str) == sample_index].iloc[0]
    assert base64.b64decode(tsv_row['image']) == official_bytes

    old_root = regenerated_dataset.img_root
    regenerated_dataset.img_root = str(tmp_path)
    try:
        dumped_path = regenerated_dataset.dump_image(pd.Series({
            'index': f'mathverse-alpha-{sample_index}',
            'image': base64.b64encode(official_bytes).decode('utf-8'),
        }))[0]
    finally:
        regenerated_dataset.img_root = old_root

    with Image.open(dumped_path) as dumped:
        assert dumped.mode == 'RGBA'
        assert np.asarray(dumped)[..., 3].min() == 0
        with Image.open(io.BytesIO(official_bytes)) as official:
            assert np.array_equal(np.asarray(dumped), np.asarray(official.convert('RGBA')))
