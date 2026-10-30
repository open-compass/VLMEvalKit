import hashlib
from pathlib import Path
from unittest import mock

import pytest

from vlmeval.dataset import refadv


def test_build_refadv_tsv_rejects_source_md5_mismatch(tmp_path):
    parquet = tmp_path / 'source.parquet'
    parquet.write_bytes(b'not-the-pinned-benchmark-parquet')
    output = tmp_path / 'Ref-Adv-s.tsv'

    with mock.patch('huggingface_hub.hf_hub_download', return_value=str(parquet)) as download:
        with pytest.raises(ValueError, match='parquet MD5 mismatch'):
            refadv.build_refadv_tsv(str(output))

    download.assert_called_once_with(
        refadv.HF_REPO_ID,
        refadv.HF_FILENAME,
        repo_type='dataset',
        revision=refadv.HF_REVISION,
    )
    assert not output.exists()


def test_load_data_rebuilds_tsv_on_md5_mismatch(tmp_path):
    expected_bytes = b'expected Ref-Adv-s TSV contents'
    expected_md5 = hashlib.md5(expected_bytes).hexdigest()
    data_path = tmp_path / 'Ref-Adv-s.tsv'
    data_path.write_bytes(b'stale TSV contents')

    def fake_build(path):
        Path(path).write_bytes(expected_bytes)

    dataset = refadv.RefAdvDataset.__new__(refadv.RefAdvDataset)
    with mock.patch.object(refadv, 'LMUDataRoot', return_value=str(tmp_path)), \
            mock.patch.object(refadv.RefAdvDataset, 'DATASET_MD5', {'Ref-Adv-s': expected_md5}), \
            mock.patch.object(refadv, 'build_refadv_tsv', side_effect=fake_build) as build, \
            mock.patch.object(refadv, 'load', return_value='loaded') as load:
        with pytest.warns(UserWarning, match='TSV MD5 mismatch'):
            assert dataset.load_data('Ref-Adv-s') == 'loaded'

    build.assert_called_once_with(str(data_path))
    load.assert_called_once_with(str(data_path))
    assert hashlib.md5(data_path.read_bytes()).hexdigest() == expected_md5
