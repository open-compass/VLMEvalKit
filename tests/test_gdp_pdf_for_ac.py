import pandas as pd
import pytest

from vlmeval.dataset import DATASET_CLASSES, build_dataset
from vlmeval.dataset.gdp_pdf import GDPPDF, GDPPDFForAC
from vlmeval.smp import md5


@pytest.fixture
def ac_dataset(monkeypatch, tmp_path):
    monkeypatch.setenv('LMUData', str(tmp_path))
    directory = tmp_path / 'GDP_pdf_for_AC'
    directory.mkdir()
    (directory / 'document.pdf').write_bytes(b'%PDF-1.4\n')
    pd.DataFrame([
        dict(task_id='a', prompt='Task A', pdf_path='document.pdf', domain='law',
             **{'rubric - 1. criterion': 'pass'}),
    ]).to_parquet(directory / 'data.parquet')
    monkeypatch.setattr(GDPPDF, 'HF_PARQUET_MD5', md5(str(directory / 'data.parquet')))

    def unexpected_render(*args, **kwargs):
        pytest.fail('GDP_pdf_for_AC must not render PDF pages')

    monkeypatch.setattr(GDPPDF, 'dump_image', unexpected_render)
    return directory


def test_gdp_pdf_for_ac_registers_and_returns_pdf_file(ac_dataset):
    assert GDPPDFForAC in DATASET_CLASSES
    dataset = build_dataset('GDP_pdf_for_AC')
    assert type(dataset) is GDPPDFForAC
    pdf_path = str(ac_dataset / 'document.pdf')
    expected = [
        dict(type='text', value='Task A'),
        dict(type='file', value=pdf_path),
    ]
    assert dataset.build_prompt(0) == expected
    assert dataset.build_prompt(dataset.data.iloc[0]) == expected
