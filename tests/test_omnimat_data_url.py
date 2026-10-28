import pandas as pd

from vlmeval.dataset import omnimat as omnimat_module
from vlmeval.dataset.omnimat import OmniMat

EXPECTED_URLS = {
    'OmniMat_QA': (
        'https://huggingface.co/datasets/'
        'Summer12138/OmniMat1K-VLMEvalKit/resolve/main/'
        'vlmevalkit/OmniMat_QA.tsv'
    ),
    'OmniMat_CAL': (
        'https://huggingface.co/datasets/'
        'Summer12138/OmniMat1K-VLMEvalKit/resolve/main/'
        'vlmevalkit/OmniMat_CAL.tsv'
    ),
}


def test_omnimat_uses_published_tsv_urls():
    assert OmniMat.DATASET_URL == EXPECTED_URLS


def test_omnimat_downloads_published_tsv_when_local_data_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(omnimat_module, 'LMUDataRoot', lambda: str(tmp_path))
    dataset = object.__new__(OmniMat)
    downloaded = pd.DataFrame(
        [{'index': '1', 'category_id': '1', 'id': '2', 'question': 'test'}]
    )
    called = []
    dataset.prepare_tsv = lambda url: called.append(url) or downloaded.copy()

    data = dataset.load_data('OmniMat_QA')

    assert called == [EXPECTED_URLS['OmniMat_QA']]
    assert data.iloc[0]['category_id'] == '01'
    assert data.iloc[0]['id'] == '002'
