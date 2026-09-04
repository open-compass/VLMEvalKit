import pandas as pd

from vlmeval.dataset.image_mcq import ImageMCQDataset


def test_mcq_tsv_preserves_none_option_text(tmp_path, monkeypatch):
    data_file = tmp_path / 'fixture.tsv'
    data_file.write_text(
        'index\timage_path\tquestion\tA\tB\thint\tanswer\n'
        '1\timage.png\tHow many objects?\tNone\tTwo\t\tA\n'
        '2\timage.png\tHow many objects?\t\tTwo\t\tB\n',
        encoding='utf-8',
    )
    monkeypatch.setattr('vlmeval.dataset.image_base.LMUDataRoot', lambda: str(tmp_path))

    dataset = ImageMCQDataset.__new__(ImageMCQDataset)
    dataset.dataset_name = 'fixture'
    dataset.meta_only = True
    dataset.data = dataset.prepare_tsv('fixture.tsv')

    assert dataset.data.loc[0, 'A'] == 'None'
    assert pd.isna(dataset.data.loc[1, 'A'])
    assert pd.isna(dataset.data.loc[0, 'hint'])

    prompt = dataset.build_prompt(dataset.data.iloc[0])[-1]['value']
    assert 'A. None' in prompt
