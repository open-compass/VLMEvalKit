import base64
import csv
import io

import pandas as pd
import pytest
from PIL import Image

from vlmeval.dataset.image_mcq import MMMU_SOURCE_TAG, MMMUDataset


@pytest.mark.parametrize('regenerated', [False, True])
def test_mmmu_tsv_load_and_dump_preserves_text_and_expected_image_mode(monkeypatch, tmp_path, regenerated):
    name = f'MMMU_TEST_{MMMU_SOURCE_TAG}' if regenerated else 'MMMU_TEST'
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('LMURoot', str(tmp_path))
    monkeypatch.setitem(MMMUDataset.DATASET_MD5, name, None)

    image = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
    image.putpixel((1, 1), (0, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    question = 'Solve $\\frac{1}{2}$:\n<image 1>\tchoose an answer'
    with (tmp_path / f'{name}.tsv').open('w', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=['index', 'question', 'image', 'A', 'B', 'C'],
                                delimiter='\t', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerow({
            'index': 1,
            'question': question,
            'image': encoded,
            'A': 'first',
            'B': 'None of the answers are correct',
            'C': '',
        })

    dataset = MMMUDataset(dataset=name)
    row = dataset.data.iloc[0]
    assert row['question'] == question
    assert row['B'] == 'None of the answers are correct'
    assert pd.isna(row['C'])

    [path] = dataset.dump_image(row)
    with Image.open(path) as dumped:
        assert dumped.mode == ('RGBA' if regenerated else 'RGB')
        if regenerated:
            assert dumped.getpixel((0, 0))[3] == 0
            assert dumped.getpixel((1, 1))[3] == 255
