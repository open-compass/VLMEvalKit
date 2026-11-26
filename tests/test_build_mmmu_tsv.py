import base64
import csv
import io

import pandas as pd
import pytest
from PIL import Image

import vlmeval.dataset.image_base as image_base
import vlmeval.smp.file as smp_file
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


@pytest.mark.parametrize('regenerated', [False, True])
def test_large_mmmu_tsv_localizes_images(monkeypatch, tmp_path, regenerated):
    name = f'MMMU_TEST_{MMMU_SOURCE_TAG}' if regenerated else 'MMMU_TEST'
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('LMURoot', str(tmp_path))
    monkeypatch.setitem(MMMUDataset.DATASET_MD5, name, None)
    monkeypatch.setattr(image_base, 'file_size', lambda *args: 2)

    original_pool = smp_file.mp.Pool
    monkeypatch.setattr(smp_file.mp, 'Pool', lambda nproc: original_pool(1))

    image = Image.new('RGBA', (2, 2), (0, 0, 0, 0))
    image.putpixel((1, 1), (255, 0, 0, 255))
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    row = {
        'index': 1,
        'question': 'What is in <image 1>?',
        'image': base64.b64encode(buffer.getvalue()).decode('ascii'),
        'A': 'red',
        'B': 'blue',
    }
    pd.DataFrame([row]).to_csv(tmp_path / f'{name}.tsv', sep='\t', index=False)

    if regenerated:
        # A legacy localization must not be reused for alpha-preserving output.
        (tmp_path / f'{name}_local.tsv').write_text('stale cache', encoding='utf-8')

    dataset = MMMUDataset(dataset=name)
    suffix = '_local_alpha.tsv' if regenerated else '_local.tsv'
    assert (tmp_path / f'{name}{suffix}').exists()
    assert dataset.meta_only
    messages = dataset.build_prompt(0)
    [path] = [message['value'] for message in messages if message['type'] == 'image']
    with Image.open(path) as localized:
        assert localized.mode == ('RGBA' if regenerated else 'RGB')
        if regenerated:
            assert localized.getpixel((0, 0))[3] == 0
            assert localized.getpixel((1, 1)) == (255, 0, 0, 255)


@pytest.mark.parametrize('mode', ['RGBA', 'P'])
def test_localize_preserves_modes_and_image_references(monkeypatch, tmp_path, mode):
    monkeypatch.setenv('LMUData', str(tmp_path))
    image = Image.new(mode, (2, 2))
    if mode == 'P':
        image.putpalette([0, 0, 0] * 256)
        image.info['transparency'] = 0
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG' if mode == 'CMYK' else 'PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    data = pd.DataFrame({'index': [1, 2], 'image': [encoded, '1']})

    localized = smp_file.localize_df(data, 'mode_test', nproc=1, convert_rgb=False)
    assert localized['image_path'][0] == localized['image_path'][1]
    with Image.open(localized['image_path'][0]) as result:
        assert result.mode == mode
        if mode == 'P':
            assert result.convert('RGBA').getpixel((0, 0))[3] == 0
