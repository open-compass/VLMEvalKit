import pandas as pd

from vlmeval.dataset.image_base import img_root_map
from vlmeval.dataset.image_vqa import MATHVISTA_SOURCE_TAG, MathVista


def test_mathvista_original_and_regenerated_dataset_entries():
    assert MATHVISTA_SOURCE_TAG == '2b6ad6'
    assert MathVista.DATASET_URL['MathVista_MINI'].startswith('https://')

    dataset_name = f'MathVista_MINI_{MATHVISTA_SOURCE_TAG}'
    assert MathVista.DATASET_URL[dataset_name].endswith('MathVista_MINI_2b6ad6.tsv')
    assert MathVista.DATASET_MD5[dataset_name] == 'd5f6f330680fc41d8f8c583838a2ac41'
    # A separate image root prevents the regenerated images from reusing stale
    # JPEG dumps created by the original hosted TSV.
    assert img_root_map(dataset_name) == dataset_name


def test_mathvista_regenerated_variant_preserves_alpha(monkeypatch):
    data = pd.DataFrame({
        'index': ['1'],
        'image': ['x' * 65],
        'question': ['question'],
    })
    monkeypatch.setattr(MathVista, 'load_data', lambda self, dataset: data.copy())

    original = MathVista('MathVista_MINI')
    regenerated = MathVista(f'MathVista_MINI_{MATHVISTA_SOURCE_TAG}')

    assert original.convert_to_rgb is True
    assert regenerated.convert_to_rgb is False
