import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset


class _DummyImageDataset(ImageBaseDataset):

    def load_data(self, dataset):
        return pd.DataFrame({'index': ['0'], 'question': ['q']})

    def evaluate(self, eval_file, **judge_kwargs):
        return {}


def test_image_base_dataset_defaults_to_preserving_alpha(monkeypatch, tmp_path):
    monkeypatch.setenv('LMUData', str(tmp_path))
    monkeypatch.setenv('LMURoot', str(tmp_path))

    dataset = _DummyImageDataset(dataset='Dummy')

    assert dataset.convert_to_rgb is False
