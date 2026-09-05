from .image_vqa import CountBenchQA


class PixmoCountDataset(CountBenchQA):
    TYPE = 'VQA'
    DATASET_URL = {
        'PixmoCount': ''
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['PixmoCount']
