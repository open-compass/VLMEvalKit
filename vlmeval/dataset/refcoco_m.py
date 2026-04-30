from .refcoco import RefCOCODataset


class RefCOCOMDataset(RefCOCODataset):
    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'
    DATASET_URL = {
        'RefCOCO-M': ''
    }
    DATASET_MD5 = {}

    @classmethod
    def supported_datasets(cls):
        return ['RefCOCO-M']
