from vlmeval import *
from .image_base import ImageBaseDataset


class ImageYORNDataset(ImageBaseDataset):

    TYPE = 'YORN'

    DATASET_URL = {
        'MME': 'https://opencompass.openxlab.space/utils/VLMEval/MME.tsv',
        'HallusionBench': 'https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv',
        'POPE': 'https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv',
    }

    DATASET_MD5 = {
        'MME': 'b36b43c3f09801f5d368627fb92187c3',
        'HallusionBench': '0c23ac0dc9ef46832d7a24504f2a0c7c',
        'POPE': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
    }
