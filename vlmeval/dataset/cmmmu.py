from .image_base import ImageBaseDataset
import re
import random
from collections import Counter


class CMMMU(ImageBaseDataset):

    DATASET_URL = {
        'CMMMU_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/CMMMU_TEST.tsv',
    }

    DATASET_MD5 = {
        'CMMMU_TEST': '521afc0f3bf341e6654327792781644d',
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        pass

    def build_prompt(self, line):

        pass
