from vlmeval import *

from .image_base import ImageBaseDataset

dataset_URLs = {
    'MME': 'https://opencompass.openxlab.space/utils/VLMEval/MME.tsv',
    'HallusionBench': 'https://opencompass.openxlab.space/utils/VLMEval/HallusionBench.tsv',
    'POPE': 'https://opencompass.openxlab.space/utils/VLMEval/POPE.tsv',
}

dataset_md5_dict = {
    'MME': 'b36b43c3f09801f5d368627fb92187c3',
    'HallusionBench': '0c23ac0dc9ef46832d7a24504f2a0c7c',
    'POPE': 'c12f5acb142f2ef1f85a26ba2fbe41d5',
}


class ImageYORNDataset(ImageBaseDataset):

    TYPE = 'YORN'

    def build_prompt(self, line, dataset=None):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))

        return msgs
