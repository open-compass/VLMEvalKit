from vlmeval.smp import *
from .image_base import ImageBaseDataset


class TextBaseDataset(ImageBaseDataset):
    MODALITY = 'TEXT'

    def __init__(self, dataset='MMBench', **kwargs):
        self.dataset_name = dataset

        data = self.load_data(dataset)
        data['index'] = [str(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def prepare_tsv(self, url, file_md5=None):
        if osp.isfile(url) and osp.exists(url):
            return load(url)
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        print('data_path: ', data_path)
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
        return load(data_path)

    def dump_image(self, line):
        return []

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL[dataset]
        file_md5 = self.DATASET_MD5[dataset]
        return self.prepare_tsv(url, file_md5)

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        msgs = []
        msgs.append(dict(type='text', value=question))
        return msgs
