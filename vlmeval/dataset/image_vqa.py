from .image_base import ImageBaseDataset
from ..smp import *


class ImageVQADataset(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'OCRVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TEST.tsv',
        'OCRVQA_TESTCORE': 'https://opencompass.openxlab.space/utils/VLMEval/OCRVQA_TESTCORE.tsv',
        'TextVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/TextVQA_VAL.tsv',
        'DocVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_VAL.tsv',
        'DocVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/DocVQA_TEST.tsv',
        'InfoVQA_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_VAL.tsv',
        'InfoVQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/InfoVQA_TEST.tsv',
        'ChartQA_TEST': 'https://opencompass.openxlab.space/utils/VLMEval/ChartQA_TEST.tsv',
    }

    DATASET_MD5 = {
        'OCRVQA_TEST': 'ca46a6d74b403e9d6c0b670f6fc00db9',
        'OCRVQA_TESTCORE': 'c5239fe77db8bdc1f2ad8e55e0d1fe97',
        'TextVQA_VAL': 'b233b31f551bbf4056f2f955da3a92cd',
        'DocVQA_VAL': 'd5ee77e1926ff10690d469c56b73eabf',
        'DocVQA_TEST': '6a2f28cac26ef2d3447374e8c6f6c8e9',
        'InfoVQA_VAL': '2342e9c225222f0ef4dec545ebb126fe',
        'InfoVQA_TEST': 'df535bf51b88dc9718252c34131a6227',
        'ChartQA_TEST': 'c902e0aa9be5582a7aad6dcf52734b42',
    }

    def post_build(self, dataset):
        if 'OCRVQA' in dataset:
            self.img_root = osp.join(LMUDataRoot(), 'images', 'OCRVQA')

    def build_prompt(self, line):
        msgs = super().build_prompt(line)
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += '\nAnswer the question using a single word or phrase.'
        return msgs


class OCRBench(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {'OCRBench': 'https://opencompass.openxlab.space/utils/VLMEval/OCRBench.tsv'}
    DATASET_MD5 = {'OCRBench': 'e953d98a987cc6e26ef717b61260b778'}


class MathVista(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {'MathVista_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/MathVista_MINI.tsv'}
    DATASET_MD5 = {'MathVista_MINI': 'f199b98e178e5a2a20e7048f5dcb0464'}


class LLaVABench(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {'LLaVABench': 'https://opencompass.openxlab.space/utils/VLMEval/LLaVABench.tsv'}
    DATASET_MD5 = {'LLaVABench': 'd382a093f749a697820d3dadd61c8428'}


class MMVet(ImageBaseDataset):

    TYPE = 'VQA'
    DATASET_URL = {'MMVet': 'https://opencompass.openxlab.space/utils/VLMEval/MMVet.tsv'}
    DATASET_MD5 = {'MMVet': '748aa6d4aa9d4de798306a63718455e3'}


class CustomVQADataset(ImageBaseDataset):

    TYPE = 'VQA'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None):
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def evaluate(self, eval_file, **judge_kwargs):
        raise NotImplementedError
