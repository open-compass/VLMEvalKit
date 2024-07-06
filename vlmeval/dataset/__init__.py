import warnings

from .image_caption import ImageCaptionDataset
from .image_yorn import ImageYORNDataset
from .image_mcq import ImageMCQDataset, MMMUDataset, CustomMCQDataset
from .image_vqa import ImageVQADataset, OCRBench, MathVista, LLaVABench, MMVet, CustomVQADataset
from .mmbench_video import MMBenchVideo
from .utils.judge_util import build_judge
from ..smp import *


def DATASET_TYPE(dataset):
    raise NotImplementedError


img_root_map = {}


def build_dataset(dataset_name, **kwargs):
    if dataset_name == 'MMBench-Video':
        return MMBenchVideo(dataset_name, **kwargs)
    datasets = [
        ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, ImageVQADataset,
        MMMUDataset, OCRBench, MathVista, LLaVABench, MMVet
    ]
    for cls in datasets:
        if dataset_name in cls.supported_datasets():
            return cls(dataset=dataset_name, **kwargs)

    warnings.warn(f'Dataset {dataset_name} is not officially supported. ')

    data_file = osp.join(LMUDataRoot(), f'{dataset_name}.tsv')
    assert osp.exists(data_file), f'Data file {data_file} does not exist.'
    data = load(data_file)
    if 'A' in data and 'B' in data:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
        return CustomMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


__all__ = [
    'MMBenchVideo', 'ImageYORNDataset', 'ImageMCQDataset', 'MMMUDataset',
    'ImageCaptionDataset', 'ImageVQADataset', 'OCRBench', 'MathVista', 'LLaVABench', 'MMVet',
    'CustomMCQDataset', 'CustomVQADataset', 'build_dataset', 'build_judge'
]
