import warnings

from .image_base import img_root_map
from .image_caption import ImageCaptionDataset
from .image_yorn import ImageYORNDataset
from .image_mcq import ImageMCQDataset, MMMUDataset, CustomMCQDataset
from .image_vqa import ImageVQADataset, OCRBench, MathVista, LLaVABench, MMVet, CustomVQADataset
from .mmbench_video import MMBenchVideo
from .utils import build_judge, extract_answer_from_item, prefetch_answer, DEBUG_MESSAGE
from ..smp import *

DATASET_CLASSES = [
    ImageCaptionDataset, ImageYORNDataset, ImageMCQDataset, MMMUDataset,
    CustomMCQDataset, ImageVQADataset, OCRBench, MathVista, LLaVABench, MMVet,
    CustomVQADataset, MMBenchVideo
]


def DATASET_TYPE(dataset):
    for cls in DATASET_CLASSES:
        if dataset in cls.supported_datasets():
            return cls.TYPE


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
    if not osp.exists(data_file):
        warnings.warn(f'Data file {data_file} does not exist. Dataset building failed. ')
        return None

    data = load(data_file)
    if 'question' not in [x.lower() for x in data.columns]:
        warnings.warn(f'Data file {data_file} does not have a `question` column. Dataset building failed. ')
        return None

    if 'A' in data and 'B' in data:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom MCQ dataset. ')
        return CustomMCQDataset(dataset=dataset_name, **kwargs)
    else:
        warnings.warn(f'Will assume unsupported dataset {dataset_name} as a Custom VQA dataset. ')
        return CustomVQADataset(dataset=dataset_name, **kwargs)


__all__ = [
    'MMBenchVideo', 'ImageYORNDataset', 'ImageMCQDataset', 'MMMUDataset',
    'ImageCaptionDataset', 'ImageVQADataset', 'OCRBench', 'MathVista', 'LLaVABench', 'MMVet',
    'CustomMCQDataset', 'CustomVQADataset', 'build_dataset', 'img_root_map',
    'build_judge', 'extract_answer_from_item', 'prefetch_answer', 'DEBUG_MESSAGE'
]
