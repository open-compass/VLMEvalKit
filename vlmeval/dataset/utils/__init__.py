from .judge_util import build_judge, DEBUG_MESSAGE
from .multiple_choice import extract_answer_from_item, prefetch_answer
from .vqa_eval import levenshtein_distance


# Add your dataset class here
from .mmlongbench import MMLongBench
from .vcr import VCRDataset
dataset_list = [
    'MMLongBench',
    'VCRDataset',
]


__all__ = [
    'build_judge',
    'extract_answer_from_item',
    'prefetch_answer',
    'levenshtein_distance',
    'DEBUG_MESSAGE'
] + dataset_list
