from .image_yorn import ImageYORNDataset
from .image_mcq import ImageMCQDataset, MMMUDataset
from .mmbench_video import MMBenchVideo


# def build_dataset(dataset_name, **kwargs):
#     if dataset_name == 'MMBench-Video':
#         return MMBenchVideo(dataset_name, **kwargs)
#     else:
#         return TSVDataset(dataset_name, **kwargs)


__all__ = [
    'MMBenchVideo', 'ImageYORNDataset', 'ImageMCQDataset', 'MMMUDataset'
]
