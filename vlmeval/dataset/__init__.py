from .config import dataset_URLs, dataset_md5_dict, img_root_map, DATASET_TYPE, abbr2full
from .image_dataset import TSVDataset, split_MMMU
from .video_dataset import TSVDatasetVideo, MMBenchVideo


def build_dataset(dataset_name, **kwargs):
    if dataset_name == 'MMBench-Video':
        return MMBenchVideo(dataset_name, **kwargs)
    else:
        return TSVDataset(dataset_name, **kwargs)


__all__ = [
    'dataset_URLs', 'dataset_md5_dict', 'img_root_map', 'DATASET_TYPE', 'abbr2full',
    'TSVDataset', 'TSVDatasetVideo', 'MMBenchVideo', 'split_MMMU', 'build_dataset'
]
