import json
from numbers import Integral

import pandas as pd


def _normalize_requested_index(value):
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise ValueError('dataset indices must be integers or strings')
    if isinstance(value, str) and not value:
        raise ValueError('dataset indices must not be empty strings')
    return str(value)


def parse_data_indices(raw_value):
    """Parse the JSON mapping accepted by ``--data-indices``."""
    if raw_value is None:
        return {}
    try:
        config = json.loads(raw_value)
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError('Unable to parse --data-indices as a JSON object') from e

    if not isinstance(config, dict):
        raise ValueError('--data-indices must be a JSON object keyed by dataset name')

    parsed = {}
    for dataset_name, indices in config.items():
        if not isinstance(dataset_name, str) or not dataset_name:
            raise ValueError('--data-indices keys must be non-empty dataset names')
        if not isinstance(indices, list) or not indices:
            raise ValueError(f'--data-indices[{dataset_name!r}] must be a non-empty JSON list')

        normalized = [_normalize_requested_index(index) for index in indices]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f'--data-indices[{dataset_name!r}] contains duplicate indices')
        parsed[dataset_name] = normalized
    return parsed


def validate_data_indices(data_indices, dataset_names):
    unknown = sorted(set(data_indices) - set(dataset_names))
    if unknown:
        raise ValueError(
            '--data-indices contains datasets that are not selected by --data/--config: '
            + ', '.join(unknown)
        )


def filter_frame_by_indices(frame, requested_indices, *, require_all=True):
    """Return rows matching requested indices, preserving their source order."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError('dataset data must be a pandas DataFrame')
    if 'index' not in frame:
        raise ValueError('dataset data must contain an index column')

    normalized = frame['index'].map(str)
    requested = set(requested_indices)
    if require_all:
        available = set(normalized)
        missing = [index for index in requested_indices if index not in available]
        if missing:
            raise ValueError('requested dataset indices were not found: ' + ', '.join(missing))

    return frame.loc[normalized.isin(requested)].copy().reset_index(drop=True)


def subset_dataset(dataset, requested_indices):
    """Restrict a dataset object to selected sample indices in-place."""
    original_size = len(dataset.data)
    dataset.data = filter_frame_by_indices(dataset.data, requested_indices)

    if hasattr(dataset, 'videos') and 'video' in dataset.data:
        selected_videos = set(dataset.data['video'])
        dataset.videos = [video for video in dataset.videos if video in selected_videos]

    return original_size, len(dataset.data)
