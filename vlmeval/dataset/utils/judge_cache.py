import hashlib
import logging
import os
import os.path as osp
import pickle
import re
import tempfile
import warnings

from vlmeval.smp import get_file_extension, get_intermediate_file_path
from vlmeval.utils import track_progress_rich

DEFAULT_FAIL_MSG = 'Failed to obtain answer via API.'
logger = logging.getLogger(__name__)


def _safe_artifact_name(value, default):
    raw = default if value is None else str(value)
    normalized = re.sub(r'[^0-9A-Za-z._-]+', '_', raw).strip('._-')
    if not normalized:
        normalized = default
    # Preserve readable names while preventing two different unsafe names from
    # collapsing onto the same cache path after sanitization.
    if normalized != raw:
        digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
        normalized = f'{normalized}_{digest}'
    return normalized


def _normalize_format(fmt):
    normalized = str(fmt).lower().lstrip('.')
    if not normalized or re.fullmatch(r'[0-9a-z]+', normalized) is None:
        raise ValueError(f'Invalid judge artifact format: {fmt!r}')
    return normalized


def get_judge_cache_file(eval_file, stage, model):
    stage_name = _safe_artifact_name(stage, 'eval')
    judge_name = _safe_artifact_name(model, 'exact_matching')
    return get_intermediate_file_path(eval_file, f'_judge_{stage_name}_{judge_name}_cache', 'pkl')


def get_judge_detail_file(eval_file, stage, model, fmt=None):
    if fmt is None:
        fmt = get_file_extension(eval_file)
    stage_name = _safe_artifact_name(stage, 'eval')
    judge_name = _safe_artifact_name(model, 'exact_matching')
    return get_intermediate_file_path(
        eval_file,
        f'_judge_{stage_name}_{judge_name}_detail',
        _normalize_format(fmt),
    )


def get_judge_score_file(eval_file, model, fmt):
    return get_intermediate_file_path(
        eval_file,
        f'_score_{_safe_artifact_name(model, "exact_matching")}',
        _normalize_format(fmt),
    )


def get_judge_named_legacy_cache_file(eval_file, model):
    """Return a pre-contract cache path only when the judge name is unambiguous."""
    judge_name = 'exact_matching' if model is None else str(model)
    if re.fullmatch(r'[0-9A-Za-z._-]+', judge_name) is None:
        return None
    return get_intermediate_file_path(eval_file, f'_{judge_name}', 'pkl')


def dump_judge_cache(cache, cache_file):
    """Atomically persist a sample-id-to-result judge cache."""
    if not isinstance(cache, dict):
        raise ValueError(
            f'Judge cache {cache_file} must be a dict, got {type(cache).__name__}.'
        )

    directory = osp.dirname(osp.abspath(cache_file))
    if not osp.isdir(directory):
        raise ValueError(f'Judge cache directory does not exist: {directory}')

    fd, temporary_file = tempfile.mkstemp(
        dir=directory,
        prefix=f'.{osp.basename(cache_file)}.',
        suffix='.tmp',
    )
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(cache, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_file, cache_file)
    except Exception:
        if osp.exists(temporary_file):
            os.unlink(temporary_file)
        raise


def _read_cache_file(path):
    try:
        with open(path, 'rb') as stream:
            data = pickle.load(stream)
    except Exception as err:
        message = f'Failed to load judge cache {path}: {type(err).__name__}: {err}'
        raise ValueError(message) from err

    if not isinstance(data, dict):
        raise ValueError(
            f'Judge cache {path} must contain a dict, got {type(data).__name__}.'
        )
    return data


def load_judge_cache(cache_file, legacy_file=None):
    """Load a cache, optionally migrating one judge-specific legacy cache."""
    if osp.exists(cache_file):
        return _read_cache_file(cache_file)

    if legacy_file is None or not osp.exists(legacy_file):
        return {}

    try:
        cache = _read_cache_file(legacy_file)
    except ValueError as err:
        warnings.warn(str(err), RuntimeWarning, stacklevel=2)
        return {}

    logger.info(
        'Migrating legacy judge cache %s to %s with %d entries.',
        legacy_file,
        cache_file,
        len(cache),
    )
    dump_judge_cache(cache, cache_file)
    return cache


def is_failed_judge_text(result, fail_msg=DEFAULT_FAIL_MSG):
    return (
        not isinstance(result, str)
        or not result.strip()
        or result.strip().lower() == 'fail'
        or fail_msg in result
    )


def run_cached_tasks(func, tasks, keys, cache_file, nproc=4, chunksize=None):
    """Run caller-selected tasks and merge their results into the cache."""
    if len(tasks) != len(keys):
        raise ValueError('tasks and keys must have the same length')

    cache = load_judge_cache(cache_file)
    if keys:
        track_progress_rich(
            func,
            tasks,
            nproc=nproc,
            chunksize=chunksize or nproc,
            keys=keys,
            save=cache_file,
            save_func=dump_judge_cache,
        )
        cache = load_judge_cache(cache_file)
    return cache
