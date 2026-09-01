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


class JudgeCacheError(ValueError):
    """Raised when a judge cache does not satisfy the on-disk contract."""


def _normalize_artifact_component(value, default):
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


def normalize_judge_name(model):
    return _normalize_artifact_component(model, 'exact_matching')


def normalize_stage_name(stage):
    return _normalize_artifact_component(stage, 'eval')


def _normalize_format(fmt):
    normalized = str(fmt).lower().lstrip('.')
    if not normalized or re.fullmatch(r'[0-9a-z]+', normalized) is None:
        raise ValueError(f'Invalid judge artifact format: {fmt!r}')
    return normalized


def get_judge_cache_file(eval_file, stage, model):
    stage_name = normalize_stage_name(stage)
    judge_name = normalize_judge_name(model)
    return get_intermediate_file_path(eval_file, f'_judge_{stage_name}_{judge_name}_cache', 'pkl')


def get_judge_detail_file(eval_file, stage, model, fmt=None):
    if fmt is None:
        fmt = get_file_extension(eval_file)
    stage_name = normalize_stage_name(stage)
    judge_name = normalize_judge_name(model)
    return get_intermediate_file_path(
        eval_file,
        f'_judge_{stage_name}_{judge_name}_detail',
        _normalize_format(fmt),
    )


def get_judge_score_file(eval_file, model, fmt):
    return get_intermediate_file_path(
        eval_file,
        f'_score_{normalize_judge_name(model)}',
        _normalize_format(fmt),
    )


def dump_judge_cache(cache, cache_file):
    """Atomically persist a sample-id-to-result judge cache."""
    if not isinstance(cache, dict):
        raise JudgeCacheError(
            f'Judge cache {cache_file} must be a dict, got {type(cache).__name__}.'
        )

    directory = osp.dirname(osp.abspath(cache_file))
    if not osp.isdir(directory):
        raise JudgeCacheError(f'Judge cache directory does not exist: {directory}')

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


def _read_cache_file(path, strict):
    try:
        with open(path, 'rb') as stream:
            data = pickle.load(stream)
    except Exception as err:
        message = f'Failed to load judge cache {path}: {type(err).__name__}: {err}'
        if strict:
            raise JudgeCacheError(message) from err
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return None

    if not isinstance(data, dict):
        message = f'Judge cache {path} must contain a dict, got {type(data).__name__}.'
        if strict:
            raise JudgeCacheError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return None
    return data


def has_judge_failure(result, fail_msg=DEFAULT_FAIL_MSG):
    if result is None:
        return True
    if isinstance(result, str):
        return fail_msg in result
    if isinstance(result, dict):
        return any(has_judge_failure(v, fail_msg=fail_msg) for v in result.values())
    if isinstance(result, (list, tuple)):
        return any(has_judge_failure(v, fail_msg=fail_msg) for v in result)
    return False


def is_failed_result(result, fail_msg=DEFAULT_FAIL_MSG, failure_fn=None):
    if failure_fn is not None:
        return failure_fn(result)
    return has_judge_failure(result, fail_msg=fail_msg)


def load_judge_cache(cache_file, legacy_files=None, ignored_legacy_files=None):
    for path in ignored_legacy_files or []:
        if osp.exists(path):
            warnings.warn(
                f'Ignoring untrusted legacy judge cache {path}; its judge source cannot be verified.',
                RuntimeWarning,
                stacklevel=2,
            )

    if osp.exists(cache_file):
        return _read_cache_file(cache_file, strict=True)

    cache = {}
    migrated = False
    for path in legacy_files or []:
        if not osp.exists(path):
            continue
        data = _read_cache_file(path, strict=False)
        if data is None:
            continue
        cache.update(data)
        migrated = True
        logger.info(
            'Migrating legacy judge cache %s to %s with %d entries.',
            path,
            cache_file,
            len(data),
        )
    if migrated:
        dump_judge_cache(cache, cache_file)
    return cache


def filter_cached_tasks(tasks, keys, cache, fail_msg=DEFAULT_FAIL_MSG, failure_fn=None):
    pending_tasks, pending_keys = [], []
    for task, key in zip(tasks, keys):
        if key not in cache or is_failed_result(cache[key], fail_msg=fail_msg, failure_fn=failure_fn):
            pending_tasks.append(task)
            pending_keys.append(key)
    return pending_tasks, pending_keys


def run_cached_tasks(func, tasks, keys, cache_file, nproc=4, chunksize=None, fail_msg=DEFAULT_FAIL_MSG,
                     legacy_files=None, ignored_legacy_files=None, failure_fn=None, **kwargs):
    cache = load_judge_cache(
        cache_file,
        legacy_files=legacy_files,
        ignored_legacy_files=ignored_legacy_files,
    )
    pending_tasks, pending_keys = filter_cached_tasks(
        tasks,
        keys,
        cache,
        fail_msg=fail_msg,
        failure_fn=failure_fn,
    )
    if pending_keys:
        track_progress_rich(
            func,
            pending_tasks,
            nproc=nproc,
            chunksize=chunksize or nproc,
            keys=pending_keys,
            save=cache_file,
            save_func=dump_judge_cache,
            **kwargs,
        )
        cache = load_judge_cache(cache_file)
    return cache
