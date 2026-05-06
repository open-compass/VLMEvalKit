import os.path as osp
import re

from vlmeval.smp import dump, get_file_extension, get_intermediate_file_path, load
from vlmeval.utils import track_progress_rich

DEFAULT_FAIL_MSG = 'Failed to obtain answer via API.'


def normalize_judge_name(model):
    if model is None:
        model = 'exact_matching'
    return re.sub(r'[^0-9A-Za-z._-]+', '_', str(model)).strip('_')


def get_judge_cache_file(eval_file, stage, model):
    return get_intermediate_file_path(eval_file, f'_judge_{stage}_{normalize_judge_name(model)}', 'pkl')


def get_judge_detail_file(eval_file, stage, model, fmt=None):
    if fmt is None:
        fmt = get_file_extension(eval_file)
    return get_intermediate_file_path(eval_file, f'_judge_{stage}_{normalize_judge_name(model)}', fmt)


def get_judge_score_file(eval_file, model, fmt):
    return get_intermediate_file_path(eval_file, f'_score_{normalize_judge_name(model)}', fmt)


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


def load_judge_cache(cache_file, legacy_files=None):
    cache = {}
    for path in legacy_files or []:
        if osp.exists(path):
            data = load(path)
            if isinstance(data, dict):
                cache.update(data)
    if osp.exists(cache_file):
        data = load(cache_file)
        if isinstance(data, dict):
            cache.update(data)
    if cache and not osp.exists(cache_file):
        dump(cache, cache_file)
    return cache


def filter_cached_tasks(tasks, keys, cache, fail_msg=DEFAULT_FAIL_MSG, failure_fn=None):
    pending_tasks, pending_keys = [], []
    for task, key in zip(tasks, keys):
        if key not in cache or is_failed_result(cache[key], fail_msg=fail_msg, failure_fn=failure_fn):
            pending_tasks.append(task)
            pending_keys.append(key)
    return pending_tasks, pending_keys


def run_cached_tasks(func, tasks, keys, cache_file, nproc=4, chunksize=None, fail_msg=DEFAULT_FAIL_MSG,
                     legacy_files=None, failure_fn=None, **kwargs):
    cache = load_judge_cache(cache_file, legacy_files=legacy_files)
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
            **kwargs,
        )
        cache = load_judge_cache(cache_file, legacy_files=legacy_files)
    return cache
