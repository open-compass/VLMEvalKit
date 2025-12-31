from pathlib import Path
from typing import Callable, Iterable

import os.path as osp
from ..smp import load, dump


def cpu_count():
    # Handle K8s LXCFS setting.
    period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    quota = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    try:
        if period.exists() and quota.exists():
            return int(quota.read_text()) // int(period.read_text())
    except:
        pass

    import os
    return os.cpu_count()


def new_func(idx: int, func, inputs):
    if not isinstance(inputs, (tuple, list, dict)):
        inputs = (inputs, )
    if isinstance(inputs, dict):
        return idx, func(**inputs)
    else:
        return idx, func(*inputs)


def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int | None = 1,
        save=None,
        keys=None,
        use_process: bool = False,
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    if save is not None:
        assert osp.exists(osp.dirname(save)) or osp.dirname(save) == ''
        if not osp.exists(save):
            dump({}, save)
    if keys is not None:
        assert len(keys) == len(tasks)
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(
            f'tasks must be an iterable object, but got {type(tasks)}')
    assert nproc is None or nproc > 0, 'nproc must be a positive number'
    nproc = nproc or cpu_count()
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    executor_type = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    with executor_type(max_workers=nproc) as executor:
        futures = []

        for idx, inputs in enumerate(tasks):
            future = executor.submit(new_func, idx, func, inputs)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, result = future.result()
            results[idx] = result
            if keys is not None:
                res[keys[idx]] = result
                dump(res, save)

    if save is not None:
        dump(res, save)
    return results
