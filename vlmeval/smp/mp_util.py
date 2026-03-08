import multiprocessing as mp
from multiprocessing import Pool
from typing import Callable, Iterable, Sized
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os.path as osp
import time
from . import load, dump


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
        desc='',
        **kwargs) -> list:

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
    if nproc is None:
        # For safety reasons, we don't want the nproc to be too large
        nproc = min(mp.cpu_count(), 32)
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = [None for _ in range(len(tasks))]

    executor_type = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    counter, save_n_iter = 0, 10
    with executor_type(max_workers=nproc) as executor:
        futures = []

        for idx, inputs in enumerate(tasks):
            future = executor.submit(new_func, idx, func, inputs)
            futures.append(future)

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            idx, result = future.result()
            results[idx] = result
            if keys is not None:
                res[keys[idx]] = result
                counter += 1
                if counter % save_n_iter == 0 or len(futures) - counter < 2 * save_n_iter:
                    dump(res, save)

    if save is not None:
        dump(res, save)
    return results
