from multiprocessing import Pool
import os
from typing import Callable, Iterable, Sized

from rich.progress import (BarColumn, MofNCompleteColumn, Progress, Task,
                           TaskProgressColumn, TextColumn, TimeRemainingColumn)
from rich.text import Text
import os.path as osp
import portalocker
from ..smp import load, dump


def track_progress_rich(
        func: Callable,
        tasks: Iterable = tuple(),
        nproc: int = 1,
        save=None,
        keys=None,
        **kwargs) -> list:

    from concurrent.futures import ThreadPoolExecutor
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
    assert nproc > 0, 'nproc must be a positive number'
    res = load(save) if save is not None else {}
    results = []

    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []

        for inputs in tasks:
            if not isinstance(inputs, (tuple, list, dict)):
                inputs = (inputs, )
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        for i, future in enumerate(tqdm(futures)):
            tmp = future.result()
            if keys is not None:
                res[keys[i]] = tmp
                if i % 10 == 0 and save is not None:
                    dump(res, save)
            results.append(tmp)

    if save is not None:
        dump(res, save)
    return results
