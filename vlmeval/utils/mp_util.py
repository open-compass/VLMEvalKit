import asyncio
import multiprocessing as mp
import os.path as osp
import time
from pathlib import Path
from typing import Callable, Iterable

from ..smp import dump, load


def cpu_count():
    # Handle K8s LXCFS setting.
    period = Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    quota = Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
    try:
        if period.exists() and quota.exists():
            return int(quota.read_text()) // int(period.read_text())
    except Exception:
        pass

    import os
    return os.cpu_count()


async def async_wait_process(process: mp.Process, timeout=None, poll_interval=0.1):
    """Wait for a child process without blocking the asyncio event loop."""
    start = time.monotonic()
    while process.is_alive():
        if timeout is not None and time.monotonic() - start >= timeout:
            return process.exitcode
        await asyncio.sleep(poll_interval)

    process.join(timeout=0)
    return process.exitcode


async def async_recv_process_message(conn, process: mp.Process, process_name: str,
                                     poll_interval=0.1):
    """Receive one pipe message without blocking the asyncio event loop."""
    while True:
        if conn.poll():
            try:
                return conn.recv()
            except EOFError:
                process.join(timeout=0)
                break

        if not process.is_alive():
            process.join(timeout=0)
            if conn.poll():
                continue
            break

        await asyncio.sleep(poll_interval)

    raise RuntimeError(f"{process_name} exited unexpectedly with code {process.exitcode}")


def terminate_processes(processes, name='process', timeout=5, logger=None):
    """Terminate child processes with SIGTERM first, then SIGKILL."""
    alive = [p for p in processes if p is not None and p.is_alive()]
    for p in alive:
        if logger is not None:
            logger.debug(f"Terminating {name} (pid={p.pid})")
        p.terminate()
    for p in alive:
        p.join(timeout=timeout)
        if p.is_alive():
            if logger is not None:
                logger.debug(f"Force killing {name} (pid={p.pid})")
            p.kill()
            p.join(timeout=timeout)


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

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

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
