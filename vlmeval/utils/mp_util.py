from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Optional, List, Dict, Any
import os.path as osp
import os
import portalocker
from rich.progress import Progress
from ..smp import load, dump

from tqdm import tqdm

# 保存原始的 __init__ 方法
original_init_func = tqdm.__init__

def custom_tqdm_init(self, *args, **kwargs):
    kwargs.setdefault('ncols', 135) # 修改 tqdm 的默认进度条长度
    # kwargs.setdefault('disable', True)
    original_init_func(self, *args, **kwargs)

# 用自定义的方法替换 tqdm 的 __init__ 方法
tqdm.__init__ = custom_tqdm_init

FAIL_MSG = 'Failed to obtain answer via API.'

def track_progress_rich(
    func: Callable,
    tasks: Iterable,
    nproc: int = 1,
    save: Optional[str] = None,
    keys: Optional[List[str]] = None,
    use_multiprocessing: bool = False,
    resume: bool = True,
    **kwargs
) -> List[Any]:
    """
    Execute tasks in parallel with rich progress tracking and optional saving.
    
    Args:
        func: Function to execute
        tasks: Iterable of task inputs
        nproc: Number of parallel processes/threads
        save: Path to save results (optional)
        keys: Keys for saving results (optional)
        use_multiprocessing: Use ProcessPoolExecutor instead of ThreadPoolExecutor
        resume: Resume from saved results if available
        **kwargs: Additional arguments for the function
    
    Returns:
        List of results
    """
    # Validation
    if not callable(func):
        raise TypeError('func must be a callable object')
    if not isinstance(tasks, Iterable):
        raise TypeError(f'tasks must be an iterable object, but got {type(tasks)}')
    if nproc <= 0:
        raise ValueError('nproc must be a positive number')
    
    tasks = list(tasks)  # Convert to list to get length
    
    if keys is not None and len(keys) != len(tasks):
        raise ValueError('Length of keys must match length of tasks')
    
    # Initialize save file and results
    saved_results = {}
    if save is not None:
        save_dir = osp.dirname(save)
        if save_dir and not osp.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if osp.exists(save) and resume:
            try:
                saved_results = load(save)
                print(f"Loaded {len(saved_results)} results from {save}")
                saved_results = {k: v for k, v in saved_results.items() if FAIL_MSG.strip() not in v.strip()}
                print(f"Filtered {len(saved_results)} results from {save}")
            except Exception as e:
                print(f"Warning: Could not load saved results: {e}")
                saved_results = {}
        elif not osp.exists(save):
            dump({}, save)
    
    # Determine which tasks need to be executed
    tasks_to_execute = []
    task_indices = []
    results = [None] * len(tasks)
    
    for i, task in enumerate(tasks):
        key = keys[i] if keys else str(i)
        if resume and key in saved_results:
            results[i] = saved_results[key]
        else:
            tasks_to_execute.append(task)
            task_indices.append(i)
    
    if not tasks_to_execute:
        print("All tasks already completed!")
        return results
    
    # Choose executor type
    if use_multiprocessing:
        from concurrent.futures import ProcessPoolExecutor
        ExecutorClass = ProcessPoolExecutor
    else:
        ExecutorClass = ThreadPoolExecutor
    
    # Execute remaining tasks
    with Progress() as progress:
        task_id = progress.add_task(
            f"[green]Processing {len(tasks_to_execute)} tasks...", 
            total=len(tasks_to_execute)
        )
        
        with ExecutorClass(max_workers=nproc) as executor:
            # Submit all tasks
            future_to_index = {}
            for i, task_input in enumerate(tasks_to_execute):
                original_index = task_indices[i]
                
                # Handle different input types
                if isinstance(task_input, dict):
                    future = executor.submit(func, **task_input)
                elif isinstance(task_input, (tuple, list)):
                    future = executor.submit(func, *task_input)
                else:
                    future = executor.submit(func, task_input)
                
                future_to_index[future] = original_index
            
            # Process completed tasks
            completed_count = 0
            for future in as_completed(future_to_index.keys()):
                original_index = future_to_index[future]
                
                try:
                    result = future.result()
                    results[original_index] = result
                    
                    # Save result if needed
                    if save is not None and keys is not None:
                        key = keys[original_index]
                        saved_results[key] = result
                        _safe_save(saved_results, save)
                    
                except Exception as e:
                    print(f"Task {original_index} failed with error: {e}")
                    results[original_index] = None
                
                completed_count += 1
                progress.update(task_id, advance=1)
    
    # Final save
    if save is not None:
        _safe_save(saved_results, save)
    
    return results

def _safe_save(data: Dict[str, Any], filepath: str) -> None:
    """Safely save data to file with file locking."""
    try:
        with open(filepath, 'w') as f:
            portalocker.lock(f, portalocker.LOCK_EX)
            dump(data, filepath)
            portalocker.unlock(f)
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
