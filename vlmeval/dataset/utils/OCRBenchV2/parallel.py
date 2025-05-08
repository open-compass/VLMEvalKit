from multiprocessing import Pool
from tqdm import tqdm

def parallel_process(items, func, use_kwargs=False, n_jobs=1, front_num=1):
    """Process items in parallel using multiprocessing Pool.
    
    Args:
        items (list): List of items to process
        func (callable): Function to apply to each item
        use_kwargs (bool): Whether to pass items as keyword arguments
        n_jobs (int): Number of parallel jobs
        front_num (int): Number of items to process in front
        
    Returns:
        list: Results of processing items
    """
    if n_jobs == 1:
        return [func(**item) if use_kwargs else func(item) for item in tqdm(items)]
    
    with Pool(n_jobs) as pool:
        if use_kwargs:
            results = list(tqdm(pool.imap(lambda x: func(**x), items), total=len(items)))
        else:
            results = list(tqdm(pool.imap(func, items), total=len(items)))
    return results 