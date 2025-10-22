import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import psutil
from collections import defaultdict
from reward_score import compute_score


async def single_compute_score(pred, gold, problem, executor, timeout):
    loop = asyncio.get_running_loop()
    try:
        future = loop.run_in_executor(
            executor,
            partial(compute_score, pred, gold, problem)
        )
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        print(f"[Timeout] Task timeout after {timeout}s: {pred[:80]}")
        return None
    except Exception as e:
        print(f"[Error] Task failed: {e}, pred: {pred[:80]}")
        return None

async def parallel_compute_score_async(preds, golds, problems, num_processes, timeout):
    results = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        try:
            tasks_async = [
                single_compute_score(pred, gold, problem, executor, timeout)
                for pred, gold, problem in zip(preds, golds, problems)
            ]
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
            print("[Success] All tasks gathered.")
        except Exception as e:
            print(f"[Exception] async gather failed: {e}")
            raise
        finally:
            print("[Shutdown] Cleaning up remaining subprocesses...")
            terminated_count = 0
            for pid, proc in executor._processes.items():
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    try:
                        p.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    terminated_count += 1
                except Exception:
                    pass
            print(f"[Shutdown] {terminated_count} subprocess(es) terminated.")

    # Format results
    formatted = []
    for result in results:
        if isinstance(result, Exception) or result is None:
            formatted.append({
                "score": 0.,
                "acc": False,
                "extracted_gt": None,
                "extracted_pred": None,
            })
        elif isinstance(result, dict):
            formatted.append(result)
        else:
            formatted.append(result[0])
    return formatted

def run_reward_scoring(preds, golds, problems, num_processes=64, timeout=300.):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(parallel_compute_score_async(
            preds, golds, problems, num_processes, timeout
        ))
    finally:
        loop.close()



def verifier_manager(data, return_dict: bool = True):
    assert isinstance(data, list), f"data should be a list, but got {type(data)}"

    problems = [item['problem'] for item in data]
    preds = [item['response'] for item in data]
    golds = [item['answer'] for item in data]
    

    try:
        results = run_reward_scoring(
            preds=preds,
            golds=golds,
            problems=problems,
            num_processes=64,
            timeout=300.,
        )
    except asyncio.TimeoutError as e:
        print('Global timeout in reward computing! Setting all as 0.')
        results = [{
            "score": 0.,
            "acc": False,
            "extracted_gt": None,
            "extracted_pred": None,
        } for _ in range(len(data))]
    except Exception as e:
        print(f"Unexpected error in batched reward computing. Setting all as 0.: {e}")
        results = [{
            "score": 0.,
            "acc": False,
            "extracted_gt": None,
            "extracted_pred": None,
        } for _ in range(len(data))]
    
    if return_dict:
        return results
    else:
        return [result['score'] for result in results]
