import os

import torch

try:
    from ....utils.mp_util import cpu_count as quota_cpu_count
except ImportError:
    from os import cpu_count as os_cpu_count

    def quota_cpu_count():
        return os_cpu_count()


def get_available_cpu_count() -> int:
    count = quota_cpu_count()
    try:
        return max(int(count), 1)
    except (TypeError, ValueError):
        return 1


def get_env_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return max(int(raw_value), 1)
    except ValueError:
        return default


def get_metric_batch_size(
    env_name: str,
    cpu_default: int,
    cuda_default: int,
    cpu_cap: int | None = None,
) -> int:
    default = cuda_default if torch.cuda.is_available() else cpu_default
    if not torch.cuda.is_available() and cpu_cap is not None:
        default = min(default, cpu_cap)
    return get_env_positive_int(env_name, default)


def get_in_memory_dataloader_workers(
    env_name: str = "VLMEVAL_SARENA_NUM_WORKERS",
) -> int:
    default = 0 if not torch.cuda.is_available() else min(4, get_available_cpu_count())
    return get_env_positive_int(env_name, default)


def maybe_configure_torch_cpu_threads() -> dict[str, int] | None:
    if torch.cuda.is_available():
        return None

    available_cpus = get_available_cpu_count()
    target_threads = get_env_positive_int(
        "VLMEVAL_SARENA_TORCH_THREADS", available_cpus
    )
    target_interop = get_env_positive_int(
        "VLMEVAL_SARENA_TORCH_INTEROP_THREADS",
        min(max(1, target_threads // 2), 8),
    )

    try:
        torch.set_num_threads(target_threads)
    except RuntimeError:
        pass

    thread_config = {"threads": torch.get_num_threads()}

    if hasattr(torch, "set_num_interop_threads") and hasattr(
        torch, "get_num_interop_threads"
    ):
        try:
            torch.set_num_interop_threads(target_interop)
        except RuntimeError:
            pass
        thread_config["interop_threads"] = torch.get_num_interop_threads()

    return thread_config
