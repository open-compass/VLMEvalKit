from __future__ import annotations

import os
from typing import Any

# Keep the default single-task concurrency within a conservative 40-thread budget.
# For page matching, each worker may temporarily fan out into:
#   main thread + N page workers + N func_timeout workers + N timeout join threads
# so N=13 keeps the worst-case at 40 threads.
DEFAULT_MATCH_WORKERS = 13
DEFAULT_TEDS_WORKERS = 13
DEFAULT_CDM_WORKERS = 13


def _as_positive_int(value: Any, default: int) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(1, resolved)


def _env_or_default(env_name: str, default: int) -> int:
    return _as_positive_int(os.getenv(env_name, default), default)


def resolve_match_workers(dataset_cfg: dict[str, Any] | None) -> int:
    dataset_cfg = dataset_cfg or {}
    default = _env_or_default('OMNIDOCBENCH_MATCH_WORKERS', DEFAULT_MATCH_WORKERS)
    configured = dataset_cfg.get('match_workers', default)
    return _as_positive_int(configured, default)


def resolve_teds_workers(metric_cfg: dict[str, Any] | None) -> int:
    metric_cfg = metric_cfg or {}
    default = _env_or_default('OMNIDOCBENCH_TEDS_WORKERS', DEFAULT_TEDS_WORKERS)
    return _as_positive_int(metric_cfg.get('teds_workers', metric_cfg.get('workers', default)), default)


def resolve_cdm_workers(metric_cfg: dict[str, Any] | None) -> int:
    metric_cfg = metric_cfg or {}
    default = _env_or_default('OMNIDOCBENCH_CDM_WORKERS', DEFAULT_CDM_WORKERS)
    return _as_positive_int(metric_cfg.get('cdm_workers', metric_cfg.get('workers', default)), default)
