import datetime
import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from filelock import SoftFileLock

from .file import RUN_STATUS_NAME, NumpyEncoder, find_prediction_files, load
from .log import get_logger

SUMMARY_STATUS_ORDER = {
    'pending': 0,
    'infer': 1,
    'eval': 2,
    'done': 3,
}

_UNSET = object()
logger = get_logger(__name__)


def _get_run_status_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / RUN_STATUS_NAME


def _get_run_status_lock_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / f'{RUN_STATUS_NAME}.lock'


def _load_run_status_unlocked(status_file: str | Path) -> dict[str, Any]:
    status_file = Path(status_file)
    if not status_file.exists():
        return {}
    try:
        data = load(str(status_file))
    except Exception as err:
        logger.warning(f'Failed to load run status from {status_file}: {err}')
        return {}
    return data if isinstance(data, dict) else {}


def _dump_run_status_unlocked(status_file: str | Path, status: dict[str, Any]) -> None:
    status_file = Path(status_file)
    tmp_file = status_file.with_suffix('.tmp')
    with open(tmp_file, 'w', encoding='utf-8') as fout:
        json.dump(status, fout, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    tmp_file.replace(status_file)


def _init_run_status(run_status: dict[str, Any],
                     run_dir: str | Path,
                     model_name: str | None = None) -> dict[str, Any]:
    run_status.setdefault('schema_version', '1.0')
    run_status.setdefault('eval_id', Path(run_dir).name)
    run_status.setdefault('created_at', _iso_now())
    run_status.setdefault('datasets', {})
    if model_name is not None:
        run_status['model_name'] = model_name
    return run_status


def _set_or_pop(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is None or (isinstance(value, dict) and not value):
        mapping.pop(key, None)
    else:
        mapping[key] = value


def _merge_optional(existing: dict[str, Any], key: str, value: Any) -> Any:
    if value is _UNSET:
        return existing.get(key)
    return value


def load_run_status(run_dir: str | Path) -> dict[str, Any]:
    return _load_run_status_unlocked(_get_run_status_path(run_dir))


def upsert_run_status(run_dir: str | Path, **fields: Any) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = _get_run_status_path(run_dir)
    lock_file = _get_run_status_lock_path(run_dir)

    with SoftFileLock(str(lock_file)):
        run_status = _load_run_status_unlocked(status_file)
        run_status = _init_run_status(run_status, run_dir, model_name=fields.get('model_name'))
        for key, value in fields.items():
            if key == 'datasets' or value is _UNSET:
                continue
            run_status[key] = value
        run_status['updated_at'] = _iso_now()
        _dump_run_status_unlocked(status_file, run_status)
        return run_status


def _iso_now():
    return datetime.datetime.now().astimezone().isoformat()


def is_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return not pd.isna(value)
        except Exception:
            return True
    if isinstance(value, str):
        try:
            return not pd.isna(float(value))
        except Exception:
            return False
    return False


def to_number(value: Any) -> Any:
    if isinstance(value, (np.integer, int)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, str):
        if value.isdigit():
            return int(value)
        else:
            return float(value)
    return value


def _safe_key(key: Any) -> str:
    key = str(key).strip()
    return key if key else 'value'


def _flatten_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    metrics = {}
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return metrics

    numeric_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce')
        if series.notna().any():
            numeric_cols.append(col)
    if not numeric_cols:
        return metrics

    dim_cols = [c for c in df.columns if c not in numeric_cols]
    for row_idx, row in df.iterrows():
        dims = []
        for col in dim_cols:
            val = row[col]
            if pd.isna(val):
                continue
            sval = str(val).strip()
            if sval:
                dims.append(f'{_safe_key(col)}={sval}')
        dim_prefix = '|'.join(dims)

        for col in numeric_cols:
            val = row[col]
            if not is_number(val):
                continue
            key = _safe_key(col) if not dim_prefix else f'{dim_prefix}|{_safe_key(col)}'
            if key in metrics:
                key = f'{key}#{row_idx}'
            metrics[key] = to_number(val)
    return metrics


def _flatten_any(obj: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(obj, pd.DataFrame):
        for key, value in _flatten_dataframe(obj).items():
            merged_key = key if not prefix else f'{prefix}|{key}'
            if merged_key in out:
                merged_key = f'{merged_key}#dup'
            out[merged_key] = value
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            key = _safe_key(key)
            next_prefix = key if not prefix else f'{prefix}|{key}'
            _flatten_any(value, next_prefix, out)
        return

    if isinstance(obj, (list, tuple)):
        if not obj:
            return
        if all(is_number(x) for x in obj):
            if len(obj) == 1:
                key = prefix or 'value'
                if key in out:
                    key = f'{key}#0'
                out[key] = to_number(obj[0])
            else:
                for idx, val in enumerate(obj):
                    key = f'{prefix}|{idx}' if prefix else str(idx)
                    if key in out:
                        key = f'{key}#dup'
                    out[key] = to_number(val)
        else:
            for idx, val in enumerate(obj):
                next_prefix = f'{prefix}|{idx}' if prefix else str(idx)
                _flatten_any(val, next_prefix, out)
        return

    if is_number(obj):
        key = prefix or 'value'
        if key in out:
            key = f'{key}#dup'
        out[key] = to_number(obj)


def flatten_summary_metrics(metrics_source: Any) -> dict[str, Any]:
    if metrics_source is None:
        return {}
    if isinstance(metrics_source, pd.DataFrame):
        return _flatten_dataframe(metrics_source)
    out = {}
    _flatten_any(metrics_source, '', out)
    return out


def _resolve_dataset_reporter(dataset_name: str, dataset_obj=None):
    reporter = dataset_obj.__class__ if dataset_obj is not None else None
    if reporter is None:
        try:
            import vlmeval.dataset as dataset_module
            from vlmeval.dataset.video_dataset_config import supported_video_datasets

            if dataset_name in supported_video_datasets:
                video_ds = supported_video_datasets[dataset_name]
                if isinstance(video_ds, partial):
                    reporter = video_ds.func
            else:
                for dataset_cls in dataset_module.DATASET_CLASSES:
                    if dataset_name in dataset_cls.supported_datasets():
                        reporter = dataset_cls
                        break
        except Exception:
            reporter = None

    try:
        from vlmeval.dataset.image_base import ImageBaseDataset
    except Exception:
        return reporter

    if reporter is None:
        return ImageBaseDataset

    required_methods = ('report_infer_err', 'report_judge_err', 'report_primary_metric')
    if any(not hasattr(reporter, method_name) for method_name in required_methods):
        return ImageBaseDataset
    return reporter


def _serialize_primary_metric(primary_metrics: dict[str, int | float]) -> str | list[str] | None:
    keys = list(primary_metrics)
    if not keys:
        return None
    return keys[0] if len(keys) == 1 else keys


def _serialize_primary_metric_value(primary_metrics: dict[str, int | float]):
    if not primary_metrics:
        return None
    if len(primary_metrics) == 1:
        return next(iter(primary_metrics.values()))
    return primary_metrics


def _status_rank(status: str | None) -> int:
    return SUMMARY_STATUS_ORDER.get(status or '', -1)


def _rel_path(path: str | Path | None, run_dir: str | Path) -> str | None:
    if not path:
        return None
    try:
        path_obj = Path(path)
        run_dir_obj = Path(run_dir)
        if path_obj.is_absolute():
            try:
                return str(path_obj.relative_to(run_dir_obj))
            except Exception:
                return str(path_obj)
        return str(path_obj)
    except Exception:
        return str(path)


def upsert_dataset_status(
    run_dir: str | Path,
    model_name: str,
    dataset_name: str,
    status: str | None = None,
    prediction_file: str | Path | None = _UNSET,
    judge_model: str | None = _UNSET,
    source_run: str | None = _UNSET,
    reuse_aux: str | None = _UNSET,
    metrics_source: Any = _UNSET,
    skip_reason: str | None = _UNSET,
    error_message: str | None = _UNSET,
    dataset_obj=None,
):
    if status is not None and status not in SUMMARY_STATUS_ORDER:
        raise ValueError(f'Invalid summary status: {status}. '
                         f'Expected one of {list(SUMMARY_STATUS_ORDER.keys())}')

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_file = _get_run_status_path(run_dir)
    lock_file = _get_run_status_lock_path(run_dir)

    with SoftFileLock(str(lock_file)):
        run_status = _load_run_status_unlocked(status_file)
        run_status = _init_run_status(run_status, run_dir, model_name=model_name)
        run_status['updated_at'] = _iso_now()

        datasets = run_status.setdefault('datasets', {})
        dataset_status = datasets.setdefault(dataset_name, {})
        reporter = _resolve_dataset_reporter(dataset_name, dataset_obj=dataset_obj)

        if status is not None and _status_rank(status) < _status_rank(
                dataset_status.get('status')):
            return run_status

        if metrics_source is _UNSET:
            metrics = dataset_status.get('metrics', {})
            primary_metric = dataset_status.get('primary_metric')
        elif metrics_source is None:
            metrics = {}
            primary_metric = None
        else:
            metrics = flatten_summary_metrics(metrics_source)
            primary_metric = _serialize_primary_metric(reporter.report_primary_metric(metrics))

        next_status = dataset_status.get('status') if status is None else status
        next_prediction_file = _merge_optional(
            dataset_status,
            'prediction_file',
            _UNSET if prediction_file is _UNSET else _rel_path(prediction_file, run_dir),
        )
        next_judge_model = _merge_optional(dataset_status, 'judge_model', judge_model)
        next_source_run = _merge_optional(dataset_status, 'source_run', source_run)
        next_reuse_aux = _merge_optional(dataset_status, 'reuse_aux', reuse_aux)
        has_metrics_update = metrics_source is not _UNSET and metrics_source is not None
        has_skip_reason_update = skip_reason is not _UNSET and skip_reason is not None
        has_error_update = error_message is not _UNSET and error_message is not None

        if skip_reason is _UNSET:
            next_skip_reason = dataset_status.get('skip_reason')
            if has_metrics_update or has_error_update:
                next_skip_reason = None
        else:
            next_skip_reason = skip_reason

        if error_message is _UNSET:
            next_error_message = dataset_status.get('error_message')
            if has_metrics_update or has_skip_reason_update:
                next_error_message = None
        else:
            next_error_message = error_message

        dataset_status.pop('mode', None)
        dataset_status.pop('primary_metric_value', None)
        _set_or_pop(dataset_status, 'status', next_status)
        _set_or_pop(dataset_status, 'prediction_file', next_prediction_file)
        _set_or_pop(dataset_status, 'judge_model', next_judge_model)
        _set_or_pop(dataset_status, 'source_run', next_source_run)
        _set_or_pop(dataset_status, 'reuse_aux', next_reuse_aux)
        _set_or_pop(dataset_status, 'skip_reason', next_skip_reason)
        _set_or_pop(dataset_status, 'error_message', next_error_message)
        _set_or_pop(dataset_status, 'primary_metric', primary_metric)
        _set_or_pop(dataset_status, 'metrics', metrics)
        dataset_status['updated_at'] = _iso_now()

        _dump_run_status_unlocked(status_file, run_status)
        return run_status


def _resolve_prediction_file(run_dir, model_name, dataset_name, prediction_file):
    if prediction_file:
        pred_path = Path(prediction_file)
        if not pred_path.is_absolute():
            pred_path = Path(run_dir) / pred_path
        if pred_path.exists():
            return pred_path

    candidates = find_prediction_files(str(run_dir), model_name, dataset_name)
    return Path(candidates[0]) if candidates else None


def collect_run_benchmark_report(run_dir):
    run_dir = Path(run_dir)
    run_status = load_run_status(str(run_dir))
    datasets = run_status.get('datasets', {})
    if not isinstance(datasets, dict):
        return []

    model_name = run_status.get('model_name') or run_dir.parent.name
    rows = []
    for dataset_name in sorted(datasets.keys()):
        dataset_status = datasets.get(dataset_name, {})
        if not isinstance(dataset_status, dict):
            continue

        pred_path = _resolve_prediction_file(
            run_dir=run_dir,
            model_name=model_name,
            dataset_name=dataset_name,
            prediction_file=dataset_status.get('prediction_file'),
        )
        reporter = _resolve_dataset_reporter(dataset_name)
        infer_report = reporter.report_infer_err(pred_path)
        infer_failed = infer_report.get('failed', 0)
        infer_total = infer_report.get('total', 0)

        skip_reason = dataset_status.get('skip_reason')
        error_message = dataset_status.get('error_message')
        judge_failed = None
        judge_total = None

        # Eval is considered attempted when there is no skip_reason.
        if not skip_reason:
            judge_report = reporter.report_judge_err(
                pred_path,
                total_samples=infer_total,
                judge_model=dataset_status.get('judge_model'),
                error_message=error_message,
            )
            judge_failed = judge_report.get('failed')
            judge_total = judge_report.get('total')

        primary_metrics = reporter.report_primary_metric(dataset_status.get('metrics'))
        primary_metric = _serialize_primary_metric(primary_metrics)

        rows.append(
            dict(
                benchmark=dataset_name,
                infer_failed=infer_failed,
                infer_total=infer_total,
                judge_failed=judge_failed,
                judge_total=judge_total,
                primary_metric=primary_metric,
                primary_metric_value=_serialize_primary_metric_value(primary_metrics),
                skip_reason=skip_reason,
                eval_error=error_message,
            ))

    return rows
