"""Run a dataset several times and merge the runs (avg@n, pass@n, pass^n).

``--n-runs N`` (or a dataset's ``DEFAULT_N_RUNS``) makes ``run.py`` wrap the dataset in a
:class:`RepeatedDataset`. The wrapper is a composition, not a subclass or a mixin, so the dataset
it wraps needs no knowledge of repetition:

* every question appears ``N`` times in ``data``; ``index`` becomes ``"<orig>#<sample>"``, which is
  stable when ``N`` changes (so ``--reuse`` keeps matching old predictions to the right row), while
  ``orig_index`` and ``sample_id`` keep the original id and the repeat number;
* ``build_prompt``/``dump_image`` hand the wrapped dataset its own, un-repeated row back;
* ``evaluate`` splits the prediction file into ``N`` ordinary single-run files, runs the wrapped
  dataset's own ``evaluate`` on each, merges the results with the dataset's ``merge_metrics``
  (default: :func:`merge_numeric_metrics`) and, if the dataset declares ``PASS_K_HIT_COL``, adds
  ``pass@N`` / ``pass^N`` computed from the per-sample detail files. A prediction file holding
  *more* than ``N`` samples per question (one written by a larger ``--n-runs``) is trimmed to its
  first ``N``, so ``--mode eval --n-runs k`` reports any ``k`` up to the number of samples already
  generated; too few samples is an error. A run's files go in ``<stem>_runs/`` beside the
  prediction file; ``--reuse`` copies that directory along with the other evaluation auxiliaries,
  so the judge caches the wrapped dataset wrote are not paid for again.

The wrapper reports the wrapped dataset's class as ``__class__`` so class-based lookups (status
reports, ``report_primary_metric``, class names) resolve to the real dataset.
"""
import numbers
import os
import os.path as osp

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from vlmeval.smp import d2df, dump, get_intermediate_file_path, get_logger, load

logger = get_logger(__name__)

ORIG_INDEX = 'orig_index'
SAMPLE_ID = 'sample_id'


def check_n_runs(n_runs):
    if isinstance(n_runs, bool) or not isinstance(n_runs, int) or n_runs < 1:
        raise ValueError(f'n_runs must be a positive integer, got {n_runs!r}')
    return n_runs


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _merge_frames(results):
    base = results[0]
    for other in results[1:]:
        if other.shape != base.shape or list(other.columns) != list(base.columns):
            raise ValueError('Cannot merge result tables with different shapes or columns')
    numeric = [c for c in base.columns
               if all(is_numeric_dtype(r[c]) and not is_bool_dtype(r[c]) for r in results)]
    labels = [c for c in base.columns if c not in numeric]
    for other in results[1:]:
        if labels and not other[labels].reset_index(drop=True).equals(base[labels].reset_index(drop=True)):
            raise ValueError('Cannot merge result tables whose non-numeric columns differ between runs')
    merged = base.copy()
    for column in numeric:
        merged[column] = np.mean([r[column].to_numpy(dtype=float) for r in results], axis=0)
    return merged


def merge_numeric_metrics(results):
    """Default ``merge_metrics``: average the metrics of several runs.

    ``results`` are what the dataset's ``evaluate`` returned for each run. For dicts, every metric
    that is a number in all runs is averaged; any other metric is kept only if all runs agree.
    Result tables (DataFrames) are averaged column-wise when they have the same shape and their
    non-numeric columns (row labels) are identical.
    """
    if not results:
        raise ValueError('No results to merge')
    if all(isinstance(r, pd.DataFrame) for r in results):
        return _merge_frames(results)
    if not all(isinstance(r, dict) for r in results):
        raise TypeError('Cannot merge runs: evaluate() must return a dict or a DataFrame, got '
                        f'{sorted({type(r).__name__ for r in results})}')
    merged = {}
    for key in results[0]:
        values = [r.get(key) for r in results]
        if all(_is_number(v) for v in values):
            merged[key] = sum(values) / len(values)
        elif values.count(values[0]) == len(values):
            merged[key] = values[0]
    return merged


def default_pass_k_detail_paths(dataset, sub_file):
    """Candidate paths of the per-sample detail file ``dataset.evaluate`` wrote for one run.

    The default of ``get_pass_k_detail_paths``: ``PASS_K_DETAIL_SUFFIX`` in ``PASS_K_DETAIL_FORMAT``,
    or, when the format is not pinned, in every format of ``PASS_K_DETAIL_PROBE``.
    """
    fmt = dataset.PASS_K_DETAIL_FORMAT
    formats = [fmt] if fmt is not None else list(dataset.PASS_K_DETAIL_PROBE)
    return [get_intermediate_file_path(sub_file, dataset.PASS_K_DETAIL_SUFFIX, f) for f in formats]


def _repeat_rows(data, n_runs):
    clash = [c for c in (ORIG_INDEX, SAMPLE_ID) if c in data]
    if clash:
        raise ValueError(f'The dataset already has column(s) {clash}, which the repeated data needs')
    repeated = data.iloc[np.repeat(np.arange(len(data)), n_runs)].reset_index(drop=True)
    repeated[ORIG_INDEX] = repeated['index']
    repeated[SAMPLE_ID] = [i % n_runs for i in range(len(repeated))]
    repeated['index'] = [f'{o}#{s}' for o, s in zip(repeated[ORIG_INDEX], repeated[SAMPLE_ID])]
    return repeated


def _restore(inner, n_runs, data):
    obj = RepeatedDataset.__new__(RepeatedDataset)
    obj._inner = inner
    obj.n_runs = n_runs
    obj._data = data
    return obj


class RepeatedDataset:
    """``dataset`` presented as ``n_runs`` independent samples of every question."""

    def __init__(self, dataset, n_runs):
        check_n_runs(n_runs)
        if isinstance(dataset, RepeatedDataset):
            raise TypeError('The dataset is already repeated')
        self._inner = dataset
        self.n_runs = n_runs
        self._data = _repeat_rows(dataset.data, n_runs)

    # ------------------------------------------------------------------ transparency
    @property
    def __class__(self):
        inner = self.__dict__.get('_inner')
        return type(inner) if inner is not None else RepeatedDataset

    def __getattr__(self, name):
        # Only reached when normal lookup fails: everything else belongs to the wrapped dataset.
        if name.startswith('__') or name in ('_inner', '_data'):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __reduce__(self):
        # The default protocol would rebuild the object from ``__class__`` (the wrapped class).
        return (_restore, (self._inner, self.n_runs, self._data))

    def __repr__(self):
        return f'RepeatedDataset({self._inner!r}, n_runs={self.n_runs})'

    @property
    def wrapped(self):
        return self._inner

    # ------------------------------------------------------------------ data
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value
        if value is None:
            # Dropping the table is how callers release memory; release the wrapped one as well.
            self._inner.data = None

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return dict(self._data.iloc[idx])

    # ------------------------------------------------------------------ prompts and images
    def _inner_line(self, line):
        """The row as the wrapped dataset knows it: original index, no repeat bookkeeping."""
        if isinstance(line, (int, np.integer)):
            line = self._data.iloc[line]
        elif not isinstance(line, pd.Series):
            line = pd.Series(line)
        line = line.copy()
        line['index'] = line[ORIG_INDEX]
        return line.drop(labels=[ORIG_INDEX, SAMPLE_ID])

    def build_prompt(self, line, *args, **kwargs):
        return self._inner.build_prompt(self._inner_line(line), *args, **kwargs)

    def dump_image(self, line, *args, **kwargs):
        return self._inner.dump_image(self._inner_line(line), *args, **kwargs)

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, eval_file, **judge_kwargs):
        n_runs, inner = self.n_runs, self._inner
        preds = load(eval_file)
        absent = [c for c in (ORIG_INDEX, SAMPLE_ID) if c not in preds]
        if absent:
            raise ValueError(f'{eval_file} lacks {absent}; it was not produced by a {n_runs}-run dataset')
        counts = preds.groupby(ORIG_INDEX)[SAMPLE_ID].nunique()
        if (counts > n_runs).any():
            # The file was written by a larger ``--n-runs``. The samples are independent, so its
            # first ``n_runs`` samples of every question are themselves a valid ``n_runs``-sample
            # run: evaluate those rather than refuse. That is what lets `--mode eval --n-runs k`
            # report any ``k`` up to the number of samples already generated, without inferring
            # again. Too few samples is the opposite case: it cannot be salvaged and still raises.
            logger.warning(
                f'{eval_file} holds up to {int(counts.max())} samples per question but '
                f'n_runs={n_runs}; evaluating samples 0..{n_runs - 1} of every question '
                'and ignoring the rest.')
            preds = preds[pd.to_numeric(preds[SAMPLE_ID], errors='coerce') < n_runs].copy()
            counts = preds.groupby(ORIG_INDEX)[SAMPLE_ID].nunique()
        if len(preds) != len(counts) * n_runs or (counts != n_runs).any():
            raise ValueError(f'{eval_file} does not contain exactly n_runs={n_runs} samples per question')

        ext = eval_file.split('.')[-1]
        stem = osp.basename(eval_file)[:-(len(ext) + 1)]
        run_dir = osp.join(osp.dirname(eval_file), f'{stem}_runs')
        os.makedirs(run_dir, exist_ok=True)

        hit_col = getattr(inner, 'PASS_K_HIT_COL', None)
        results, details = [], []
        for s in range(n_runs):
            sub = preds[preds[SAMPLE_ID] == s].copy()
            if len(sub) != len(counts):
                raise ValueError(f'{eval_file}: sample {s} has {len(sub)} rows, expected {len(counts)}')
            sub['index'] = sub[ORIG_INDEX]
            # Sorted like the datasets' own evaluators sort, so a detail file without an `index`
            # column lines up with it positionally.
            sub = sub.drop(columns=[ORIG_INDEX, SAMPLE_ID]).sort_values(by='index')
            sub_file = osp.join(run_dir, f'{stem}_run{s}.{ext}')
            dump(sub, sub_file)
            results.append(inner.evaluate(sub_file, **judge_kwargs))
            if hit_col:
                details.append(self._align_detail(self._load_detail(sub_file, len(sub)), sub, s))

        merge = getattr(inner, 'merge_metrics', merge_numeric_metrics)
        final = merge(results)

        if details:
            all_detail = pd.concat(details, ignore_index=True)
            hit = self._hit_mask(all_detail[hit_col])
            grouped = hit.groupby(all_detail[ORIG_INDEX])
            if isinstance(final, dict):
                final[f'pass@{n_runs}'] = float(grouped.any().mean())
                final[f'pass^{n_runs}'] = float(grouped.all().mean())
            else:
                logger.warning(f'{type(inner).__name__}.evaluate returned a table, so pass@{n_runs} and '
                               f'pass^{n_runs} are not added to it')
            # A report artifact: pinned to csv so it does not move with PRED_FORMAT/EVAL_FORMAT.
            dump(all_detail, get_intermediate_file_path(
                eval_file, getattr(inner, 'PASS_K_DETAIL_SUFFIX', '_results'), 'csv'))
        dump(d2df(final) if isinstance(final, dict) else final, get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return final

    def _load_detail(self, sub_file, expected_rows):
        """Read the per-sample detail file the wrapped ``evaluate`` wrote for one run.

        The dataset owns the write side and picks its own format (often ``PRED_FORMAT``), so when
        ``PASS_K_DETAIL_FORMAT`` is not pinned, probe the formats it could plausibly have used.
        """
        tried = []
        for detail_file in self._inner.get_pass_k_detail_paths(sub_file):
            if not osp.exists(detail_file):
                tried.append(detail_file)
                continue
            detail = load(detail_file)
            if len(detail) != expected_rows:
                raise ValueError(f'{detail_file} has {len(detail)} rows, expected {expected_rows}')
            return detail
        name = type(self._inner).__name__
        raise FileNotFoundError(
            f'Could not find the per-sample detail file that `{name}.evaluate` should have written for '
            f'{sub_file}. Tried: {", ".join(tried)}. Either write it with `get_intermediate_file_path('
            f"eval_file, '{getattr(self._inner, 'PASS_K_DETAIL_SUFFIX', '_results')}')`, define "
            f'`get_pass_k_detail_paths` on {name}, or set PASS_K_DETAIL_FORMAT to the format used. Set '
            'PASS_K_HIT_COL = None if this dataset cannot expose a per-sample correctness column; '
            'pass@n/pass^n are then skipped and only the merged metrics are reported.')

    @staticmethod
    def _align_detail(detail, sub, sample_id):
        """Attach ``orig_index``/``sample_id`` to the rows ``evaluate`` wrote.

        A detail file that carries an ``index`` column is matched on it, because the writer is free
        to emit rows in its own order -- GDP.pdf writes them in ``self.data`` order, and matching
        by position there attributed every pass/fail to the wrong question and silently corrupted
        pass@k. Detail files without an ``index`` column (ZEROBench writes Question_ID/Correct?
        only) fall back to positional alignment, which is what those evaluators already guarantee:
        they iterate the eval file in its own row order. Give such an evaluator an ``index`` column
        to opt into the safer join.
        """
        sub_keys = pd.Series(sub['index'].tolist()).astype(str)
        detail = detail.copy()
        if 'index' not in detail:
            if len(detail) != len(sub_keys):
                raise ValueError(f'The detail file has {len(detail)} rows, expected {len(sub_keys)}; '
                                 'without an `index` column it is aligned positionally')
            detail[ORIG_INDEX] = sub_keys.tolist()
            detail[SAMPLE_ID] = sample_id
            return detail

        detail['index'] = detail['index'].astype(str)
        if detail['index'].duplicated().any() or len(detail) != sub_keys.nunique():
            raise ValueError('The detail file must contain each question exactly once')
        missing = set(sub_keys) - set(detail['index'])
        if missing:
            raise ValueError(f'The detail file is missing {len(missing)} question(s), e.g. '
                             f'{sorted(missing)[:3]}')
        aligned = detail.set_index('index').loc[sub_keys.tolist()].reset_index(drop=True)
        aligned[ORIG_INDEX] = sub_keys.tolist()
        aligned[SAMPLE_ID] = sample_id
        return aligned

    @staticmethod
    def _hit_mask(series):
        """Turn a correctness column into booleans; unresolved entries (NaN) count as not passed."""
        if series.dtype == bool:
            return series
        if is_numeric_dtype(series):
            # A NaN here means the sample was never fully scored, so it did not pass.
            return series.fillna(0).astype(bool)
        normalized = series.astype(str).str.strip().str.lower()
        return normalized.isin({'1', '1.0', 'true', 'true.', 'yes'})


def resolve_n_runs(dataset, n_runs=None):
    """``--n-runs`` if given, else the dataset's own default (1 when it has none)."""
    if n_runs is None:
        getter = getattr(dataset, 'get_default_n_runs', None)
        n_runs = getter() if callable(getter) else 1
    return check_n_runs(n_runs)


def wrap_dataset_runs(dataset, n_runs=None):
    """Return ``dataset`` itself for a single run, otherwise a :class:`RepeatedDataset`."""
    if dataset is None:
        return None
    n_runs = resolve_n_runs(dataset, n_runs)
    if n_runs == 1:
        return dataset
    name = type(dataset).__name__
    if getattr(dataset, 'TYPE', None) == 'MT':
        raise NotImplementedError(
            f'Running {name} {n_runs} times is not supported: multi-turn datasets use their own '
            'inference pipeline (pass --n-runs 1).')
    if getattr(dataset, 'MODALITY', None) == 'VIDEO' and getattr(dataset, 'pack', False):
        raise NotImplementedError(
            f'Running {name} {n_runs} times is not supported in pack mode: inference there is one request '
            'per video (`dataset.videos`), not per row, so the repeated rows would be ignored '
            '(pass --n-runs 1 or use a non-pack variant).')
    logger.warning(
        f'{type(dataset).__name__}: running every question {n_runs} times. Make sure the model samples '
        '(temperature > 0); with greedy decoding all runs are identical and pass@n equals the single-run score.')
    return RepeatedDataset(dataset, n_runs)
