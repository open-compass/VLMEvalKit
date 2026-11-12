"""Opt-in pass@k support for datasets whose ``evaluate`` scores one prediction per question.

Usage::

    class MyBench(PassKMixin, ImageVQADataset):
        PASS_K_HIT_COL = 'Correct?'   # column in the per-sample detail file marking a correct answer

Behaviour:

* ``k == 1`` (default): the mixin is a no-op.
* ``k > 1``: every question is repeated ``k`` times at ``post_build`` (``orig_index`` keeps the
  original id, ``sample_id`` is the repeat number, ``index`` stays unique). At evaluation time the
  prediction file is split by ``sample_id`` into ``k`` ordinary single-sample files, the dataset's own
  ``evaluate`` is run unchanged on each, numeric metrics are averaged over the ``k`` runs, and
  ``pass@k`` / ``pass^k`` are derived from the per-sample detail files.
"""
import os
import os.path as osp
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype

from vlmeval.smp import d2df, dump, get_intermediate_file_path, load


class PassKMixin:
    # Name of the 0/1 (bool) column in the per-sample detail file written by ``evaluate``.
    # ``None`` -> only the averaged metrics are reported, pass@k / pass^k are skipped.
    PASS_K_HIT_COL = None
    # Suffix of that detail file, as passed to ``get_intermediate_file_path``.
    PASS_K_DETAIL_SUFFIX = '_results'
    # Format of that detail file. The dataset's ``evaluate`` writes it and decides that format
    # itself -- most call ``get_intermediate_file_path(eval_file, '_results')`` without a format,
    # which resolves through ``PRED_FORMAT`` (default xlsx), not ``EVAL_FORMAT``. So leave this
    # ``None`` to read whichever format actually exists; set it only when the dataset hardcodes a
    # format that the probe could get wrong, and keep it in sync with what ``evaluate`` writes.
    PASS_K_DETAIL_FORMAT = None
    # Formats tried in order when ``PASS_K_DETAIL_FORMAT`` is ``None``.
    PASS_K_DETAIL_PROBE = ('csv', 'json', 'xlsx')
    # k used when the caller does not pass one.
    PASS_K_DEFAULT = 1

    def __init__(self, *args, k=None, **kwargs):
        k = self.PASS_K_DEFAULT if k is None else k
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError(f'k must be a positive integer, got {k!r}')
        self.k = k
        super().__init__(*args, **kwargs)

    def post_build(self, dataset):
        super().post_build(dataset)
        if self.k == 1:
            return
        data = self.data.copy()
        data['orig_index'] = data['index']
        data['sample_id'] = 0
        data = data.loc[data.index.repeat(self.k)].reset_index(drop=True)
        data['sample_id'] = [i % self.k for i in range(len(data))]
        data['index'] = range(len(data))
        self.data = data

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # ``PassKMixin`` has to come *first* in the bases: ``class X(PassKMixin, Dataset)``.
        #
        # This hook runs whenever ``PassKMixin`` appears anywhere in the bases, so the wrapping
        # below happens even with the order flipped. But everything the wrapped ``evaluate`` relies
        # on is resolved through the MRO: ``self.k`` comes from ``PassKMixin.__init__`` and the
        # repeated rows from ``PassKMixin.post_build``. Under ``class X(Dataset, PassKMixin)`` the
        # dataset's own ``__init__``/``post_build`` win, the mixin's never run, and ``k`` is left
        # unset while ``evaluate`` still expects it -- the caller then sees the unrelated-looking
        # ``TypeError: Dataset.__init__() got an unexpected keyword argument 'k'`` (or, if ``k`` was
        # bypassed, ``AttributeError: 'X' object has no attribute 'k'``). Report the real cause here
        # instead, before any of that can happen.
        #
        # The test is "no class between ``cls`` and ``PassKMixin`` defines these hooks", because
        # such a class shadows the mixin unless it happens to chain to ``super()``; requiring the
        # documented base order is clearer than trying to detect a cooperative ``super()`` call.
        mro = cls.__mro__
        if PassKMixin not in mro:
            raise TypeError(f'{cls.__name__} must inherit from PassKMixin')
        shadowing = [
            base.__name__ for base in mro[1:mro.index(PassKMixin)]
            if '__init__' in base.__dict__ or 'post_build' in base.__dict__
        ]
        if shadowing:
            raise TypeError(
                f'{cls.__name__} must list PassKMixin before {", ".join(shadowing)}: their '
                '`__init__`/`post_build` would shadow the mixin, leaving `k` unset and the data '
                f'unrepeated. Write `class {cls.__name__}(PassKMixin, ...)` instead.')

        # A dataset's own ``evaluate`` would shadow the mixin one; wrap it so it becomes the
        # single-sample scorer that the pass@k logic calls once per sample_id.
        own = cls.__dict__.get('evaluate')
        if own is not None and not getattr(own, '_pass_k_wrapped', False):
            def evaluate(self, eval_file, **judge_kwargs):
                return self._pass_k_evaluate(eval_file, own, **judge_kwargs)
            evaluate._pass_k_wrapped = True
            cls.evaluate = evaluate

    def evaluate(self, eval_file, **judge_kwargs):
        # Reached only when the dataset class defines no ``evaluate`` of its own.
        return self._pass_k_evaluate(eval_file, lambda ds, f, **kw: super(PassKMixin, ds).evaluate(f, **kw),
                                     **judge_kwargs)

    def _pass_k_evaluate(self, eval_file, single_eval, **judge_kwargs):
        """``single_eval(self, file, **kw)`` scores a file holding one prediction per question."""
        if self.k == 1:
            return single_eval(self, eval_file, **judge_kwargs)

        preds = load(eval_file)
        if 'sample_id' not in preds or 'orig_index' not in preds:
            raise ValueError(f'{eval_file} lacks sample_id/orig_index; it was not produced with k={self.k}')
        n_orig = preds['orig_index'].nunique()
        counts = preds.groupby('orig_index')['sample_id'].nunique()
        if len(preds) != n_orig * self.k or (counts != self.k).any():
            raise ValueError(f'{eval_file} does not contain exactly k={self.k} samples per question')

        ext = eval_file.split('.')[-1]
        stem = osp.basename(eval_file)[:-(len(ext) + 1)]
        sub_dir = osp.join(osp.dirname(eval_file), f'{stem}_pass_k')
        os.makedirs(sub_dir, exist_ok=True)

        results, details = [], []
        full_data = self.data
        # Evaluators may look at self.data; give them the un-repeated dataset.
        base = full_data[full_data['sample_id'] == 0].copy()
        base['index'] = base['orig_index']
        self.data = base.drop(columns=['orig_index', 'sample_id']).reset_index(drop=True)
        try:
            for s in range(self.k):
                sub = preds[preds['sample_id'] == s].copy()
                sub['index'] = sub['orig_index']
                sub = sub.drop(columns=['orig_index', 'sample_id']).sort_values(by='index')
                sub_file = osp.join(sub_dir, f'{stem}_s{s}.{ext}')
                dump(sub, sub_file)
                res = single_eval(self, sub_file, **judge_kwargs)
                if not isinstance(res, dict):
                    raise TypeError(f'PassKMixin requires evaluate() to return a dict, got {type(res).__name__}')
                results.append(res)
                if self.PASS_K_HIT_COL:
                    detail_file, detail = self._load_detail(sub_file, len(sub))
                    details.append(self._align_detail(detail, sub, s))
        finally:
            self.data = full_data

        final = {}
        for key in results[0]:
            vals = [r.get(key) for r in results]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                final[key] = sum(vals) / len(vals)
        if details:
            all_detail = pd.concat(details, ignore_index=True)
            hit = self._hit_mask(all_detail[self.PASS_K_HIT_COL])
            grouped = hit.groupby(all_detail['orig_index'])
            final[f'pass@{self.k}'] = grouped.any().mean()
            final[f'pass^{self.k}'] = grouped.all().mean()
            # The merged detail is a report artifact: pin its format so it does not move with
            # PRED_FORMAT/EVAL_FORMAT, matching the `_acc` csv written below.
            dump(all_detail, get_intermediate_file_path(
                eval_file, self.PASS_K_DETAIL_SUFFIX, 'csv'))
        dump(d2df(final), get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return final

    @staticmethod
    def _align_detail(detail, sub, sample_id):
        """Attach ``orig_index``/``sample_id`` to the rows ``evaluate`` wrote.

        A detail file that carries an ``index`` column is matched on it, because the writer is free
        to emit rows in its own order -- GDP.pdf writes them in ``self.data`` order, which need not
        match the sorted prediction file, and positional matching there attributed every pass/fail
        to the wrong question and silently corrupted pass@k.

        Detail files without an ``index`` column (ZEROBench writes Question_ID/Correct? only) fall
        back to positional alignment, which is what those evaluators already guarantee: they iterate
        the eval file in its own row order. Give such an evaluator an ``index`` column to opt into
        the safer join.
        """
        sub_keys = pd.Series(sub['index'].tolist()).astype(str)
        detail = detail.copy()
        if 'index' not in detail:
            if len(detail) != len(sub_keys):
                raise ValueError(f'The detail file has {len(detail)} rows, expected {len(sub_keys)}; '
                                 'without an `index` column it is aligned positionally')
            detail['orig_index'] = sub_keys.tolist()
            detail['sample_id'] = sample_id
            return detail

        detail['index'] = detail['index'].astype(str)
        if detail['index'].duplicated().any() or len(detail) != sub_keys.nunique():
            raise ValueError('The detail file must contain each question exactly once')
        missing = set(sub_keys) - set(detail['index'])
        if missing:
            raise ValueError(f'The detail file is missing {len(missing)} question(s), e.g. '
                             f'{sorted(missing)[:3]}')
        aligned = detail.set_index('index').loc[sub_keys.tolist()].reset_index(drop=True)
        aligned['orig_index'] = sub_keys.tolist()
        aligned['sample_id'] = sample_id
        return aligned

    def _pass_k_detail_paths(self, sub_file):
        """Candidate paths of the per-sample detail file ``evaluate`` wrote for one split.

        Override this when the dataset names its detail file differently (a judge-named artifact,
        for example) instead of writing ``get_intermediate_file_path(eval_file, '_results')``.
        """
        formats = ([self.PASS_K_DETAIL_FORMAT] if self.PASS_K_DETAIL_FORMAT is not None
                   else list(self.PASS_K_DETAIL_PROBE))
        return [get_intermediate_file_path(sub_file, self.PASS_K_DETAIL_SUFFIX, fmt) for fmt in formats]

    def _load_detail(self, sub_file, expected_rows):
        """Read the per-sample detail file ``evaluate`` wrote for one split.

        The dataset owns the write side and picks its own format (often ``PRED_FORMAT``), so when
        ``PASS_K_DETAIL_FORMAT`` is not pinned, probe the formats it could plausibly have used
        instead of assuming one.
        """
        tried = []
        for detail_file in self._pass_k_detail_paths(sub_file):
            if not osp.exists(detail_file):
                tried.append(detail_file)
                continue
            detail = load(detail_file)
            if len(detail) != expected_rows:
                raise ValueError(f'{detail_file} has {len(detail)} rows, expected {expected_rows}')
            return detail_file, detail
        raise FileNotFoundError(
            'PassKMixin could not find the per-sample detail file that '
            f'`{type(self).__name__}.evaluate` should have written for {sub_file}. Tried: '
            f'{", ".join(tried)}. Either write it with `get_intermediate_file_path(eval_file, '
            f"'{self.PASS_K_DETAIL_SUFFIX}')`, override `_pass_k_detail_paths`, or set "
            'PASS_K_DETAIL_FORMAT to the format used. Set PASS_K_HIT_COL = None if this dataset '
            'cannot expose a per-sample correctness column; pass@k/pass^k are then skipped and '
            'only the averaged metrics are reported.')

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

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict):
            for key in metrics:
                if re.fullmatch(r'pass@\d+', str(key)):
                    return {str(key): metrics[key]}
        return super().report_primary_metric(metrics)
