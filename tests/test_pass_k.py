import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vlmeval.dataset.utils.pass_k import PassKMixin


class _Base:
    """Minimal stand-in for ImageBaseDataset."""

    def __init__(self, dataset='toy'):
        self.data = pd.DataFrame({'index': ['a', 'b'], 'question': ['qa', 'qb']})
        self.post_build(dataset)

    def post_build(self, dataset):
        pass

    @classmethod
    def report_primary_metric(cls, metrics):
        return {'accuracy': metrics['accuracy']}


class _Toy(PassKMixin, _Base):
    PASS_K_HIT_COL = 'hit'

    def evaluate(self, eval_file, **kwargs):
        # Single-sample scorer; sees an un-repeated dataset and one prediction per question.
        assert len(self.data) == 2, 'evaluate must see the un-repeated data'
        df = pd.read_csv(eval_file, sep='\t', dtype=str).sort_values(by='index')
        assert 'orig_index' not in df
        detail = pd.DataFrame({'hit': (df['prediction'] == df['answer']).tolist()})
        detail.to_csv(eval_file.rsplit('.', 1)[0] + '_results.csv', index=False)
        return {'accuracy': detail['hit'].mean()}


class _NoHitCol(_Toy):
    PASS_K_HIT_COL = None


class _XlsxDetail(_Toy):
    """Writes its detail in xlsx, as datasets using the default PRED_FORMAT do."""
    PASS_K_DETAIL_FORMAT = None

    def evaluate(self, eval_file, **kwargs):
        assert len(self.data) == 2
        df = pd.read_csv(eval_file, sep='\t', dtype=str).sort_values(by='index')
        detail = pd.DataFrame({'hit': (df['prediction'] == df['answer']).tolist()})
        detail.to_excel(eval_file.rsplit('.', 1)[0] + '_results.xlsx', index=False)
        return {'accuracy': detail['hit'].mean()}


class _NoDetail(_Toy):
    """Declares PASS_K_HIT_COL but never writes the detail file."""

    def evaluate(self, eval_file, **kwargs):
        return {'accuracy': 0.5}


class _ShuffledDetail(_Toy):
    """Writes detail rows in reverse order with an index column, like GDP.pdf's self.data order."""

    def evaluate(self, eval_file, **kwargs):
        df = pd.read_csv(eval_file, sep='\t', dtype=str)
        detail = pd.DataFrame({
            'index': df['index'].tolist()[::-1],
            'hit': (df['prediction'] == df['answer']).tolist()[::-1],
        })
        detail.to_csv(eval_file.rsplit('.', 1)[0] + '_results.csv', index=False)
        return {'accuracy': float(detail['hit'].astype(bool).mean())}


class _BaseWithEval(_Base):
    """Parent that implements evaluate; the mixin subclass below does not override it."""
    calls = 0

    def evaluate(self, eval_file, **kwargs):
        type(self).calls += 1
        assert len(self.data) == 2, 'evaluate must see the un-repeated data'
        df = pd.read_csv(eval_file, sep='\t', dtype=str).sort_values(by='index')
        detail = pd.DataFrame({'hit': (df['prediction'] == df['answer']).tolist()})
        detail.to_csv(eval_file.rsplit('.', 1)[0] + '_results.csv', index=False)
        return {'accuracy': detail['hit'].mean()}


class _Inherit(PassKMixin, _BaseWithEval):
    PASS_K_HIT_COL = 'hit'


def _preds(k, hits):
    rows = []
    for orig, ans in [('a', '1'), ('b', '2')]:
        for s in range(k):
            ok = hits[orig][s]
            rows.append(dict(index=len(rows), orig_index=orig, sample_id=s, answer=ans,
                             prediction=ans if ok else 'x'))
    return pd.DataFrame(rows)


class TestPassK(unittest.TestCase):

    def test_invalid_k(self):
        for bad in (0, -1, 1.5, True, '3'):
            with self.assertRaises(ValueError):
                _Toy(k=bad)

    def test_k1_is_noop(self):
        ds = _Toy(k=1)
        self.assertEqual(ds.data['index'].tolist(), ['a', 'b'])
        self.assertNotIn('orig_index', ds.data)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            pd.DataFrame({'index': ['a', 'b'], 'answer': ['1', '2'], 'prediction': ['1', 'x']}).to_csv(
                f, sep='\t', index=False)
            self.assertEqual(ds.evaluate(str(f)), {'accuracy': 0.5})

    def test_post_build_repeats(self):
        ds = _Toy(k=3)
        self.assertEqual(ds.data['index'].tolist(), list(range(6)))
        self.assertEqual(ds.data['orig_index'].tolist(), ['a'] * 3 + ['b'] * 3)
        self.assertEqual(ds.data['sample_id'].tolist(), [0, 1, 2] * 2)
        self.assertEqual(ds.data['question'].tolist(), ['qa'] * 3 + ['qb'] * 3)

    def test_metrics(self):
        ds = _Toy(k=3)
        hits = {'a': [True, False, True], 'b': [False, False, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(3, hits).to_csv(f, sep='\t', index=False)
            res = ds.evaluate(str(f))
            self.assertAlmostEqual(res['accuracy'], 2 / 6)
            self.assertAlmostEqual(res['pass@3'], 0.5)
            self.assertAlmostEqual(res['pass^3'], 0.0)
            self.assertEqual(len(ds.data), 6)  # self.data restored
            acc = pd.read_csv(Path(d) / 'p_acc.csv').iloc[0]
            self.assertAlmostEqual(acc['pass@3'], 0.5)
            detail = pd.read_csv(Path(d) / 'p_results.csv')
            self.assertEqual(len(detail), 6)
            self.assertEqual(sorted(detail['orig_index'].unique()), ['a', 'b'])

    def test_pass_all(self):
        ds = _Toy(k=2)
        hits = {'a': [True, True], 'b': [True, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(2, hits).to_csv(f, sep='\t', index=False)
            res = ds.evaluate(str(f))
            self.assertAlmostEqual(res['pass@2'], 1.0)
            self.assertAlmostEqual(res['pass^2'], 0.5)

    def test_no_hit_col_reports_average_only(self):
        ds = _NoHitCol(k=2)
        hits = {'a': [True, False], 'b': [False, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(2, hits).to_csv(f, sep='\t', index=False)
            res = ds.evaluate(str(f))
            self.assertEqual(list(res), ['accuracy'])
            self.assertAlmostEqual(res['accuracy'], 0.25)

    def test_detail_format_is_probed(self):
        # evaluate() decides the detail format itself (often PRED_FORMAT, i.e. xlsx); the mixin
        # must read whatever was written instead of assuming csv.
        ds = _XlsxDetail(k=2)
        hits = {'a': [True, False], 'b': [False, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(2, hits).to_csv(f, sep='\t', index=False)
            res = ds.evaluate(str(f))
            self.assertAlmostEqual(res['pass@2'], 0.5)
            self.assertAlmostEqual(res['pass^2'], 0.0)
            self.assertEqual(len(pd.read_csv(Path(d) / 'p_results.csv')), 4)

    def test_detail_rows_are_joined_on_index_not_position(self):
        # The aggregate pass@k is invariant under permuting same-sized groups, so a positional
        # match hides here; what breaks is the per-question attribution in the merged detail.
        ds = _ShuffledDetail(k=2)
        hits = {'a': [True, True], 'b': [False, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(2, hits).to_csv(f, sep='\t', index=False)
            ds.evaluate(str(f))
            detail = pd.read_csv(Path(d) / 'p_results.csv')
            for orig in ('a', 'b'):
                rows = detail[detail['orig_index'] == orig]
                self.assertEqual(len(rows), 2)
                self.assertEqual(set(rows['hit'].astype(bool)), set(hits[orig]),
                                 f'{orig} was matched to the wrong detail rows')

    def test_pinned_detail_format_wins(self):

        ds = _XlsxDetail(k=2)
        ds.PASS_K_DETAIL_FORMAT = 'csv'  # wrong on purpose: the dataset writes xlsx
        hits = {'a': [True, False], 'b': [False, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(2, hits).to_csv(f, sep='\t', index=False)
            with self.assertRaises(FileNotFoundError):
                ds.evaluate(str(f))

    def test_missing_detail_file_is_reported_clearly(self):
        ds = _NoDetail(k=2)
        hits = {'a': [True, False], 'b': [False, False]}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(2, hits).to_csv(f, sep='\t', index=False)
            with self.assertRaises(FileNotFoundError) as ctx:
                ds.evaluate(str(f))
            self.assertIn('PASS_K_DETAIL_FORMAT', str(ctx.exception))
            self.assertIn('PASS_K_HIT_COL', str(ctx.exception))

    def test_inherited_evaluate_is_not_shadowed(self):
        # No evaluate defined on the subclass: PassKMixin.evaluate must be used and delegate to the parent.
        self.assertNotIn('evaluate', _Inherit.__dict__)
        self.assertIs(_Inherit.evaluate, PassKMixin.evaluate)
        ds = _Inherit(k=3)
        hits = {'a': [True, False, True], 'b': [False, False, False]}
        _BaseWithEval.calls = _Inherit.calls = 0
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(3, hits).to_csv(f, sep='\t', index=False)
            res = ds.evaluate(str(f))
            self.assertEqual(_Inherit.calls, 3)  # parent scorer called once per sample_id
            self.assertAlmostEqual(res['accuracy'], 2 / 6)
            self.assertAlmostEqual(res['pass@3'], 0.5)
            self.assertAlmostEqual(res['pass^3'], 0.0)

    def test_inherited_evaluate_k1_delegates_to_parent(self):
        ds = _Inherit(k=1)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            pd.DataFrame({'index': ['a', 'b'], 'answer': ['1', '2'], 'prediction': ['1', 'x']}).to_csv(
                f, sep='\t', index=False)
            self.assertEqual(ds.evaluate(str(f)), {'accuracy': 0.5})

    def test_incomplete_predictions_rejected(self):
        ds = _Toy(k=3)
        hits = {'a': [True] * 3, 'b': [True] * 3}
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            _preds(3, hits).iloc[:-1].to_csv(f, sep='\t', index=False)
            with self.assertRaises(ValueError):
                ds.evaluate(str(f))

    def test_primary_metric_prefers_pass_at_k(self):
        self.assertEqual(_Toy.report_primary_metric({'accuracy': 0.1, 'pass@5': 0.2, 'pass^5': 0.0}),
                         {'pass@5': 0.2})
        self.assertEqual(_Toy.report_primary_metric({'accuracy': 0.1}), {'accuracy': 0.1})

    def test_base_order_is_enforced(self):
        # Listing the dataset first would let its __init__/post_build shadow the mixin.
        with self.assertRaises(TypeError) as ctx:
            type('_Swapped', (_Base, PassKMixin), {})

        self.assertIn('PassKMixin', str(ctx.exception))
        self.assertIn('_Base', str(ctx.exception))

    def test_base_order_allows_cooperative_intermediate_class(self):
        # A base listed after PassKMixin cannot shadow it, even one defining the hooks.
        class _Ok(PassKMixin, _Base):
            def post_build(self, dataset):
                super().post_build(dataset)
                self.post_built = True

        self.assertTrue(_Ok(k=1).post_built)


class TestZEROBenchPassK(unittest.TestCase):

    def test_zerobench_wiring(self):
        from vlmeval.dataset.image_vqa import ZEROBench
        self.assertTrue(issubclass(ZEROBench, PassKMixin))
        self.assertEqual(ZEROBench.PASS_K_DEFAULT, 5)
        self.assertEqual(ZEROBench.PASS_K_HIT_COL, 'Correct?')

    def test_zerobench_end_to_end_metrics(self):
        from vlmeval.dataset.image_vqa import ZEROBench
        ds = ZEROBench.__new__(ZEROBench)
        ds.k = 3
        ds.data = pd.DataFrame({'index': [10, 10, 10, 20, 20, 20, 30, 30, 30],
                                'orig_index': [10] * 3 + [20] * 3 + [30] * 3,
                                'sample_id': [0, 1, 2] * 3})
        rows = []
        for i, (o, ans, preds) in enumerate([(10, '1', ['{1}', '{2}', '{1}']),
                                             (20, '2', ['{3}', '{4}', '{5}']),
                                             (30, '6', ['{6}', '{6}', '{6}'])]):
            for s, p in enumerate(preds):
                rows.append(dict(index=len(rows), orig_index=o, sample_id=s, answers=ans, prediction=p))
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / 'p.tsv'
            pd.DataFrame(rows).to_csv(f, sep='\t', index=False)
            res = ZEROBench.evaluate(ds, str(f))
            self.assertAlmostEqual(res['accuracy'], 5 / 9)
            self.assertAlmostEqual(res['pass@3'], 2 / 3)
            self.assertAlmostEqual(res['pass^3'], 1 / 3)


if __name__ == '__main__':
    unittest.main()
