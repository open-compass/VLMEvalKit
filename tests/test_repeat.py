import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.utils.repeat import (RepeatedDataset, check_n_runs, merge_numeric_metrics,
                                          wrap_dataset_runs)


class _Toy(ImageBaseDataset):
    """Scores prediction == answer; writes the per-sample detail next to the eval file."""
    TYPE = 'VQA'
    PASS_K_HIT_COL = 'hit'
    DETAIL = 'csv'          # csv | xlsx
    REVERSE_DETAIL = False  # emit detail rows in another order (with an index column)
    WITH_INDEX = False
    seen_data_lens = []

    def evaluate(self, eval_file, **kwargs):
        type(self).seen_data_lens.append(len(self.data))
        df = pd.read_csv(eval_file, sep='\t', dtype=str)
        assert 'orig_index' not in df and 'sample_id' not in df
        hit = (df['prediction'] == df['answer']).tolist()
        detail = pd.DataFrame({'hit': hit})
        if self.WITH_INDEX:
            detail.insert(0, 'index', df['index'].tolist())
        if self.REVERSE_DETAIL:
            detail = detail.iloc[::-1].reset_index(drop=True)
        base = eval_file.rsplit('.', 1)[0] + '_results'
        if self.DETAIL == 'xlsx':
            detail.to_excel(base + '.xlsx', index=False)
        else:
            detail.to_csv(base + '.csv', index=False)
        return {'accuracy': sum(hit) / len(hit)}

    def build_prompt(self, line):
        return [dict(type='text', value=f"{line['index']}|{line['question']}|{sorted(line.index)}")]

    def dump_image(self, line):
        return [f"img_{line['index']}"]


class _NoHit(_Toy):
    PASS_K_HIT_COL = None


class _Reversed(_Toy):
    REVERSE_DETAIL = True
    WITH_INDEX = True


class _Xlsx(_Toy):
    DETAIL = 'xlsx'


class _NoDetail(_Toy):

    def evaluate(self, eval_file, **kwargs):
        return {'accuracy': 0.5}


class _Custom(_Toy):
    """Detail file with a non-standard name, located through ``get_pass_k_detail_paths``."""

    def evaluate(self, eval_file, **kwargs):
        res = super().evaluate(eval_file, **kwargs)
        Path(eval_file.rsplit('.', 1)[0] + '_results.csv').rename(eval_file.rsplit('.', 1)[0] + '_mine.csv')
        return res

    def get_pass_k_detail_paths(self, sub_file):
        return [sub_file.rsplit('.', 1)[0] + '_mine.csv']


class _Merging(_Toy):

    def merge_metrics(self, results):
        return {'accuracy': max(r['accuracy'] for r in results)}


def make(cls, n_runs=None, ids=('a', 'b')):
    ds = cls.__new__(cls)
    ds.dataset_name = 'toy'
    ds.data = pd.DataFrame({'index': list(ids), 'question': [f'q{i}' for i in ids],
                            'answer': [str(n + 1) for n, _ in enumerate(ids)]})
    return ds if n_runs is None else RepeatedDataset(ds, n_runs)


def preds(ds, hits):
    """Prediction table of a repeated dataset; hits[orig_id][sample] says if that sample is right."""
    rows = []
    for _, row in ds.data.iterrows():
        ok = hits[row['orig_index']][row['sample_id']]
        rows.append(dict(index=row['index'], orig_index=row['orig_index'], sample_id=row['sample_id'],
                         answer=row['answer'], prediction=row['answer'] if ok else 'x'))
    return pd.DataFrame(rows)


def evaluate(ds, table):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'p.tsv'
        table.to_csv(f, sep='\t', index=False)
        res = ds.evaluate(str(f))
        out = dict(res=res)
        for name in ('p_acc.csv', 'p_results.csv'):
            if (Path(d) / name).exists():
                out[name] = pd.read_csv(Path(d) / name)
        return out


class TestMergeNumericMetrics(unittest.TestCase):

    def test_numbers_are_averaged(self):
        self.assertEqual(merge_numeric_metrics([{'a': 1, 'b': 2.0}, {'a': 3, 'b': 4.0}]), {'a': 2.0, 'b': 3.0})

    def test_non_numbers_kept_only_when_runs_agree(self):
        merged = merge_numeric_metrics([{'n': 'x', 'm': 'p', 'flag': True}, {'n': 'x', 'm': 'q', 'flag': True}])
        self.assertEqual(merged, {'n': 'x', 'flag': True})

    def test_dataframes_are_averaged_by_column(self):
        a = pd.DataFrame({'split': ['s1', 's2'], 'acc': [1.0, 0.0]})
        b = pd.DataFrame({'split': ['s1', 's2'], 'acc': [0.0, 0.0]})
        merged = merge_numeric_metrics([a, b])
        self.assertEqual(merged['acc'].tolist(), [0.5, 0.0])
        self.assertEqual(merged['split'].tolist(), ['s1', 's2'])

    def test_mismatched_tables_are_rejected(self):
        a = pd.DataFrame({'split': ['s1'], 'acc': [1.0]})
        with self.assertRaises(ValueError):
            merge_numeric_metrics([a, pd.DataFrame({'split': ['s2'], 'acc': [1.0]})])
        with self.assertRaises(ValueError):
            merge_numeric_metrics([a, pd.DataFrame({'split': ['s1', 's2'], 'acc': [1.0, 2.0]})])

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            merge_numeric_metrics([])
        with self.assertRaises(TypeError):
            merge_numeric_metrics([{'a': 1}, 3])

    def test_check_n_runs(self):
        for bad in (0, -1, 1.5, True, '3'):
            with self.assertRaises(ValueError):
                check_n_runs(bad)
        self.assertEqual(check_n_runs(3), 3)


class TestWrapDatasetRuns(unittest.TestCase):

    def test_single_run_returns_the_dataset_itself(self):
        ds = make(_Toy)
        self.assertIs(wrap_dataset_runs(ds), ds)
        self.assertIs(wrap_dataset_runs(ds, 1), ds)
        self.assertIsNone(wrap_dataset_runs(None, 5))

    def test_dataset_default_and_cli_override(self):
        class Five(_Toy):
            DEFAULT_N_RUNS = 5
        ds = make(Five)
        self.assertEqual(wrap_dataset_runs(ds).n_runs, 5)
        self.assertEqual(wrap_dataset_runs(ds, 2).n_runs, 2)
        self.assertIs(wrap_dataset_runs(ds, 1), ds)

    def test_invalid_n_runs(self):
        with self.assertRaises(ValueError):
            wrap_dataset_runs(make(_Toy), 0)

    def test_multi_turn_datasets_are_rejected(self):
        multi_turn = make(_Toy)
        multi_turn.TYPE = 'MT'
        with self.assertRaises(NotImplementedError):
            wrap_dataset_runs(multi_turn, 2)
        self.assertIs(wrap_dataset_runs(multi_turn, 1), multi_turn)

    def test_pack_video_datasets_are_rejected_but_plain_video_is_allowed(self):
        video = make(_Toy)
        video.MODALITY = 'VIDEO'
        video.pack = True
        with self.assertRaises(NotImplementedError) as ctx:
            wrap_dataset_runs(video, 2)
        self.assertIn('pack', str(ctx.exception))
        video.pack = False
        self.assertEqual(wrap_dataset_runs(video, 2).n_runs, 2)
        del video.pack  # datasets without the attribute are not pack datasets
        self.assertEqual(wrap_dataset_runs(video, 2).n_runs, 2)

    def test_double_wrapping_is_rejected(self):
        with self.assertRaises(TypeError):
            RepeatedDataset(make(_Toy, 2), 2)


class TestRepeatedData(unittest.TestCase):

    def test_rows_columns_and_composite_index(self):
        ds = make(_Toy, 3)
        self.assertEqual(len(ds), 6)
        self.assertEqual(ds.data['index'].tolist(), ['a#0', 'a#1', 'a#2', 'b#0', 'b#1', 'b#2'])
        self.assertEqual(ds.data['orig_index'].tolist(), ['a'] * 3 + ['b'] * 3)
        self.assertEqual(ds.data['sample_id'].tolist(), [0, 1, 2] * 2)
        self.assertEqual(ds.data['question'].tolist(), ['qa'] * 3 + ['qb'] * 3)

    def test_index_is_stable_when_n_changes(self):
        small, large = make(_Toy, 3), make(_Toy, 5)
        self.assertTrue(set(small.data['index']) <= set(large.data['index']))

    def test_integer_ids_are_supported(self):
        ds = make(_Toy, 2, ids=(10, 20))
        self.assertEqual(ds.data['index'].tolist(), ['10#0', '10#1', '20#0', '20#1'])
        self.assertEqual(ds.data['orig_index'].tolist(), [10, 10, 20, 20])

    def test_existing_bookkeeping_columns_are_rejected(self):
        ds = make(_Toy)
        ds.data['sample_id'] = 0
        with self.assertRaises(ValueError):
            RepeatedDataset(ds, 2)

    def test_wrapped_dataset_is_untouched(self):
        inner = make(_Toy)
        RepeatedDataset(inner, 3)
        self.assertEqual(inner.data['index'].tolist(), ['a', 'b'])
        self.assertEqual(list(inner.data.columns), ['index', 'question', 'answer'])


class TestTransparency(unittest.TestCase):

    def test_class_looks_like_the_wrapped_dataset(self):
        ds = make(_Toy, 2)
        self.assertIs(ds.__class__, _Toy)
        self.assertTrue(isinstance(ds, _Toy))
        self.assertIs(type(ds), RepeatedDataset)
        self.assertEqual(ds.__class__.__name__, '_Toy')
        self.assertIs(ds.__class__.report_primary_metric.__func__, _Toy.report_primary_metric.__func__)

    def test_attributes_are_forwarded(self):
        ds = make(_Toy, 2)
        self.assertEqual(ds.MODALITY, 'IMAGE')
        self.assertEqual(ds.TYPE, 'VQA')
        self.assertEqual(ds.dataset_name, 'toy')
        self.assertEqual(ds.get_default_judge_model(), None)
        with self.assertRaises(AttributeError):
            ds.no_such_attribute

    def test_own_methods_win_over_the_wrapped_ones(self):
        ds = make(_Toy, 2)
        self.assertEqual(len(ds), 4)
        self.assertEqual(ds[0]['index'], 'a#0')
        self.assertIsNot(ds.data, ds.wrapped.data)

    def test_data_setter_replaces_the_wrapper_table_only(self):
        ds = make(_Toy, 2)
        table = ds.data.copy()
        table['prediction'] = 'p'
        ds.data = table
        self.assertIn('prediction', ds.data)
        self.assertNotIn('prediction', ds.wrapped.data)

    def test_releasing_data_releases_the_wrapped_table_too(self):
        ds = make(_Toy, 2)
        ds.data = None
        self.assertIsNone(ds.data)
        self.assertIsNone(ds.wrapped.data)

    def test_prompt_and_image_see_the_original_row(self):
        ds = make(_Toy, 2)
        by_row = ds.build_prompt(ds.data.iloc[3])
        by_position = ds.build_prompt(3)
        self.assertEqual(by_row, by_position)
        text = by_row[0]['value']
        self.assertTrue(text.startswith('b|qb|'), text)
        self.assertNotIn('orig_index', text)
        self.assertNotIn('sample_id', text)
        self.assertEqual(ds.dump_image(ds.data.iloc[3]), ['img_b'])

    def test_copy_and_pickle_keep_the_wrapper(self):
        ds = make(_Toy, 2)
        for clone in (copy.copy(ds), copy.deepcopy(ds), pickle.loads(pickle.dumps(ds))):
            self.assertIs(type(clone), RepeatedDataset)
            self.assertIs(clone.__class__, _Toy)
            self.assertEqual(clone.n_runs, 2)
            self.assertEqual(clone.data['index'].tolist(), ds.data['index'].tolist())
            self.assertEqual(clone.build_prompt(3), ds.build_prompt(3))


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        _Toy.seen_data_lens = []

    def test_metrics(self):
        ds = make(_Toy, 3)
        hits = {'a': [True, False, True], 'b': [False, False, False]}
        out = evaluate(ds, preds(ds, hits))
        self.assertAlmostEqual(out['res']['accuracy'], 2 / 6)
        self.assertAlmostEqual(out['res']['pass@3'], 0.5)
        self.assertAlmostEqual(out['res']['pass^3'], 0.0)
        self.assertEqual(len(ds.data), 6)
        self.assertAlmostEqual(out['p_acc.csv'].iloc[0]['pass@3'], 0.5)
        self.assertEqual(len(out['p_results.csv']), 6)
        # evaluate() ran once per run, on the un-repeated dataset
        self.assertEqual(_Toy.seen_data_lens, [2, 2, 2])

    def test_pass_all(self):
        ds = make(_Toy, 2)
        out = evaluate(ds, preds(ds, {'a': [True, True], 'b': [True, False]}))
        self.assertAlmostEqual(out['res']['pass@2'], 1.0)
        self.assertAlmostEqual(out['res']['pass^2'], 0.5)

    def test_without_hit_column_only_merged_metrics_are_reported(self):
        ds = make(_NoHit, 2)
        out = evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertEqual(list(out['res']), ['accuracy'])
        self.assertAlmostEqual(out['res']['accuracy'], 0.25)

    def test_detail_rows_are_joined_on_index_not_position(self):
        # The aggregates survive a permutation of equally sized groups; the per-question
        # attribution in the merged detail does not.
        ds = make(_Reversed, 2)
        hits = {'a': [True, True], 'b': [False, False]}
        out = evaluate(ds, preds(ds, hits))
        detail = out['p_results.csv']
        for orig in ('a', 'b'):
            rows = detail[detail['orig_index'] == orig]
            self.assertEqual(set(rows['hit'].astype(bool)), set(hits[orig]), orig)

    def test_detail_format_is_probed(self):
        ds = make(_Xlsx, 2)
        out = evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertAlmostEqual(out['res']['pass@2'], 0.5)
        self.assertEqual(len(out['p_results.csv']), 4)

    def test_pinned_detail_format_wins(self):
        ds = make(_Xlsx, 2)
        ds.wrapped.PASS_K_DETAIL_FORMAT = 'csv'  # wrong on purpose: the dataset writes xlsx
        with self.assertRaises(FileNotFoundError):
            evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))

    def test_missing_detail_file_is_reported_clearly(self):
        ds = make(_NoDetail, 2)
        with self.assertRaises(FileNotFoundError) as ctx:
            evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertIn('get_pass_k_detail_paths', str(ctx.exception))
        self.assertIn('PASS_K_HIT_COL', str(ctx.exception))

    def test_detail_path_hook(self):
        ds = make(_Custom, 2)
        out = evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertAlmostEqual(out['res']['pass@2'], 0.5)

    def test_nan_in_the_hit_column_counts_as_not_passed(self):
        class Nan(_Toy):
            def evaluate(self, eval_file, **kwargs):
                res = super().evaluate(eval_file, **kwargs)
                path = eval_file.rsplit('.', 1)[0] + '_results.csv'
                detail = pd.read_csv(path).astype(float)
                detail.loc[:, 'hit'] = detail['hit'].where(detail['hit'] > 0)  # False -> NaN
                detail.to_csv(path, index=False)
                return res
        ds = make(Nan, 2)
        out = evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertAlmostEqual(out['res']['pass@2'], 0.5)
        self.assertAlmostEqual(out['res']['pass^2'], 0.0)

    def test_dataset_merge_metrics_override_is_used(self):
        ds = make(_Merging, 2)
        out = evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertAlmostEqual(out['res']['accuracy'], 0.5)  # max of (0.5, 0.0), not the mean

    def test_incomplete_predictions_are_rejected(self):
        ds = make(_Toy, 3)
        table = preds(ds, {'a': [True] * 3, 'b': [True] * 3})
        with self.assertRaises(ValueError):
            evaluate(ds, table.iloc[:-1])

    def test_extra_samples_are_trimmed_to_n_runs(self):
        # Predictions written by `--n-runs 5`, evaluated by `--n-runs 3`: samples 3 and 4 are
        # dropped, so only a's first three (hit, hit, miss) and b's first three (all miss) count.
        # The rows are shuffled to prove the trim goes by sample_id, not by position.
        table = preds(make(_Toy, 5), {'a': [True, True, False, True, True], 'b': [False] * 5})
        table = table.sample(frac=1, random_state=0).reset_index(drop=True)
        out = evaluate(make(_Toy, 3), table)
        self.assertAlmostEqual(out['res']['accuracy'], 2 / 6)
        self.assertAlmostEqual(out['res']['pass@3'], 0.5)
        self.assertAlmostEqual(out['res']['pass^3'], 0.0)
        self.assertEqual(len(out['p_results.csv']), 6)
        self.assertEqual(_Toy.seen_data_lens, [2, 2, 2])

    def test_too_few_samples_per_question_are_rejected(self):
        table = preds(make(_Toy, 2), {'a': [True, False], 'b': [True, False]})
        with self.assertRaises(ValueError):
            evaluate(make(_Toy, 5), table)

    def test_prediction_file_from_a_plain_run_is_rejected(self):
        ds = make(_Toy, 2)
        with self.assertRaises(ValueError):
            evaluate(ds, pd.DataFrame({'index': ['a', 'b'], 'prediction': ['1', '2']}))

    def test_table_results_have_no_pass_k_added(self):
        class Table(_Toy):
            def evaluate(self, eval_file, **kwargs):
                super().evaluate(eval_file, **kwargs)
                return pd.DataFrame({'split': ['all'], 'acc': [0.5]})
        ds = make(Table, 2)
        out = evaluate(ds, preds(ds, {'a': [True, False], 'b': [False, False]}))
        self.assertEqual(list(out['res'].columns), ['split', 'acc'])


class TestRealDatasets(unittest.TestCase):

    def test_zerobench_pass_k(self):
        from vlmeval.dataset.image_vqa import ZEROBench
        self.assertEqual(ZEROBench.DEFAULT_N_RUNS, 5)
        self.assertEqual(ZEROBench.PASS_K_HIT_COL, 'Correct?')
        base = ZEROBench.__new__(ZEROBench)
        base.dataset_name = 'ZEROBench'
        base.data = pd.DataFrame({'index': [10, 20, 30], 'question': ['x'] * 3, 'answers': ['1', '2', '6']})
        ds = RepeatedDataset(base, 3)
        answers = {10: ['{1}', '{2}', '{1}'], 20: ['{3}', '{4}', '{5}'], 30: ['{6}', '{6}', '{6}']}
        table = ds.data.copy()
        table['prediction'] = [answers[o][s] for o, s in zip(table['orig_index'], table['sample_id'])]
        out = evaluate(ds, table)
        self.assertAlmostEqual(out['res']['accuracy'], 5 / 9)
        self.assertAlmostEqual(out['res']['pass@3'], 2 / 3)
        self.assertAlmostEqual(out['res']['pass^3'], 1 / 3)
        # ZEROBench keeps pass@k as its primary metric, and the wrapper resolves to that class.
        self.assertEqual(ds.__class__.report_primary_metric({'accuracy': 0.1, 'pass@5': 0.2, 'pass^5': 0.0}),
                         {'pass@5': 0.2})
        self.assertEqual(ds.__class__.report_primary_metric({'accuracy': 0.1}), {'accuracy': 0.1})

    def test_gdp_pdf_keeps_all_pass_as_primary_metric(self):
        from vlmeval.dataset.gdp_pdf import GDPPDF
        self.assertEqual(GDPPDF.DEFAULT_N_RUNS, 5)
        self.assertEqual(GDPPDF.PASS_K_HIT_COL, 'all_pass')
        self.assertEqual(GDPPDF.report_primary_metric({'all_pass': 0.4, 'pass@5': 0.9}), {'all_pass': 0.4})
        gdp = GDPPDF.__new__(GDPPDF)
        gdp._judge_name = 'judge-x'
        self.assertEqual(gdp.get_pass_k_detail_paths('/tmp/a_run0.tsv'),
                         ['/tmp/a_run0_judge_rubric_v1_judge-x_detail.tsv'])


if __name__ == '__main__':
    unittest.main()
