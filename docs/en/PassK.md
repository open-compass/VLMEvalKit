# Support pass@k in a benchmark

> 📐 How to make a dataset report `pass@k` / `pass^k` / `avg@n`

Some benchmarks are scored over *several attempts at the same question*: a model is asked the same
prompt `N` times, and the score is `pass@N` (at least one attempt correct), `pass^N` (every attempt
correct) or the average over attempts. VLMEvalKit runs that through `--n-runs N`, which wraps the
dataset in a `RepeatedDataset` (`vlmeval/dataset/utils/repeat.py`).

**The dataset does not implement repetition.** The wrapper repeats every question `N` times, calls
the dataset's own `evaluate` once per run, merges the metrics, and derives `pass@N` / `pass^N` from
the per-sample detail files those runs produced. Supporting `pass@k` is therefore a matter of
telling the wrapper two things: how many runs you want by default, and where it can read a
per-sample correctness column.

## Option 1: run a dataset N times from the command line

No code change is needed for the repetition itself. `--n-runs N` works on any dataset that has a
single-run `evaluate` returning a `dict` or a `pd.DataFrame`:

```bash
python run.py --data YourBench --model YourModel --n-runs 5 --api-nproc 64
```

Every metric `evaluate` reports as a number is averaged over the runs, and reported under its own
key -- so `accuracy` becomes the mean accuracy of the 5 attempts, i.e. an `avg@5`. `pass@5` /
`pass^5` are added **only if** the dataset exposes a per-sample correctness column -- see below.

With `N = 1` (the default) the dataset is not wrapped at all, so nothing about a normal run changes.

`--n-runs` requires a sampling model. With greedy decoding all `N` runs are identical and `pass@N`
degenerates into the single-run score; the wrapper logs a warning for this reason.

## Option 2: declare it in the dataset

Two class attributes on the dataset are enough for the common case:

```python
class YourBench(ImageVQADataset):
    DEFAULT_N_RUNS = 5          # what `--n-runs` uses when it is not given on the command line
    PASS_K_HIT_COL = 'correct'  # the 0/1 column of the per-sample detail file `evaluate` writes
```

`DEFAULT_N_RUNS` only sets the default; `--n-runs` on the command line always wins. `PASS_K_HIT_COL`
is what makes `pass@N` and `pass^N` possible at all: without it the wrapper only merges the metrics.

### What `evaluate` has to write

The wrapper reads the per-sample detail file that your `evaluate` already writes for a single run
and takes the column named by `PASS_K_HIT_COL`. By default that file is looked for at the path
produced by

```python
get_intermediate_file_path(eval_file, '_results')
```

in any of the formats `csv`, `json`, `xlsx` (probed in that order), so a dataset that already writes
a per-sample result table usually needs no change at all. Two things are worth doing:

- **Include an `index` column.** The wrapper joins detail rows to questions on `index` when the
  column exists. Without it, rows are aligned by position, which only works if your evaluator emits
  them in the order of the eval file. An evaluator that reorders rows (for example by iterating
  `self.data` rather than the predictions) will silently attribute every pass/fail to the wrong
  question, and the aggregated score will still look plausible — this has happened.
- **Make the hit column unambiguous.** It is interpreted as a boolean: `True`/`1`, numeric `1`/`0`,
  or a text column whose values are `1`, `1.0`, `true`, `yes`. A NaN (an attempt that was never
  scored) counts as *not passed*, which is the safe direction for `pass@N`.

A `pass@N` from ZEROBench, which needs nothing beyond the two attributes:

```python
class ZEROBench(ImageVQADataset):
    DEFAULT_N_RUNS = 5
    PASS_K_HIT_COL = 'Correct?'

    def evaluate(self, eval_file, **judge_kwargs):
        ...
        dump(output_df, get_intermediate_file_path(eval_file, '_results', 'csv'))
        dump(d2df(result), get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return result
```

## The full set of knobs

All three base classes (`ImageBaseDataset`, `TextBaseDataset`, `VideoBaseDataset`) declare the same
attributes and hooks.

| Attribute | Default | Meaning |
|---|---|---|
| `DEFAULT_N_RUNS` | `1` | Runs per question when `--n-runs` is not given |
| `PASS_K_HIT_COL` | `None` | The 0/1 column of the detail file; `None` disables `pass@N` / `pass^N` |
| `PASS_K_DETAIL_SUFFIX` | `'_results'` | Suffix of that detail file, as passed to `get_intermediate_file_path` |
| `PASS_K_DETAIL_FORMAT` | `None` | Pin the detail file's format; `None` means probe |
| `PASS_K_DETAIL_PROBE` | `('csv', 'json', 'xlsx')` | Formats tried, in order, when the format is not pinned |

Overridable methods, for datasets that do not fit the defaults:

| Method | Override when |
|---|---|
| `get_default_n_runs()` | The default depends on the dataset variant rather than being constant |
| `get_pass_k_detail_paths(sub_file)` | The detail file is not named after a fixed suffix |
| `merge_metrics(results)` | The metrics cannot be averaged |

### Where the detail file is: `get_pass_k_detail_paths`

The default is `<eval_file stem>_results.<format>`. Override it when the file is named after
something the base class cannot know — GDP.pdf names it after the judge that produced it:

```python
def get_pass_k_detail_paths(self, sub_file):
    return [get_judge_detail_file(sub_file, 'rubric_v1', self._judge_name, 'tsv')]
```

It returns a list of candidates, tried in order, so a dataset can offer a fallback.

### How the runs are merged: `merge_metrics`

The default (`merge_numeric_metrics`) averages every metric that is numeric in all runs and keeps a
non-numeric metric only when all runs agree on it. Override it when averaging is wrong — for a
maximum, a sum, or a metric that has to be recomputed from the runs' outputs:

```python
def merge_metrics(self, results):
    return {'accuracy': max(r['accuracy'] for r in results)}
```

### Which metric is reported as primary

`report_primary_metric` picks the key the summary table shows. After a repeated run the metrics
contain `accuracy`, `pass@N` and `pass^N`; if your benchmark's headline number is `pass@N`, say so,
otherwise the generic key chooser will report `accuracy`:

```python
@classmethod
def report_primary_metric(cls, metrics):
    for key in metrics:
        if re.fullmatch(r'pass@\d+', str(key)):
            return {str(key): metrics[key]}
    return super().report_primary_metric(metrics)
```

Note that this hook must also keep working for a **single** run, where no `pass@N` key exists: fall
back to the parent implementation rather than assuming the key is there.

## Running evaluation separately, and reusing it

Inference and evaluation can be split, and one inference of `N` runs can be scored at any smaller
`k`:

```bash
# 5 samples per question, judged once
python run.py --data YourBench --model YourModel --n-runs 5 --mode all
# re-score the same predictions as k = 3, without inferring or judging again
python run.py --data YourBench --model YourModel --n-runs 3 --mode eval --reuse
```

A prediction file holding *more* samples per question than `n_runs` is trimmed to its first
`n_runs`, since samples are independent and the first `k` of them are a valid `k`-sample run. Fewer
samples than requested is an error. `--mode eval` on a file written by a smaller `--n-runs` is
skipped as an incomplete inference result.

Each run's files go into a `<stem>_runs/` directory beside the prediction file, and `--reuse`
copies that directory along with the other evaluation auxiliaries — so a judge cache written per run
is not paid for twice. Keep anything a run writes derived from the `eval_file` you are handed, and it
will be reused; write it somewhere else and it will not be.

## Limitations

- Multi-turn datasets (`TYPE = 'MT'`) and pack-mode video reject `N > 1`: both infer outside the
  per-row path the wrapper relies on (pack mode issues one request per video, so repeated rows would
  be ignored). Non-pack video is supported.
- The `N` runs of a question do not share a judge cache: each run has its own, keyed by content. With
  sampling they mostly differ anyway; with greedy decoding identical answers are judged `N` times.
- `--n-runs 1` never wraps the dataset, so an eval run with `--n-runs 1` over a repeated prediction
  file is skipped as incomplete rather than scored.

## Verifying an integration

Run the dataset with `--n-runs 2` on a small subset and check three files in the run directory:

- `<model>_<alias>.<ext>` — the predictions, one row per (question, sample), with the `index` column
  in the form `<original id>#<sample>`. The extension follows `PRED_FORMAT` (`tsv`, `xlsx` — the
  default — or `json`); set `PRED_FORMAT=tsv` if you want to inspect it as text
- `<model>_<alias>_acc.csv` — the merged metrics; `pass@2` and `pass^2` must be there
- `<model>_<alias>_results.csv` — the merged per-sample detail, one row per sample

If `pass@2` is missing, either `PASS_K_HIT_COL` is `None` -- in which case the wrapper silently
reports only the merged metrics, so set it -- or the detail file was not found, in which case
`evaluate` raises a `FileNotFoundError` listing every path it tried and how to name the file
instead.
