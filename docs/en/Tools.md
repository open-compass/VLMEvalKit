# CLI Tool: vlmutil (ve)

VLMEvalKit provides a CLI tool `vlmutil` (alias `ve`). The entrypoint is defined in `setup.py`, and the implementation lives under `vlmeval/tools/`. This page lists the main commands and typical use cases.

## 1. Basic usage

```bash
vlmutil -h
vlmutil <command> -h
```

## 2. Model and dataset lists

### 2.1 Model list

```bash
vlmutil mlist VLM   # understanding models
vlmutil mlist ULM   # generation models
vlmutil mlist API   # API models
```

### 2.2 Dataset list

```bash
vlmutil dlist all            # all datasets
vlmutil dlist MMBench        # a dataset group (if exists)
```

`dlist` checks `DATASET_GROUPS` first; otherwise it prints all `SUPPORTED_DATASETS`.

## 3. Minimal model usability check

```bash
vlmutil check InternVL3-8B
```

This command constructs several minimal inputs using `assets/apple.jpg`:

- single image + single question
- multi-image + multi-question
- for generation models, it also runs T2I / TI2I / TI2TI checks depending on `EXPERTISE`

It is useful for quickly verifying that a model can run.

## 4. Evaluation utilities

### 4.1 Evaluate a single prediction file

```bash
vlmutil eval /path/to/{model}_{dataset}.tsv --judge GPT4o --api-nproc 32
```

Common options:

- `--judge`: LLM used for evaluation
- `--api-nproc`: concurrency
- `--retry`: retry count on failures
- `--verbose`: more verbose logs
- `--rerun`: delete existing judge artifacts and rerun evaluation

### 4.2 Print historical results

```bash
vlmutil print_acc {model_name} {dataset_name} --root /path/to/outputs
```

This summarizes and displays scores from existing result files.

## 5. Entrypoint proxy

```bash
vlmutil run --data MMBench_DEV_EN --model InternVL3-8B --verbose
```

`vlmutil run` forwards arguments to `run.py`, making it easy to run from any directory.

## 6. Data and result helper commands

### 6.1 circular: MCQ circular augmentation

```bash
vlmutil circular /path/to/data.tsv
```

Applies a “circular” transformation to MCQ datasets and writes `*_circular.tsv`.

### 6.2 localize: localize large TSV

```bash
vlmutil localize /path/to/data.tsv
```

Creates a `_local.tsv` for very large files or path issues.

### 6.3 merge_pkl: merge distributed inference results

```bash
vlmutil merge_pkl /path/to/pkl_dir --world-size 4
```

Merges per-rank shard `.pkl` files produced during multi-GPU inference.

### 6.4 missing: check missing datasets in a group

```bash
vlmutil missing /path/to/outputs/{model_name} --group MMBench
```

Shows which tasks in a dataset group have not been completed.

### 6.5 fetch: collect all results for a dataset

```bash
vlmutil fetch MMBench --source /path/to/outputs --target /path/to/debug
```

Collects prediction and evaluation files related to a specific dataset.

### 6.6 upload / upload_data: upload to TOS

```bash
vlmutil upload /path/to/file --bucket vlmeval-data --prefix data
vlmutil upload_data /path/to/file
```

Requires configuring TOS access keys (via env vars or `.env`).

## 7. Command cheat sheet

| Command | Purpose |
| --- | --- |
| `mlist` | list models |
| `dlist` | list datasets |
| `check` | minimal model validation |
| `eval` | evaluate a prediction file |
| `print_acc` | summarize scores from existing results |
| `run` | proxy to `run.py` |
| `circular` | circular augmentation for MCQ datasets |
| `localize` | localize TSV |
| `merge_pkl` | merge distributed intermediate shards |
| `missing` | check missing tasks in a dataset group |
| `fetch` | collect results for a dataset |
| `upload` | upload a file to TOS |
| `upload_data` | upload a dataset file |
