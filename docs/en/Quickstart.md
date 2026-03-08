# Quickstart

This document is for first-time VLMEvalKit users. It covers installation, API key setup, choosing models/datasets, inference and evaluation, output directory layout, and common environment variables. For deeper structure and development details, see:

- Documentation entry: [/docs/en/README.md](/docs/en/README.md)
- Model interfaces and best practices: [/docs/en/Model.md](/docs/en/Model.md)
- Dataset formats and conventions: [/docs/en/Dataset.md](/docs/en/Dataset.md)
- Config system: [/docs/en/ConfigSystem.md](/docs/en/ConfigSystem.md)
- CLI tools: [/docs/en/Tools.md](/docs/en/Tools.md)

## 1. Installation

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

Use an isolated Python environment and make sure it matches your model dependencies (e.g., `transformers`). The package currently requires `python>=3.7` (see `setup.py`).

## 2. Set API Keys (Only Needed for API Models or LLM Judge)

API models and LLM Judge require keys. You can:

- Create a `.env` file at the repo root, or
- Set environment variables directly

Example `.env` (common items only; add as needed):

```bash
DASHSCOPE_API_KEY=
GOOGLE_API_KEY=
OPENAI_API_KEY=
OPENAI_API_BASE=
STEPAI_API_KEY=
GLMV_API_KEY=
CW_API_BASE=
CW_API_KEY=
SENSENOVA_API_KEY=
HUNYUAN_SECRET_KEY=
HUNYUAN_SECRET_ID=
EVAL_PROXY=
```

Entrypoint scripts load `.env` automatically.

## 3. List Available Models / Datasets

`vlmutil` is the CLI entry (alias: `ve`):

```bash
vlmutil mlist VLM     # available understanding models
vlmutil mlist ULM     # available generation models
vlmutil mlist API     # available API models
vlmutil dlist all     # all supported datasets
vlmutil dlist MMBench # a dataset group (if exists)
```

To quickly check whether a model can run:

```bash
vlmutil check InternVL3-8B
```

## 4. Evaluate Understanding Models (run.py)

The understanding entrypoint is `run.py`. It supports image, video, and multi-turn dialogue datasets.

### 4.1 Basic usage

```bash
python run.py --data MMBench_DEV_EN MME --model InternVL3-8B --verbose
```

Inference only (no evaluation):

```bash
python run.py --data MMBench_DEV_EN --model InternVL3-8B --mode infer
```

Evaluation only (when predictions already exist):

```bash
python run.py --data MMBench_DEV_EN --model InternVL3-8B --mode eval
```

### 4.2 Multi-GPU parallelism

`torchrun` launches multiple processes; each process instantiates one model instance. Each instance is allocated `N_GPU // N_PROC` GPUs (see `run.py` launcher logic).

```bash
torchrun --nproc-per-node=4 run.py --data MME --model InternVL3-8B
```

### 4.3 Using a config file

When the (model, dataset) combinations get complex, use `--config`:

```bash
python run.py --config config.json
```

See [/docs/en/ConfigSystem.md](/docs/en/ConfigSystem.md) for the schema.

## 5. Evaluate Generation Models (run_gen.py)

For generation datasets (e.g., T2I/TI2I/TI2TI), you must use `run_gen.py`:

```bash
python run_gen.py --data DPGBench --model Janus-Pro-7B
```

`run_gen.py` supports `--num-generations` for multiple samples per instance; see [/docs/en/Workflow.md](/docs/en/Workflow.md).

## 6. Output Directory Layout

The default output directory is `./outputs`, and can be overridden by `MMEVAL_ROOT`:

```bash
export MMEVAL_ROOT=/path/to/outputs
```

Typical output layout for understanding evaluation:

```
outputs/
  {model_name}/
    T{date}_G{git_hash}/
      {model_name}_{dataset}.tsv
      {model_name}_{dataset}_{judge}_score.csv
```

Generation evaluation is similar but may include extra sample/instance-level record files.

## 7. Common Environment Variables

| Variable | Purpose |
| --- | --- |
| `LMUData` | dataset download/cache root, default `~/LMUData` |
| `MMEVAL_ROOT` | output root for inference and evaluation |
| `PRED_FORMAT` | prediction format (default `tsv`, optional `xlsx/json`) |
| `EVAL_FORMAT` | evaluation result format (default `csv`, optional `json`) |
| `EVAL_PROXY` | API proxy for the evaluation (judge) phase |
| `DIST_TIMEOUT` | distributed timeout (seconds) |
| `SKIP_ERR=1` | skip and record inference errors |

## 8. FAQ

### 8.1 Result differences across runs

Different environments (transformers/torch/CUDA versions) can lead to differences. Keep intermediate files under `outputs` for reproducibility and debugging.

### 8.2 Model does not support interleaving

If `INTERLEAVE=False`, the framework falls back to “first image + concatenated text”. In a custom model, you can handle this via `message_to_promptimg`.
