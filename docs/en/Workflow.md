# Workflow and Artifacts (Understanding Eval / Generation Eval)

This document explains VLMEvalKit’s end-to-end flow: building datasets/models, running inference, running evaluation, organizing output files, and making runs reproducible. It targets users who have already “made it run” and now want to explain or automate the process.

## 1. Two main flows: run.py and run_gen.py

VLMEvalKit separates evaluation into two entrypoints:

- Understanding evaluation: `run.py`
  - For: MCQ, VQA, OCR, video understanding, multi-turn dialogue, tool-use tasks, etc. (outputs are typically text)
- Generation evaluation: `run_gen.py`
  - For: text-to-image (T2I), image editing (TI2I), edit-then-text (TI2TI), etc. (outputs can be images or mixed)

If you try to run a generation dataset with `run.py`, it will raise an explicit error telling you to use `run_gen.py`.

## 2. run.py (Understanding) execution order

For each `model_name` and each `dataset_name`, `run.py` performs:

1. Generate a run id `eval_id = T{date}_G{git_hash}` (for reproducibility and isolation)
2. Create run directories:
   - `pred_root = {work_dir}/{model_name}/{eval_id}`
   - `pred_root_meta = {work_dir}/{model_name}` (for “latest results” symlink)
3. Build the dataset object: `dataset = build_dataset(dataset_name, **kwargs)`
4. Run inference:
   - Video datasets: `vlmeval/inference_video.py`
   - Multi-turn datasets: `vlmeval/inference_mt.py`
   - Others: `vlmeval/inference.py`
5. Run evaluation:
   - Only on `RANK==0`
   - Compute metrics via `dataset.evaluate(result_file, **judge_kwargs)`
6. Link and archive: link artifacts into `pred_root_meta` for quick access

## 3. run_gen.py (Generation) execution order

The generation flow is similar, but inference is implemented in `vlmeval/inference_gen.py`. In addition:

- Both the model and dataset must satisfy `SUPPORT_GEN=True`
- It commonly produces sample/instance-level intermediate records (for multi-sampling and aggregation)

## 4. Key artifacts from inference

### 4.1 Prediction file

The prediction file is the most important artifact. Typical naming:

```
{model_name}_{dataset_name}.{PRED_FORMAT}
```

Default `PRED_FORMAT=tsv`; you can switch to `xlsx/json` via environment variables. A prediction file usually includes:

- Original sample fields (index/question/answer/options/metadata, etc.)
- `prediction`: model outputs (string or serializable structure)
- For some datasets: helper fields such as `stats`, `raw_prediction`, `thinking`, etc.

Notes on why `tsv` is the default:

- Tabular format is readable and keeps consistent columns across samples, easy to load/write via pandas
- `xlsx` is avoided because Excel cells have length limits; long model outputs can exceed the limit and break the pipeline

### 4.2 Distributed intermediate files (pickle shards)

With multi-process inference via `torchrun`, the inference stage often creates per-rank `.pkl` shards for merging. These are usually cleaned up after merging.

If an open-source model run is interrupted and then restarted with a different `WORLD_SIZE`, you may need to manually merge shards:

```bash
vlmutil merge_pkl /path/to/pred_root --world-size {WORLD_SIZE}
```

### 4.3 Resume / continuation (PREV)

Inference commonly uses `*_PREV.pkl` to cache completed and valid predictions for resuming:

- If an existing `result_file` is found, valid predictions are extracted and written into PREV
- Subsequent inference only generates missing or invalid samples

## 5. Key artifacts from evaluation

Evaluation is handled by dataset `evaluate()`. For MCQ-style datasets, common artifacts include:

- `*_score.csv`: aggregated metrics (overall + group breakdowns)
- `*_result.tsv` / `*_openai_result.tsv`: per-sample judge outputs/logs (only when LLM Judge is used)

Different datasets vary significantly; artifact formats are defined by dataset implementations.

## 6. Recommended reproducibility checklist

To make results reproducible, record:

- Code version: git commit hash (included in `eval_id`)
- Runtime environment: python/torch/cuda/transformers versions
- Run parameters: the `run.py`/`run_gen.py` command lines
- Key env vars: `LMUData`, `MMEVAL_ROOT`, `PRED_FORMAT`, `EVAL_PROXY`, etc.

In practice, containers or pinned conda envs are the most reliable, and teams often maintain a “model family → environment” mapping.

## 7. Common debugging approaches

- `vlmutil check {MODEL}`: verify minimal model usability first
- `--mode infer`: run inference only; inspect prediction file before evaluation
- `vlmutil eval <pred_file>`: evaluate a single prediction file
- `SKIP_ERR=1`: skip failures without aborting the whole run (open-source models)
- `--api-nproc`: reduce concurrency to debug API rate limits and timeouts
