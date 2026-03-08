# VLMEvalKit-Lite Documentation

This page is the entry point for the English translation of the documentation originally located under `/docs/zh-CN/`. The goal is to help you quickly find information along the path “run it → explain it → extend it”.

If this is your first time using VLMEvalKit, read in this order:

1. [/docs/en/Quickstart.md](/docs/en/Quickstart.md): installation, API key setup, and your first evaluation command
2. [/docs/en/Workflow.md](/docs/en/Workflow.md): end-to-end inference/evaluation workflow, artifacts, and reproducibility
3. [/docs/en/Model.md](/docs/en/Model.md): API/open-source model interfaces, message format, best practices
4. [/docs/en/Dataset.md](/docs/en/Dataset.md): dataset TSV conventions, image materialization, naming rules
5. [/docs/en/Tools.md](/docs/en/Tools.md): CLI tool `vlmutil` (`ve`) for checks/eval/helpers
6. [/docs/en/Environment.md](/docs/en/Environment.md): environment and dependency recommendations
7. [/docs/en/Troubleshooting.md](/docs/en/Troubleshooting.md): common issues and debugging

## 1. What This Is

VLMEvalKit (Python package name: `vlmeval`) is a toolchain for evaluating vision-language models (VLM) and generation-style models (ULM/UG). It mainly provides:

- Unified dataset management: auto download, caching, image materialization, prediction file conventions
- Unified model input: interleaved multimodal messages (image/text/video)
- Inference entrypoints: `run.py` (understanding tasks), `run_gen.py` (generation tasks)
- Evaluation entrypoint: dataset class implements `evaluate()`; optional LLM Judge is supported
- CLI tool: `vlmutil` (`ve`) to list/check/evaluate/organize outputs

## 2. Key Concepts (Recommended First Read)

### 2.1 Model Categories: VLM / ULM / API

At the configuration layer, VLMEvalKit groups models into three categories (see `vlmeval/config/__init__.py`):

- VLM: understanding models (open-source/local or API), implement `generate()` to produce text answers
- ULM: generation models (e.g., T2I/TI2I/TI2TI), implement `generate()` to produce images or mixed outputs
- API: a collection of API-backed models (many also appear in VLM/ULM depending on capabilities)

A quick rule of thumb:

- Evaluating “image QA / MCQ / multi-turn dialogue / video understanding” → use `run.py` (VLM flow)
- Evaluating “text-to-image / image editing / generation tasks with image output” → use `run_gen.py` (ULM flow)

### 2.2 Responsibilities of a Dataset Object

In code, each benchmark corresponds to a dataset class. Its core responsibilities are:

- `build_prompt(line)`: build a sample into the unified message input format
- `evaluate(eval_file, **judge_kwargs)`: read the prediction file, compute metrics, call LLM Judge if needed

See [/docs/en/Dataset.md](/docs/en/Dataset.md) for details.

### 2.3 Standard Message Format (Multimodal Input)

The central convention is: model inputs are represented as an interleaved list, where each element is a dict:

```python
[
  {"type": "image", "value": "/abs/path/to/img.png"},
  {"type": "text", "value": "Question ..."},
]
```

Both open-source and API models normalize various user inputs (string, path list, dict, role-based chat structure) into this format before inference. See [/docs/en/Model.md](/docs/en/Model.md).

## 3. Entrypoints and Workflow

### 3.1 Understanding Entrypoint: run.py

`run.py` runs per `(model, dataset)` combination:

1. Build dataset (supports auto download/prepare)
2. Inference: dispatch by dataset type
   - Video: `vlmeval/inference_video.py`
   - Multi-turn: `vlmeval/inference_mt.py`
   - Others (image/general understanding): `vlmeval/inference.py`
3. Evaluation: call dataset `evaluate()`, optional LLM Judge
4. Linking/archiving: create a symlink to “latest results” under `outputs/{model_name}/`

### 3.2 Generation Entrypoint: run_gen.py

`run_gen.py` runs generation and generation-evaluation for models/datasets with `SUPPORT_GEN=True`. The inference entry is `vlmeval/inference_gen.py`.

### 3.3 Output Directory and Reproducibility

The default output root is `./outputs` and can be overridden by `MMEVAL_ROOT`. Each run creates an `eval_id` (date + git hash) for reproducibility:

```
outputs/{model_name}/T{date}_G{git_hash}/...
outputs/{model_name}/...   # symlink to the latest run artifacts
```

See [/docs/en/Workflow.md](/docs/en/Workflow.md).

## 4. Navigation (By Task)

### 4.1 “I just want to run one model”

- [/docs/en/Quickstart.md](/docs/en/Quickstart.md)
- [/docs/en/Tools.md](/docs/en/Tools.md) (`vlmutil check/mlist/dlist`)
- [/docs/en/Environment.md](/docs/en/Environment.md)

### 4.2 “I want to understand artifacts and reproduce runs”

- [/docs/en/Workflow.md](/docs/en/Workflow.md)
- [/docs/en/Dataset.md](/docs/en/Dataset.md)
- [/docs/en/Troubleshooting.md](/docs/en/Troubleshooting.md)

### 4.3 “I want to integrate a new model”

- [/docs/en/Model.md](/docs/en/Model.md)
- [/docs/en/Development.md](/docs/en/Development.md)

### 4.4 “I want to integrate a new dataset”

- [/docs/en/Dataset.md](/docs/en/Dataset.md)
- [/docs/en/Development.md](/docs/en/Development.md)

### 4.5 “I want to batch-run combinations via config”

- [/docs/en/ConfigSystem.md](/docs/en/ConfigSystem.md)

## 5. Quick Conventions

| Item | Recommendation |
| --- | --- |
| Parallel inference | For open-source models, prefer `torchrun` for data parallel; for API models, use `--api-nproc` concurrency |
| Image materialization | Use dataset `dump_image` (injected into model by inference) |
| Output formats | `PRED_FORMAT` controls prediction extension; `EVAL_FORMAT` controls evaluation extension |
| Failed samples | With `SKIP_ERR=1`, failures write `FAIL_MSG` for later analysis and reruns |
