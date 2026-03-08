# Dataset Format and Dataset Classes (VLMEvalKit)

This document introduces how “datasets/benchmarks” are organized in VLMEvalKit, the recommended TSV packaging convention, rules for handling image fields, naming conventions for evaluation artifacts, and two concrete examples (MathVista and ERIQ).

## Overview: what is a dataset?

In VLMEvalKit, a benchmark is represented as a dataset class. The core responsibilities of a dataset class are:

- Read and normalize data (usually a TSV mapped to a `pandas.DataFrame`)
- Build model inputs for each sample (`build_prompt`)
- Evaluate model prediction files and output metrics (`evaluate`)
- (Recommended) Follow unified naming conventions for prediction/eval intermediate files, so the framework can auto-summarize and `report`

Most image datasets inherit from [ImageBaseDataset](/vlmeval/dataset/image_base.py) or its subclasses (e.g., [ImageMCQDataset](/vlmeval/dataset/image_mcq.py)).

## Required interfaces: build_prompt / evaluate / supported_datasets

### 1) build_prompt(self, line)

The default implementation in the base class is [ImageBaseDataset.build_prompt](/vlmeval/dataset/image_base.py#L273-L292). You can reuse it directly or override it for more complex interleaving/templates.

- Input `line`:
  - `int`: row index in `self.data`, will be converted into `pd.Series`
  - `pd.Series` or `dict`: one sample record
- Output: VLMEvalKit’s unified “multimodal message format”: an interleavable list of dicts, each item is `dict(type=..., value=...)`:

```python
[
    dict(type='text', value='...'),
    dict(type='image', value='/abs/path/to/img.png'),
    dict(type='video', value='/abs/path/to/video.mp4'),
]
```

Notes:

- Common `type` values are `image` and `text`
- `video` is mainly used by video models (usually not in image base datasets)
- For `image`, `value` is usually a local path (it may also be a URL depending on the model implementation)

The base strategy is: put images first (one or multiple), then append the `question` text (see [ImageBaseDataset.build_prompt](/vlmeval/dataset/image_base.py#L273-L292)).

### 2) evaluate(self, eval_file, **judge_kwargs)

Each dataset class must implement the evaluation function (abstract method: [ImageBaseDataset.evaluate](/vlmeval/dataset/image_base.py#L293-L296)).

- Input `eval_file`: path to the model inference output file (usually `{model_name}_{dataset_name}.tsv`; the extension depends on env vars)
- Input `judge_kwargs`: when evaluation needs an external LLM/Judge, this passes judge parameters (e.g., judge model name `model`, concurrency `nproc`, timeout `timeout`, etc.)
- Return: evaluation metrics, typically a `dict` or `pd.DataFrame`

In practice, `eval_file` is often a copy of the original TSV with an added `prediction` column. The `image` column is often removed to reduce file size.

### 3) supported_datasets(cls)

Declares the list of dataset names supported by this class. The framework uses it for:

- Aggregating `SUPPORTED_DATASETS` and parsing file names ([dataset/__init__.py](/vlmeval/dataset/__init__.py#L355-L363), and [extract_model_dataset](/vlmeval/dataset/image_base.py#L110-L118))
- Constructing datasets via `build_dataset(dataset_name, **kwargs)` ([build_dataset](/vlmeval/dataset/__init__.py#L406-L433))

The base implementation returns keys of `DATASET_URL` ([ImageBaseDataset.supported_datasets](/vlmeval/dataset/image_base.py#L256-L260)). A recommended pattern is to maintain `DATASET_URL` / `DATASET_MD5`, so you can reuse base `supported_datasets()` and `__init__()`.

## Recommended TSV packaging convention (best practice)

VLMEvalKit recommends packaging each benchmark as one TSV (a `pandas.DataFrame`), and letting the dataset class handle download/loading.

### 1) Common fields (recommended for all image datasets)

- `index`: unique identifier per sample. Base class casts it to `str` ([ImageBaseDataset.__init__](/vlmeval/dataset/image_base.py#L55-L83))
- `question`: question text (default `build_prompt` reads this field, see [build_prompt](/vlmeval/dataset/image_base.py#L283-L291))
- One of the following image fields:
  - `image`: base64 string (`str`) or base64 list (`list[str]`); the list can also be stored as a string `"[...]"` and parsed on load
  - `image_path`: local path (`str`) or list of paths (`list[str]` or string `"[...]"`)
- `answer`: ground truth; may be empty/missing for test-only splits, but evaluation typically needs it
- Optional: `category`, `l2-category`, `split`, `hint`, any custom metadata

### 2) MCQ TSV template

For single-choice MCQ, recommended fields (minimal + common enhancements):

- Required: `index`, `question`, `answer`, `image`/`image_path`
- Option columns: `A`, `B`, `C`, `D`, ... (extendable)

Constraints:

- `answer` should be the option letter directly (`A/B/C/D/...`)
- If you use [ImageMCQDataset](/vlmeval/dataset/image_mcq.py) without a configured LLM judge, it may fall back to “exact matching” (no LLM). In that case, if the model output is not a clean single letter, extraction may fail and the sample may be marked wrong (see ERIQ example and FAQ).

### 3) Fill-in / open-ended VQA TSV template

Recommended fields:

- Required: `index`, `question`, `answer`, `image`/`image_path`
- Recommended: `category`, task/difficulty fields, etc.

Open-ended datasets often need more complex evaluation (regex/rules/external judge). Implement that logic in `evaluate()`.

## Image fields and materialization rules (very important)

### 1) LMUDataRoot and directory layout

The default root for dataset files and images is determined by [LMUDataRoot](/vlmeval/smp/file.py#L84-L90):

- Default: `$HOME/LMUData`
- Override: `LMUData=/path/to/LMUData`

The image base dataset class stores images under:

- `LMUDataRoot()/images/<img_root_map(dataset_name)>/`

In most cases `<img_root_map>` is the dataset name, but some datasets have normalized names (e.g., MMBench/OCRVQA). See [img_root_map](/vlmeval/dataset/image_base.py#L6-L29).

### 2) image: base64 (or base64 list for multi-image)

If the TSV has an `image` column, the base class:

- Supports `image` as base64 string or list (lists can also be stored as a string `"[...]"` and parsed via [toliststr](/vlmeval/smp/misc.py#L284-L291))
- Supports referencing another sample’s image by index to save space: if a row’s `image` length is ≤ 64 and exactly equals another `index`, it is replaced by the referenced sample’s base64 (see [ImageBaseDataset.__init__](/vlmeval/dataset/image_base.py#L59-L71))
- Decodes base64 to files under `self.img_root` on demand during prompt building ([dump_image](/vlmeval/dataset/image_base.py#L217-L248))

Materialized filename strategy (summary):

- Multi-image: prefer `image_path` as the filename; otherwise use `{index}_{i}.png`
- Single image: if no `image_path`, use `{index}.png`

### 3) image_path: directly reference local image paths

If the TSV only provides `image_path` (and no `image`), the dataset is marked as `meta_only=True`. Then `build_prompt` uses paths directly without decoding ([build_prompt](/vlmeval/dataset/image_base.py#L278-L282)).

Path rules:

- `image_path` can be absolute
- or relative; if not found, the framework tries to join it under `self.img_root` ([dump_image](/vlmeval/dataset/image_base.py#L238-L247))

### 4) Too many images: zip packaging + relative paths in TSV

If there are too many images and base64 TSV becomes huge, recommended approach:

- Package images as a zip
- Store only `image_path` in TSV (relative to `LMUDataRoot()/images`)
- In the dataset class, call base `prepare_tsv(..., img_zip=..., img_zip_md5=...)`

On first run, the base class downloads and unzips into `LMUDataRoot()/images/`, and rewrites TSV `image_path` into absolute paths (see [prepare_tsv unzip branch](/vlmeval/dataset/image_base.py#L169-L199)). ERIQ is implemented this way.

## Dataset download, MD5, and “localizing” large files

It is recommended to maintain two tables in the dataset class:

- `DATASET_URL: dict[str, str]`
- `DATASET_MD5: dict[str, str]`

Base `load_data()` downloads the TSV to `LMUDataRoot()` from `DATASET_URL[dataset_name]`, and validates against `DATASET_MD5` (see [load_data/prepare_tsv](/vlmeval/dataset/image_base.py#L261-L207)).

When a TSV file is larger than 1GB, the framework generates and prefers `*_local.tsv` (use `FORCE_LOCAL=1` to refresh). See [prepare_tsv localization logic](/vlmeval/dataset/image_base.py#L201-L206).

## Naming conventions and report

### 1) Naming templates (*_FORMAT)

The base class defines default templates (datasets may override):

- `PRED_FORMAT = "{model_name}_{dataset_name}.tsv"`
- `JUDGE_FORMAT = "{model_name}_{dataset_name}_openai_result.tsv"`
- `RATING_FORMAT = "{model_name}_{dataset_name}_acc.csv"`

See [ImageBaseDataset class attributes](/vlmeval/dataset/image_base.py#L34-L41) and `*_file_basename()` ([pred/judge/rating_file_basename](/vlmeval/dataset/image_base.py#L92-L109)).

The framework infers `(model_name, dataset_name)` from filenames ([extract_model_dataset](/vlmeval/dataset/image_base.py#L110-L118)), so it is recommended to follow `{model}_{dataset}` and ensure `dataset` is a registered supported dataset name.

### 2) report: error rates + metric summary

The base class provides a unified `report()` that summarizes:

- Error rate of `prediction` in inference outputs (empty/NaN/contains FAIL_MSG/“thinking too long”/OpenAI error, etc.), see [is_response_err](/vlmeval/dataset/image_base.py#L298-L305)
- Error rate of judge intermediate files (if a `log` column exists, it counts `log` errors; it also counts `prediction`), see [report_judge_err_rate](/vlmeval/dataset/image_base.py#L318-L339)
- Overall metrics in the rating file, see [report_score](/vlmeval/dataset/image_base.py#L368-L387)

Default outputs root:

- `vlmeval/../outputs/`, override with `MMEVAL_ROOT=/path/to/outputs` ([report](/vlmeval/dataset/image_base.py#L389-L401))

### 3) PRED_FORMAT / EVAL_FORMAT environment variables

You can control prediction/evaluation extensions via env vars (see [get_pred_file_format/get_eval_file_format](/vlmeval/smp/file.py#L185-L202)):

- `PRED_FORMAT`: `tsv` (default) / `xlsx` / `json`
- `EVAL_FORMAT`: `csv` (default) / `json`

## LLM Judge conventions

If evaluation needs an external LLM/Judge:

- Set `DEFAULT_JUDGE` in the dataset class
- Use `build_judge(**judge_kwargs)` inside `evaluate(..., **judge_kwargs)`
- Judge model `model` often supports abbreviations (mapping: [JudgeAbbr_JudgeName](/vlmeval/dataset/utils/judge_util.py#L8-L26))
- Use `JUDGE_ROUTER` to choose judge routing (default/modelcard/openapi/openrouter), see [build_judge](/vlmeval/dataset/utils/judge_util.py#L188-L200)

## Example 1: MathVista (open-ended QA + LLM-assisted answer extraction)

Implementation: [MathVista](/vlmeval/dataset/image_vqa.py#L313-L455).

### 1) Data file fields (TSV)

Besides common fields, MathVista evaluation uses these columns (read in [utils/mathvista.py](/vlmeval/dataset/utils/mathvista.py#L148-L189)):

- `question_type`: distinguishes `multi_choice` from other types
- `choices`: required when `question_type == 'multi_choice'`, must be a Python-list string that can be `eval()`’d
- `task`: used for task-grouped accuracy
- `skills`: used for skill-grouped accuracy (code tries `eval(skills)`; if it fails, it falls back to a single-element list)

Example row (illustrative only):

```text
index: 12
image: "<base64...>"  # or image_path
question: "..."
answer: "..."
question_type: "multi_choice"
choices: "['A. ...', 'B. ...', 'C. ...', 'D. ...']"
task: "geometry"
skills: "['spatial', 'calculation']"
```

Since MathVista is not a pure MCQ dataset, it uses its own storage formats (options in `choices`, the answer in `answer` rather than `A/B/C/D/...`). The dataset class handles this in `build_prompt` and `evaluate`.

### 2) build_prompt

MathVista can usually reuse the base “images + question” prompt ([ImageBaseDataset.build_prompt](/vlmeval/dataset/image_base.py#L273-L292)), unless you want to include more fields (e.g., hints, formatting requirements) into the text prompt.

### 3) evaluate: judge is required

MathVista `evaluate_heuristic` requires `judge_kwargs['model']` to exist and the judge to be usable ([evaluate_heuristic](/vlmeval/dataset/image_vqa.py#L333-L345)):

- The judge extracts the final answer from model `prediction` (with weak equivalence checks)
- Per-sample `res/log/hit` are written back, and accuracies aggregated by `Overall/task/skill` are output ([MathVista_auxeval/MathVista_acc](/vlmeval/dataset/utils/mathvista.py#L148-L189))

Artifacts (summary):

- Intermediate details: `..._{judge}_*.tsv` (per-sample hit/log/res)
- Aggregated scores: `..._{judge}_score.csv` (read by `report_score`)

## Example 2: ERIQ (multi-image interleave + zipped images + MCQ)

Implementation: [ERIQ](/vlmeval/dataset/spatial_easi.py#L8-L44), which subclasses [ImageMCQDataset](/vlmeval/dataset/image_mcq.py).

### 1) Image packaging: TSV stores image_path + download/unzip zip

ERIQ overrides `prepare_tsv` and uses the base unzip branch ([ERIQ.prepare_tsv](/vlmeval/dataset/spatial_easi.py#L28-L30) and [prepare_tsv unzip](/vlmeval/dataset/image_base.py#L169-L199)):

- TSV must contain `image_path`
- TSV must not contain `image`
- `image_path` is recommended to be relative to `LMUDataRoot()/images`; after unzip it is rewritten to absolute paths

### 2) Interleaved prompt: insert images via <image> placeholders

ERIQ uses `question_raw` and supports `<image>` placeholders to control the image/text interleave positions ([ERIQ.build_prompt/build_msgs](/vlmeval/dataset/spatial_easi.py#L31-L63)).

Simplified example:

```text
question_raw: "Consider the following. <image> Now look at another view. <image> What is the answer?"
image_path: ["eriq/0001_a.png", "eriq/0001_b.png"]
```

Generated messages look like (where `$LMUData` is env var `LMUData`, in absolute paths):

```python
[
    dict(type='text', value='Consider the following. '),
    dict(type='image', value='$LMUData/images/eriq/0001_a.png'),
    dict(type='text', value=' Now look at another view. '),
    dict(type='image', value='$LMUData/images/eriq/0001_b.png'),
    dict(type='text', value=' What is the answer?'),
]
```

## FAQ

### 1) Too many images; cannot pack into a single TSV (base64 too large). What to do?

Follow ERIQ’s approach:

- Package images into a zip
- Store only relative `image_path` in TSV
- Use `prepare_tsv(..., img_zip=..., img_zip_md5=...)` to unzip into `LMUDataRoot()/images/`

### 2) How to represent multiple images per sample?

Both approaches are supported:

- Store `image` as a base64 list (or a string `"[...]"`); materialization uses `{index}_{i}.png` unless `image_path` provides filenames ([dump_image](/vlmeval/dataset/image_base.py#L217-L248))
- Store `image_path` as a path list (or a string `"[...]"`); prompt inserts images in order

### 3) My dataset name is not registered in the official list. Can I still run it?

Yes. `build_dataset(dataset_name, **kwargs)` attempts to build a custom dataset from `LMUDataRoot()/{dataset_name}.tsv` for unrecognized names ([build_dataset fallback](/vlmeval/dataset/__init__.py#L413-L433)):

- If there are `A/B` option columns and an image field, it infers Custom MCQ (supports inference and evaluation)
- Otherwise it infers Custom VQA; in that case it supports inference only (no evaluation)

However, for stable `{model}_{dataset}` filename parsing, it is still recommended to include your dataset name in `supported_datasets()` (i.e., register it officially), or ensure it cannot be ambiguous with existing supported dataset names.
