# OmniDocBench v1.6 Evaluation Guide

This document explains how to evaluate **OmniDocBench v1.6** (document parsing / PDF-to-Markdown benchmark) in VLMEvalKit.

v1.6 reuses the official evaluation code (vendored under `odb_v16_vendor/`), including **MGAM adaptive matching**, a **Python CDM**, and the official new **Overall** metric. The data TSV is **built locally** (not downloaded from HuggingFace).

> The legacy `OmniDocBench` (v1.0, 981 pages) is still available with its original evaluation logic for historical reproduction. This guide only covers `OmniDocBench_v1.6`.

---

## Directory layout

```
vlmeval/dataset/OmniDocBench/
├── omnidocbench.py          # Dataset class: registers OmniDocBench / OmniDocBench_v1.6, dispatches by name
├── omnidocbench_v16.py      # v1.6 adapter: bridges the dataframe <-> official pipeline
├── build_v16_tsv.py         # Script that builds the v1.6 TSV locally
├── odb_v16_vendor/src/      # Official v1.6 evaluation code (vendored as-is, incl. MGAM / CDM)
└── requirements.txt         # Dependencies (incl. notes on optional CDM system deps)
```

---

## Step 1: Install dependencies

```bash
# From the VLMEvalKit root
pip install -e .
pip install -r vlmeval/dataset/OmniDocBench/requirements.txt
```

Key new dependencies: `loguru`, `tabulate`, `PyYAML`, `matplotlib`, `evaluate`, `Levenshtein`, `func_timeout`, `apted`, `pylatexenc`.

CDM (the formula score) needs extra **system-level** tools — see [Step 4](#step-4-optional-set-up-cdm-formula-metric).

---

## Step 2: Prepare the official data and build the TSV locally

### 2.1 Download the official v1.6 data

Download the official release (`opendatalab/OmniDocBench`) from OpenDataLab or HuggingFace. You should get:

- `OmniDocBench.json`: GT annotations for 1651 pages (a list of page objects)
- `images/`: the corresponding page images

### 2.2 Build the local TSV

```bash
python -m vlmeval.dataset.OmniDocBench.build_v16_tsv \
    --json /path/to/OmniDocBench.json \
    --image-dir /path/to/images
```

The script converts the data into VLMEvalKit's three-column TSV layout and writes it to `$LMUData/OmniDocBench_v1.6.tsv`:

| Column | Meaning |
| --- | --- |
| `index` | Row id (int) |
| `image` | Base64 of the page image (used during inference) |
| `answer` | Full per-page GT JSON (used during evaluation) |

Optional arguments:
- `--output`: custom output path (defaults to `$LMUData/OmniDocBench_v1.6.tsv`)
- `--fmt`: base64 image encoding format, `JPEG` (default) or `PNG`

On completion it prints the number of pages written, any skipped pages (missing images), and the file MD5. To enable integrity checks, copy the MD5 into `DATASET_MD5['OmniDocBench_v1.6']` in `omnidocbench.py` (optional).

> `$LMUData` defaults to `~/LMUData`; override it with the `LMUData=/your/path` environment variable.

---

## Step 3: Run the evaluation

```bash
export PRED_FORMAT=tsv

# Inference + evaluation
python run.py --data OmniDocBench_v1.6 --model Qwen3.6-35B-A3B --mode all --verbose

# Inference only
python run.py --data OmniDocBench_v1.6 --model Qwen3.6-35B-A3B --mode infer --verbose

# Evaluation only (reuse existing predictions)
python run.py --data OmniDocBench_v1.6 --model Qwen3.6-35B-A3B --mode eval --reuse --verbose
```

If the local TSV is missing, the run fails with a message asking you to build it first (Step 2).

### Output metrics

Results are reported with the official leaderboard columns and saved to `*_v16_score.csv`:

| Column | Meaning | Direction |
| --- | --- | --- |
| `Overall` | Overall score | higher is better |
| `TextEdit` | Text edit distance | lower is better |
| `FormulaCDM` | Formula CDM score | higher is better |
| `TableTEDS` | Table TEDS | higher is better |
| `TableTEDS-S` | Table TEDS (structure only) | higher is better |
| `ReadOrderEdit` | Reading-order edit distance | lower is better |

Overall uses the official formula:

```
Overall = ((1 - TextEdit) * 100 + TableTEDS + FormulaCDM) / 3
```

Detailed per-attribute / per-page breakdowns are saved to `*_v16_metric_result.json` and `*_v16_run_summary.json`.

---

## Step 4 (optional): Set up CDM (formula metric)

**Overall requires the CDM score.** CDM renders LaTeX and needs the following system tools that **cannot be installed via pip**:

- TeX Live (provides `pdflatex`, `kpsewhich`; Chinese formulas also need CJK packages)
- Ghostscript (`gs`)
- ImageMagick (`magick` / `convert`, with PDF read/write enabled)

VLMEvalKit probes for them at runtime:

- **Found** → CDM is computed and the official Overall is reported
- **Not found** → CDM is skipped, `FormulaCDM` and `Overall` show `-`, and an **unofficial** reference value `Overall_no_CDM = ((1 - TextEdit) * 100 + TableTEDS) / 2` is reported with a warning

Using the official Docker image is recommended for a consistent environment:

```bash
docker pull ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204
```

Relevant switches (environment variables):
- `OMNIDOCBENCH_DISABLE_CDM=1`: force-skip CDM even when the toolchain is installed
- You can also pass `enable_cdm=True/False` via `judge_kwargs`

---

## Matching methods

Set via the `OMNIDOCBENCH_MATCH_METHOD` environment variable or the `match_method` field in `judge_kwargs`. Default is `quick_match` (official recommendation):

- `quick_match`: official default, MGAM adaptive-granularity matching with truncation/merge and timeout fallback — the most robust
- `simple_match`: split on double newlines then align one-to-one, fastest; requires the model to segment well
- `no_split`: no segmentation, evaluate the whole page as one document (no per-attribute / reading-order results)

> Note: `quick_match` can be slow on a few complex pages (it has a timeout fallback, default ~300s per page).

---

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `FileNotFoundError: .../OmniDocBench_v1.6.tsv not found` | TSV not built yet — run `build_v16_tsv` (Step 2) |
| `Overall` and `FormulaCDM` show `-` | CDM toolchain not detected — install it (Step 4) or use the official Docker image |
| `ModuleNotFoundError: loguru / apted / ...` | v1.6 deps missing — run `pip install -r requirements.txt` |
| Many "image not found" warnings during build | `--image-dir` is wrong — point it at the folder that actually holds the page images |
| A page is very slow to evaluate | `quick_match` hit the timeout fallback; temporarily switch to `simple_match` to speed up |

---

## Implementation notes

- v1.6 evaluation **fully reuses the official code** (`odb_v16_vendor/src/` vendored as-is). The adapter only restores VLMEvalKit's `answer` (per-page GT JSON) / `prediction` (markdown) into the inputs the official pipeline expects (a GT JSON file + a prediction folder with one `{image_name}.md` per page), runs it, and parses the results.
- The official `skills/` directory is a convenience wrapper for "generate config + run Docker + parse results", not a standalone metric. Its logic is covered by the adapter, so it was not ported separately.
