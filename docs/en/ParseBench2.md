# ParseBench-2

This adapter uses the public [llamaindex/ParseBench-2 dataset](https://huggingface.co/datasets/llamaindex/ParseBench-2), dataset release **v2.01** at revision `d4e0eaa1b5181f7a84636149eb5717a7aad56c80`: 4,317 cases in 13 subsets. It is separate from ParseBench-1. Dataset versions and scorer package versions are independent.

Dataset v2.01 corrects eight page annotations and per-page scoring options; the source documents and case roster are unchanged. Re-score saved predictions against this same dataset revision before comparing scores. See the [versioned dataset changelog](https://huggingface.co/datasets/llamaindex/ParseBench-2/blob/d4e0eaa1b5181f7a84636149eb5717a7aad56c80/CHANGELOG.md).

At validation on 2026-11-13, the [official leaderboard mapping](https://github.com/run-llama/ParseBench/blob/61d67a0d029a50ea2435a68fd729d46a7c83c082/leaderboard/results-v2.yaml) still pinned dataset v2.0 (`a12c9785cd8cd015b5284e0ad0d74722091762f2`). This adapter targets corrected dataset v2.01. Its scores are not directly interchangeable with those published rows; comparisons require matching dataset and scorer revisions.

Use Python 3.12 or newer and install the optional scorer and renderer:

```sh
pip install -e '.[parsebench2]'
python run.py --data ParseBench2 --model YOUR_SUPPORTED_MODEL --verbose
```

The model must support multiple images per request. Each PDF is rendered in full, at 144 DPI, with `pypdfium2==4.30.0`; all images are RGB. No reference sidecars or annotations enter the input DataFrame, prompt, or model messages. The fixed v1 prompt asks for Markdown plus normalized structured layout. It is an integration prompt, not a claim to reproduce a provider's original benchmark configuration.

Available dataset names are `ParseBench2` and `ParseBench2_SUBSET`, with these subset suffixes: `text`, `headings`, `strikeout`, `underline`, `tables`, `complex_table_headers`, `financial_tables`, `insurance_filing_tables`, `dense_tables`, `charts`, `layout`, `forms`, `fillable_forms`. A subset run is diagnostic and does not establish full benchmark coverage.

Evaluation calls `parse-bench==2.0.1` directly, including its subset aggregation and six-dimension combination. Scores are fractions on the official 0–1 scale; the six dimensions are text, formatting, tables, charts, layout, and forms. Missing or invalid model predictions remain as empty predictions. Duplicate or unknown IDs are errors. Missing table rows are scored as missing predictions against every selected input.

For native structured responses, `bbox` is normalized `x,y,w,h` and is converted to official `layout_segments`. A valid explicit `layout_pages` channel on any response declares layout support for the run; subsequent missing layout predictions keep their official zeros. If no response declares layout support, layout and overall are unavailable, rather than presenting a Markdown-only run as a complete evaluation.

Official saved `InferenceResult` JSON records can also be supplied in the prediction column for replay. Their IDs and pipeline identity are checked, and raw provider payloads are preserved. Replay does not demonstrate fresh inference quality or leaderboard equivalence.

On successful evaluation, the adapter writes `_parsebench2_details.jsonl` and `_score.json` alongside the predictions. These contain per-case status, official metrics, pins, and the official combined report. There is no LLM judge and no provider inference in the scoring path.

## Current reference eligibility limitation

The pinned public references for `text/text_ocr__code` and `text/text_sparse__anex` yield no official extended metrics with the pinned scorer. The full run therefore stops before writing an aggregate. The adapter does not skip these cases, invent annotations, or change the denominator to force a passing comparison. A versioned benchmark-owner reference or eligibility decision is required before claiming complete 4,317-case parity. Subsets without these cases remain available for diagnostics.

Fresh model output still needs to satisfy the structured response contract. Successful execution alone is not evidence of layout quality, full-run parity, or published leaderboard equivalence.
