# GDP.pdf variants

`GDP_pdf` retains the Surge AI rubric protocol and existing page-image adapter.
`GDP_pdf_AA` (`GDPPDFAA`) implements the document inputs and scoring described in
[Artificial Analysis' methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdp-pdf)
(checked November 12, 2026). Both use the same pinned 100-task dataset.
`GDP_pdf_AA_text` uses the same AA preparation and scoring, with complete OCR
text as its document input and no page images.

| Setting | `GDP_pdf` | `GDP_pdf_AA` |
| --- | --- | --- |
| Document input | All page images, 144 DPI | Complete LiteParse English OCR text, plus page images at 150 DPI by default |
| Input order | Task, then images | Page images, then task and complete extracted text in one user message |
| Default judge | Gemini 3.5 Flash | GPT-5.6 Luna, medium reasoning |
| Judge inputs | Answer and one criterion | Original task, answer and one criterion; no PDF or model identity |
| Headline metric | `all_pass` | `all_pass` |
| Secondary metric | `mean_criteria` | `mean_pass`, averaged equally across tasks |
| Missing predictions | Requires complete predictions | Zero, retaining every task in the denominator |
| Incomplete grading | Unscored results excluded from means | No score published until all criteria have verdicts |
| Domain scores | All-pass and Mean Criteria | Mean Pass |

Each evaluator call scores **one attempt per task**. The shared runner repeats
each task five times by default; pass `--n-runs 5` explicitly for this setting,
or `--n-runs 1` for a single attempt. It averages the per-run metrics and adds
`pass@5` / `pass^5`; `all_pass` remains the headline metric. Repetition is handled
by the shared runner, not this adapter. The five retries for malformed judge
JSON are parsing retries, not repeated contestant attempts.

The AA judge reuses the existing Surge criterion prompt with the original task
prepended. Parsing, retry and numeric pass-threshold behavior are shared. A judge
parsing failure remains unscored, saves its diagnostic/cache, and stops evaluation;
rerun evaluation to retry it. It is not treated as a contestant failure. The
original `gdp_pdf.py` implementation is unchanged.

AA does not publish exact prompt templates in the methodology. This adapter
implements the documented message contents and ordering using its own templates;
it does not claim byte-for-byte prompt parity with AA's private harness.

## Preparation

In addition to the existing GDP.pdf requirements (PyMuPDF and parquet support),
the AA variant requires the npm package `@llamaindex/liteparse@2.5.0` and its
English OCR resources. Install it in the runtime environment and ensure `lit`
is available on `PATH`, for example:

```bash
npm install -g @llamaindex/liteparse@2.5.0
```

When OCR extraction is needed, a missing `lit` executable raises an `ImportError`
stating that LiteParse is not installed or is not on `PATH`.
The implementation checks the npm package manifest because this release's
`lit --version` incorrectly prints `2.0.0`. Prepare/download OCR resources on a
machine with network access before running offline.

Source files and versioned extracted-text caches live in `LMUData/{dataset}/`;
images for `GDP_pdf_AA` live in `LMUData/images/GDP_pdf_AA/`. `load_data()` extracts
and validates every page, preserves text whitespace, and invalidates the OCR cache when the PDF
changes. It does not summarize or truncate extracted text. Evaluation also uses
this preparation path.

## Endpoint settings

The dataset cannot infer the target endpoint's vision support, context window,
or request-size limits. Configure these before running inference:

```json
{
  "GDP_pdf_AA": {
    "class": "GDPPDFAA",
    "dataset": "GDP_pdf_AA",
    "dpi": 150,
    "max_images": 20
  }
}
```

Pass this JSON through `run.py --data-config` and select `--data GDP_pdf_AA`.
`max_images` defaults to `null` (no dataset-level image-count cap).
Choose the image DPI/count to fit the endpoint's context and request-size limits.
For text-only input, select `--data GDP_pdf_AA_text`;
when using a data config, set both the entry name and `dataset` to `GDP_pdf_AA_text`.

When capped, images are combined into labeled two-page and then four-page grids.
If four pages per image still exceeds the image-count cap, only leading pages
have images; all pages remain in the text. The prompt describes composite images
and the last page covered. Images remain PNG at the configured DPI; this adapter
does not automatically adjust them to a byte budget. No provider document-input
feature is used.

Contestant generation settings remain runner/model settings: AA specifies
temperature 0 for non-reasoning models, 0.6 for reasoning models unless the model
creator recommends another value; output limits are 16,384 tokens for
non-reasoning models (subject to their limits), and the model's maximum for
reasoning models. Select these explicitly for the contestant model. Judge options
can be overridden through `--judge` / `--judge-args`; changing the model or
reasoning effort changes the evaluation protocol.
