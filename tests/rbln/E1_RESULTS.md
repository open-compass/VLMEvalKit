# RBLN Backend — Scored Accuracy Parity (judge-free, minimal E1)

This is a single, judge-free score-parity check to show that the RBLN
in-process backend produces benchmark scores comparable to the same model
on GPU / the official report. It is intentionally narrow (one model, one
benchmark, no LLM judge); see *Verified vs. unverified* below.

## Setup
- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` via `RBLNQwen2VL` (`--device rbln`, `tensor_parallel_size=8`).
- **Benchmark:** `ChartQA_TEST` — full **2500** samples.
- **Metric:** `relaxed_accuracy` (VLMEvalKit heuristic scorer — **no LLM judge**, so the score is reproducible without an API key).
- **Hardware:** RBLN NPU (RBLN-CA25), compiled artifact loaded (`export=False`).

## Reference (GPU / official)
| Source | ChartQA relaxed-accuracy |
|---|---|
| Qwen2.5-VL technical report (official) | **87.3%** |
| Independent eval, A100 FP16 ([moured/qwen-vl2.5-chartqa](https://github.com/moured/qwen-vl2.5-chartqa)) | Overall **87.84%** (test_human 80.72 / test_augmented 94.96) |

## RBLN result
_(filled in after the 2500-sample run completes)_

| Split | RBLN relaxed-accuracy |
|---|---|
| test_human | 78.32 |
| test_augmented | 94.80 |
| **Overall** | **86.56** |

| | Overall |
|---|---|
| RBLN (NPU) | **86.56** |
| Reference (GPU/official) | 87.3 – 87.84 |
| Δ | **−0.7 to −1.3 %p** |
| infer_fail_rate | **0.00% (0/2500)** |

**Verdict: score parity established.** RBLN Overall 86.56 sits ~0.7–1.3 %p
below the GPU/official reference (87.3–87.84), within the ±3–5 %p band.
`test_augmented` matches closely (94.80 vs 94.96); the gap concentrates in
`test_human` (78.32 vs 80.72), consistent with minor NPU/GPU numerical
differences rather than a wrapper defect. All 2500 samples
produced a prediction (0% infer-fail). Run: `--device rbln`, compiled
artifact loaded, no LLM judge.

## Tolerance rationale
RBLN NPU inference is not bit-identical to GPU (different hardware/kernels),
and the compiled artifact's `max_seq_len=114688` is below the HF context
length `131072`, which can truncate unusually long inputs on RBLN but not
on GPU. A parity band of **±3–5 percentage points** on Overall is therefore
the success criterion, not exact equality.

## Verified vs. unverified
- **Verified (this doc):** judge-free score parity for Qwen2.5-VL-7B on ChartQA_TEST.
- **Not verified:** judge-based benchmarks (MMVet, MMMU, MMBench, …) scored parity
  — these need an LLM judge (`OPENAI_API_KEY` / `--judge-base-url`); and scored
  parity for the other 10 wrapper families. Per-family **inference** (not scored)
  is exercised by `scripts/rbln_smoke.sh` and was run on the NPU for all families.
