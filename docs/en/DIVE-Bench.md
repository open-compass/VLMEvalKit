# DIVE-Bench and GRT

This integration adds two scenarios named after the bundled revised manuscript,
**Educational High-FPS Videos** and **High-Motion High-FPS Videos**, together with
the three GRT profiles selected for the 2026-08-20 public leaderboard.

Project: <https://www.zhanghaichao.xyz/DenseVideoUnderstand/>.
Code, numeric release bundle and audit:
<https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/2a79fcce2707b1eb74648a5ed135c469b17eb4e1>.
The earlier [arXiv paper](https://arxiv.org/abs/2509.14199) covers the educational
version; it should not be cited as if it contained the later high-motion release.

## Installation

Use Python 3.10 or newer and VLMEvalKit's documented editable installation from
the checkout containing this integration:

```bash
python -m pip install -e .
```

Run the `run.py` commands below from that checkout. A standalone upstream wheel
is not a validated installation route: the audited upstream packaging omits
existing MEGABench parsing modules, so importing that wheel fails before reaching
DIVE-Bench. This integration does not change unrelated upstream packaging.

## Dataset names and access

| Identifier | Scenario | Examples |
| --- | --- | ---: |
| `dive_bench_educational_high_fps` | Educational High-FPS Videos | 634 QA rows / 317 videos |
| `dive_bench_high_motion_high_fps` | High-Motion High-FPS Videos, full release | 3,243 clips |
| `dive_bench_high_motion_high_fps_preview1000` | High-Motion High-FPS Videos, published preview | First 1,000 clips |

Compatibility aliases: `densevideo` selects the educational task;
`densevideo_highmotion` selects the **preview**, not the full high-motion task.
The preview is not affected by `DENSEVIDEO_HIGHMOTION_MAX_EXAMPLES`.

Annotations are selected explicitly and checked by SHA-256:

- Educational: `haichaozhang/DenseVideoEvaluation`, revision
  `5cc61a045c8e5e95d1d9c87e22ccd0f699575aea`, **only** `LPM_videos.parquet`.
  `LPM_slides.parquet` is an identical duplicate and must not be concatenated.
- High-motion: `haichaozhang/highmotion_densevideounderstand`, `Egodex_traj.parquet`.
  A public revision has not yet been established; authorized local release bytes
  are accepted only when they match the recorded SHA-256.

On the release audit date, educational annotations were gated and high-motion
data could not be accessed by the tested public/cached-token accounts. These are
real access prerequisites, not successful-download claims. Accept the educational
Hub terms and authenticate, or supply authorized local annotation files. This PR
does not grant access or redistribute videos. Underlying LPM/YouTube and EgoDex
assets retain their respective source terms; do not infer a common dataset license
from the code license. See the release's `docs/DATASET_RELEASE.md` for details.

Extract videos under `DIVE_BENCH_DATA_ROOT` while preserving source-relative
paths, for example:

```text
data-root/
  DenseVideo-LPM/videos/<video-id>.mp4
  egodex/<action>/<numeric-id>.mp4
```

Educational `videos/<video-id>.mp4` is also supported. High-motion files must
retain action directories because numeric basenames/qids repeat across actions.
Annotations, all selected video paths, counts and reference lengths are checked
before evaluation. The implementation deliberately does not auto-download or
extract a 27 GB archive.

For an editable VLMEvalKit checkout and authorized Hub access:

```bash
export DIVE_BENCH_DATA_ROOT=/path/to/data-root
python run.py --data dive_bench_educational_high_fps --model YOUR_VIDEO_MODEL
```

For local annotations or other frame budgets, use `--data-config`:

```bash
python run.py --data dive_bench_high_motion_high_fps_preview1000 \
  --model YOUR_MULTI_IMAGE_MODEL \
  --data-config '{"dive_bench_high_motion_high_fps_preview1000": {"class": "DIVEBench", "dataset": "dive_bench_high_motion_high_fps_preview1000", "nframe": 8, "annotation_file": "/path/to/Egodex_traj.parquet", "data_root": "/path/to/data-root"}}'
```

The default is eight frames, not processing every high-FPS source frame. Always
report both the frame budget and full/preview split with results.

## Metrics and frame alignment

Educational CER, WER, Token-F1 and Exact Match are arithmetic means of per-example
scores. Normalization lowercases and collapses whitespace, retaining punctuation.
CER/WER may exceed one. The evaluator computes full edit distances, not a fast
mode that silently drops long examples. Predicted answers cannot replace the
authoritative annotation references; missing/duplicate/extra example IDs are
rejected. Explicit failed/empty predictions remain in the denominator.

High-motion targets and decoded image frames use endpoint-inclusive integer
`linspace(0, F-1, min(K,F))`. Grid accuracy, normalized grid-center ADE/FDE,
transition accuracy and Token-F1 reproduce the released objective definitions.
Missing positions incur `sqrt(2)` displacement. High-motion prompts contain the
sample count, never the correct regions. For ordinary models this dataset sends
the sampled **images**. Generic wrappers with `VIDEO_LLM=True` are rejected for
this track, because several dispatch to a video-only path based on dataset modality
or use an incompatible sampler. Use an ordered-image model with `VIDEO_LLM=False`;
this does not claim compatibility with every multi-image model. In particular,
stock LLaVA-OneVision video wrappers and Qwen's vLLM video path are not supported
by this generic High-Motion image-input path. GRT uses its eight-frame video
loader through the custom prompt interface.

There is no inline GPT/MOS score: disabled or failed judge calls are not reported
as zero-quality observations. The official leaderboard's Open MOS uses a separate
pinned Qwen3-VL-32B judge. Use the released `dive-reproduce --with-mos` workflow for
the full matched-control/judge protocol, rather than treating VLMEvalKit objective
results as automatically equivalent to a leaderboard promotion.

## Optional GRT runtime

Registered models are `GRT-LLaVA-OneVision-0.5B` (route31), `GRT-Qwen2.5-VL-3B`
(t0.3), and `GRT-Qwen2.5-VL-7B` (dual-route keep-ratio floors/cap48). Their exact
checkpoint revisions, sampling settings, thresholds and generation caps come from
the release's `tools/densevideo/profiles.json`, without modifying stock VLMEvalKit
wrappers. The adapter passes a real `generate_until` request and a video-only
document to the released model; it does not substitute an ordinary baseline.

Install the minimal runtime in a dedicated environment. It owns the `lmms_eval`
namespace and must not be installed over a separate lmms-eval distribution.
The released commit below is the installation pin. The shared tested runtime contract is
Transformers 4.57.6 / PyTorch 2.9.x with the release's pinned model revisions.

```bash
python -m pip install 'dive-bench @ git+https://github.com/Hai-chao-Zhang/DenseVideoUnderstand.git@2a79fcce2707b1eb74648a5ed135c469b17eb4e1'
export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DIVE_BENCH_DATA_ROOT=/path/to/data-root
python run.py --data dive_bench_educational_high_fps --model GRT-LLaVA-OneVision-0.5B
```

Use one visible CUDA GPU and one process. Sampling is greedy and deterministic
algorithms are enforced; effective Python/NumPy/Torch seeds are 0/1234/1234.
Eight frames are required; frame overrides, vLLM and
unsupported task names fail explicitly. For matched controls use a model config
with class `GRT`, profile `route31`, `qwen3` or `qwen7`, and role `base` or `all`;
the default role is `candidate`. The 7B generation cap is 48 tokens, and the
other two caps are 128 (route31 separately caps subtitle outputs at 31).

These three selected profiles were validated on the educational split. Running
them on high-motion is a new evaluation, **not** a reproduction of the old site's
`grt_llava_ov_0_5b` historical high-motion row, which used a different wrapper.
Patch-recompute ratios are not total-model FLOPs reductions; `effective_fps` in
the release denotes sampling density, not request throughput. CPU tests verify
registration, annotation handling, metric contracts and the adapter boundary;
they are not GPU inference or benchmark-score measurements.

```bash
pytest -q tests/test_dive_bench.py tests/test_grt.py
```
