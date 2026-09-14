# DIVE-Bench and GRT

This integration adds two scenarios named after the bundled revised manuscript,
**Educational High-FPS Videos** and **High-Motion High-FPS Videos**, together with
the three GRT profiles selected for the 2026-08-20 public leaderboard.

Project: <https://www.zhanghaichao.xyz/DenseVideoUnderstand/>.
Code, numeric release bundle and audit:
<https://github.com/Hai-chao-Zhang/DenseVideoUnderstand/tree/release/dive-bench-minimal>.
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
- High-motion: `haichaozhang/highmotion_densevideounderstand`, revision
  `d44407f607fdf020c59b816884f06ed6d453cf26`, `Egodex_traj.parquet`.
  The owner-authenticated Hub file has SHA-256
  `518e2896749b4d6e957d7e9fb0ae16f75c28954e50ef84303889070253cf8ecd`;
  the historical local serialization has SHA-256
  `39f9da7aca9020d79f383953646a5893f09c6f8e5f60433560011280ee987b2d`.
  Only these two audited byte serializations are accepted. Both contain the
  same 3,243 ordered task rows, independently checked against content SHA-256
  `90ee915016105f6a709f391e8a03a6d0e99bc5c908f945cdf7b80d0cb289e789`
  before selecting the preview. The content fingerprint uses `(video_path, qid,
  question, answer, frame_count)` and is independent of Parquet metadata.

The educational source is gated. On 2026-09-14, an owner-authenticated audit
verified that the pinned high-motion source repository is **private** and downloaded
its annotations; this is not a successful public/anonymous download claim.
Accept the educational Hub terms and obtain access to the private high-motion
source, or supply authorized local annotation files. This PR
does not grant access or redistribute videos. Underlying LPM/YouTube and EgoDex
assets retain their respective source terms; do not infer a common dataset license
from the code license. See the release's `docs/DATASET_RELEASE.md` for details.

On 2026-09-14, the owner also configured the **private** canonical
[`haichaozhang/DIVE-Bench`](https://huggingface.co/datasets/haichaozhang/DIVE-Bench)
repository at revision `d80461fccf879d5efdeece0edce8608a72d64f10` with exactly two
annotation configurations: `educational_high_fps` (634 rows) and
`high_motion_high_fps` (3,243 rows). Named configurations were added to the old
source cards at revisions `c3ff65dfc37239ebee05bd190cfa5b5126f49146` (educational)
and `25cc1aeaef5209776625ce4e72a3ba425d4ae929` (high-motion); their non-card
objects and access settings were unchanged. This adapter retains the original
immutable source-data pins above. Configured annotations do not establish
public access, redistribute videos, or certify end-to-end GPU reproduction.

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
The historical wrapper's frame protocol was not aligned to the eight-frame,
full-clip reference: the source audit found fewer than eight input frames for
787 of its first 1,000 clips and a ten-second truncation affecting 148 clips.
That legacy row is not a fair matched-protocol comparison against eight-frame
baselines and must not support a claim that GRT outperforms them. These counts
audit sampling behavior, not a new GPU generation run; no historical scores or
GRT algorithms are changed here.
Patch-recompute ratios are not total-model FLOPs reductions; `effective_fps` in
the release denotes sampling density, not request throughput. CPU tests verify
registration, annotation handling, metric contracts and the adapter boundary;
they are not GPU inference or benchmark-score measurements.

```bash
pytest -q tests/test_dive_bench.py tests/test_grt.py
```
