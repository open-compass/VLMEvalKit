# Prompt replay latency test

This experiment evaluates nine models: Qwen2.5-VL 3B, 7B, 32B and 72B;
MiniCPM-V 4.5 and MiniCPM-o 4.5; and Gemma 3 4B, 12B and 27B. It compares
the original image-text prompt (`image_text_CoT`) with the repeated
image-text-image-text prompt (`image_text_image_text_CoT`) using the
repository's replay implementation and vLLM backend.

## Primary metric: end-to-end latency

The primary metric reported for the efficiency experiment is **end-to-end
(E2E) latency**, the wall-clock time in seconds spent in the `model.generate`
call for each sample. It includes processing performed inside that call; it
does not include dataset loading or prompt construction performed before the
call. Use `avg_e2e_latency_s` in `overall_summary.csv` for the sample-level
mean and `std_e2e_latency_s` for its standard deviation. Samples are pooled
across datasets, so larger sampled datasets contribute more to the overall
mean. `per_dataset_summary.csv` contains the corresponding dataset-level
statistics, and `all_latency_results.csv` contains per-sample values in
`e2e_latency_s`.

TTFT, TTPT, calculated total latency and output token counts are also recorded
for analysis. Total latency is calculated as TTFT plus TTPT multiplied by
the number of remaining output tokens; use the E2E columns above for the
primary result. TTFT and TTPT use vLLM engine steps when available. If engine
stepping fails, the script prints a warning and estimates them from wall time
and output length. MiniCPM's chat-based path estimates TTFT, TTPT and token
counts. These estimated figures are not direct measurements. Compare output
lengths as well as latency, since different generations affect total time.

## Results for all nine models

Each row summarizes 2,610 measured samples across 12 datasets, with CoT
enabled. Mean and standard deviation are sample-level E2E latency in seconds.

| Model | Configuration | Mean E2E (s) | E2E std. (s) | Mean output tokens |
| --- | --- | ---: | ---: | ---: |
| Qwen2.5-VL-3B | `image_text_CoT` | 0.7782 | 1.309 | 141.8 |
| Qwen2.5-VL-3B | `image_text_image_text_CoT` | 0.7834 | 1.1995 | 120.8 |
| Qwen2.5-VL-7B | `image_text_CoT` | 0.9546 | 1.1761 | 163.8 |
| Qwen2.5-VL-7B | `image_text_image_text_CoT` | 0.9992 | 1.2257 | 149.2 |
| Qwen2.5-VL-32B | `image_text_CoT` | 5.7278 | 5.6348 | 342.2 |
| Qwen2.5-VL-32B | `image_text_image_text_CoT` | 5.8682 | 5.5018 | 342.1 |
| Qwen2.5-VL-72B | `image_text_CoT` | 5.0112 | 6.3431 | 159.0 |
| Qwen2.5-VL-72B | `image_text_image_text_CoT` | 5.469 | 6.7628 | 161.4 |
| MiniCPM-V-4.5 | `image_text_CoT` | 5.2004 | 4.1229 | 960.4 |
| MiniCPM-V-4.5 | `image_text_image_text_CoT` | 5.3622 | 4.1242 | 951.9 |
| MiniCPM-o-4.5 | `image_text_CoT` | 4.4216 | 3.9454 | 894.8 |
| MiniCPM-o-4.5 | `image_text_image_text_CoT` | 4.3194 | 3.9406 | 841.8 |
| Gemma3-4B | `image_text_CoT` | 1.1264 | 2.2683 | 256.3 |
| Gemma3-4B | `image_text_image_text_CoT` | 1.0822 | 2.0807 | 227.7 |
| Gemma3-12B | `image_text_CoT` | 1.873 | 3.8807 | 225.0 |
| Gemma3-12B | `image_text_image_text_CoT` | 1.9329 | 3.9037 | 219.3 |
| Gemma3-27B | `image_text_CoT` | 3.9642 | 7.8824 | 273.2 |
| Gemma3-27B | `image_text_image_text_CoT` | 3.4744 | 7.1358 | 228.0 |

The additional Qwen timing fields from the recorded results are shown below.
TTFT and TTPT may be estimates when the engine-step timing falls back.

| Model | Configuration | Mean total latency (s) | Total latency std. (s) | Mean TTFT (s) | TTFT std. (s) | Mean TTPT (s/token) | TTPT std. (s/token) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-3B | `image_text_CoT` | 0.7556 | 1.3081 | 0.1545 | 0.1259 | 0.004203 | 0.00023 |
| Qwen2.5-VL-3B | `image_text_image_text_CoT` | 0.7451 | 1.1963 | 0.2329 | 0.1866 | 0.004127 | 0.000316 |
| Qwen2.5-VL-7B | `image_text_CoT` | 0.9316 | 1.1735 | 0.1611 | 0.1302 | 0.004687 | 0.000242 |
| Qwen2.5-VL-7B | `image_text_image_text_CoT` | 0.9609 | 1.2215 | 0.259 | 0.201 | 0.00465 | 0.000222 |
| Qwen2.5-VL-32B | `image_text_CoT` | 5.7045 | 5.6338 | 0.2683 | 0.195 | 0.015985 | 0.000481 |
| Qwen2.5-VL-32B | `image_text_image_text_CoT` | 5.8264 | 5.4994 | 0.4709 | 0.3403 | 0.015643 | 0.000179 |
| Qwen2.5-VL-72B | `image_text_CoT` | 4.988 | 6.3415 | 0.4098 | 0.291 | 0.029055 | 0.000508 |
| Qwen2.5-VL-72B | `image_text_image_text_CoT` | 5.4284 | 6.7594 | 0.7587 | 0.5299 | 0.029057 | 0.000418 |

The source summaries were `overall_summary.csv` files under
`runs/latency_9models_10pct/<model>/`, except Gemma3-12B and Gemma3-27B,
which came from `runs/latency_9models_10pct1/<model>/`. These experiment
outputs are not included in the repository.

## Run all nine models

Install the repository's dependencies and a compatible vLLM environment, make
the models available locally, and prepare the datasets with VLMEvalKit. Some
dataset definitions in this repository use local paths; configure them for your
installation. Edit `scripts/configs/latency_9models.yaml`, especially the
model paths, GPU IDs and tensor parallel sizes. Run from the repository root:

```bash
python scripts/latency_test.py --config scripts/configs/latency_9models.yaml
```

Models run sequentially in separate subprocesses. Results for each model are
written under `results_root/<model_name>/`. To run a subset, remove or comment
out entries in `models`. CLI overrides for the model path, registry, framework,
GPU and tensor parallel size do not override individual `models` entries.
The script defaults Hugging Face to offline mode, so model files must already
be present.

The nine-model YAML specifies the two CoT replay configurations, 12 datasets,
10% sampling with seed 42, batch size 1 and one warmup sample per
configuration. These settings match the recorded Qwen2.5-VL-3B results:
for the 11 datasets directly checked against the available local TSV files,
the sampled indices and their order match the original input records. The
original run did not preserve a complete configuration and environment
snapshot; the YAML uses placeholder paths and adjustable GPU settings, so
exact timing reproduction is not guaranteed. Dataset versions, hardware,
software and generation behavior can affect results.

With this configuration the script passes a 2048-token generation limit to
Qwen and MiniCPM. Gemma uses 4096 by default, configurable through
`GEMMA3_MAX_NEW_TOKENS`. MiniCPM uses its `MINICPM45_VLLM_*` environment
variables for some vLLM settings instead of the shared `vllm` configuration.

## Run Qwen2.5-VL-3B alone

Edit `scripts/configs/latency_qwen25vl_3b.yaml`, then run:

```bash
python scripts/latency_test.py --config scripts/configs/latency_qwen25vl_3b.yaml
```

For a small run, add `--fraction 0.001 --datasets AI2D_TEST`. Other overrides
are listed by `--help`.
