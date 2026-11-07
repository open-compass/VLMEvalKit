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

For reference, the original Qwen2.5-VL-3B result contained 2,610 measured
samples from 12 datasets per configuration (one additional sample per
configuration was discarded as warmup):

| Configuration | Mean E2E latency (s) | E2E standard deviation (s) |
| --- | ---: | ---: |
| `image_text_CoT` | 0.7782 | 1.3090 |
| `image_text_image_text_CoT` | 0.7834 | 1.1995 |

TTFT, TTPT, calculated total latency and output token counts are also recorded
for analysis. Total latency is calculated as TTFT plus TTPT multiplied by
the number of remaining output tokens; use the E2E columns above for the
primary result. TTFT and TTPT use vLLM engine steps when available. If engine
stepping fails, the script prints a warning and estimates them from wall time
and output length. MiniCPM's chat-based path estimates TTFT, TTPT and token
counts. These estimated figures are not direct measurements. Compare output
lengths as well as latency, since different generations affect total time.

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
