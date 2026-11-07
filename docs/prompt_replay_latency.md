# Prompt replay latency test

Compare Qwen2.5-VL inference on the original image-text prompt with the
repeated image-text-image-text prompt. The script uses this repository's
replay implementation and vLLM backend.

Install the repository's dependencies and a compatible vLLM environment, make
the model available locally, and prepare the datasets with VLMEvalKit. Some
dataset definitions in this repository use local paths; configure them for your
installation. Edit `scripts/configs/latency_qwen25vl_3b.yaml`, especially
`model_path`, GPU settings, datasets and sampling fraction. Run from the repo
root:

```bash
python scripts/latency_test.py --config scripts/configs/latency_qwen25vl_3b.yaml
```

For a small run, add `--fraction 0.001 --datasets AI2D_TEST`. Other overrides
are listed by `--help`. The script defaults Hugging Face to offline mode, so
model files must already be present. Results are written under `results_root`:
`overall_summary.csv` contains sample-level averages,
`per_dataset_summary.csv` contains dataset-level averages, and
`all_latency_results.csv` contains per-sample timings and output previews.
The first sample of each configuration is excluded as warmup. Sampling is
seeded with 42.

`e2e_latency_s` measures the `model.generate` call. TTFT and TTPT use vLLM
engine steps when available; total latency is calculated as TTFT plus TTPT
multiplied by the number of remaining output tokens. If engine stepping fails,
the script prints a warning and estimates TTFT/TTPT from wall time and output
length. These fallback figures are not direct measurements. Compare output
lengths as well as latency, since different generations affect total time.

The example uses 10% sampling and the two CoT modes from the Qwen2.5-VL-3B
experiment. The exact original environment, dataset versions, sampled indices,
and hardware are not captured in this configuration, so it is a starting
point rather than a guarantee of reproducing the original numbers.
