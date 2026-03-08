# Troubleshooting

This page collects the most common issues and debugging paths when using VLMEvalKit, prioritizing cases that help you locate and recover quickly.

## 1. Minimal pre-run checks

### 1.1 Verify the toolchain works

```bash
python -c "import vlmeval; print(vlmeval.__version__)"
python -c "import vlmeval.tools as t; print(sorted(t.cli.commands.keys()))"
```

### 1.2 Check the model itself first

```bash
vlmutil check {MODEL_NAME}
```

If `check` fails, do not proceed to large-scale evaluation combinations.

## 2. Dataset-related issues

### 2.1 Dataset download is slow or fails

Suggestions:

- Make sure `$LMUData` points to a path with enough space and accessibility
- In some networks, you may need a proxy (especially for LLM Judge in evaluation)
- Start with a small dataset on a single machine/single process to confirm basics

### 2.2 Image field errors (missing paths / decode failure)

Key points:

- Whether the TSV contains `image` (base64) or `image_path`
- Whether `image_path` is absolute or relative; if relative, whether it’s relative to `LMUDataRoot()/images/...`
- Whether image materialization is triggered during `build_prompt()`; check whether images are generated under `LMUData/images/...`

See [/docs/en/Dataset.md](/docs/en/Dataset.md) for the full image materialization rules.

## 3. Inference-related issues

### 3.1 CUDA OOM

Common mitigations:

- Use `python run.py` (single instance, multi-GPU) instead of `torchrun` (multiple instances)
- Reduce `--nproc-per-node` or reduce visible GPUs via `CUDA_VISIBLE_DEVICES`
- If the model supports lower-memory backends (e.g., lmdeploy/vllm for some families), consider switching backend configs

### 3.2 torchrun hangs on barrier / times out

Checklist:

- Ensure all ranks can access the same dataset and output directory (especially on network filesystems)
- Increase distributed timeout:

```bash
export DIST_TIMEOUT=7200
```

- If one rank exits early with an exception, others may hang at a barrier; inspect the earliest failing logs first

### 3.3 API model rate limits / timeouts

Suggestions:

- Reduce concurrency: `--api-nproc`
- Increase `--retry` (and increase wrapper timeouts if supported)
- Verify keys and base URLs are correct (and that `.env` is loaded)

## 4. Evaluation (Judge) issues

### 4.1 Judge cannot be called or proxy does not work

Checks:

- Keys are configured in `.env`
- `EVAL_PROXY` is set (used during evaluation) and network policies allow it
- If only evaluation needs a proxy, configure `EVAL_PROXY` only, without changing inference networking

### 4.2 Scores are unexpectedly low

Common causes:

- Model output format does not match dataset extraction logic (e.g., MCQ expects a clean `A/B/C/D`)
- LLM Judge is not used, falling back to “exact matching” and failing extraction
- Dataset `build_prompt` does not match the model’s prompting strategy (consider enabling model custom prompts)

Open the prediction file (tsv/xlsx) and inspect the `prediction` field to confirm the actual output format.
