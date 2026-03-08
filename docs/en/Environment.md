# Environment and Dependency Recommendations

VLMEvalKit covers a wide range of models and datasets, and the dependency matrix changes with the open-source ecosystem. This page provides reproducibility-first guidance to help teams run evaluations reliably across machines and environments.

## 1. Principles

1. Pin environments per “model family”, rather than trying to run all models in one environment.
2. Pin four key components: Python, PyTorch, CUDA, and transformers (plus any required vision deps). Follow the target model’s official requirements to avoid unexpected behavior.
3. For reproducibility, record both the git commit (enabled by default) and environment versions.

## 2. Recommended environment layout

### 2.1 Multi-environment strategy (recommended)

Use conda/venv to create separate envs for different model families:

- `vlmeval-tf433`: compatible with older Qwen/IDEFICS families
- `vlmeval-tf437`: compatible with some LLaVA/InternVL/DeepSeek-VL families
- `vlmeval-latest`: track latest transformers for new models you have verified

### 2.2 Single-environment strategy (use with caution)

If you must use a single environment:

- Only include a small set of model families you definitely need
- Before large-scale runs, validate each model with `vlmutil check`

## 3. torchrun / WORLD_SIZE notes

In `torchrun` scenarios, the framework uses a “multi-process, multi-instance” strategy: one model instance per process, each using a subset of GPUs. This is often better for smaller-memory models or cases where multiple instances are feasible.

However, in newer transformers versions, some `device_map="auto"` logic can auto-enable TP parallelism under torchrun, which may conflict with the “multi-instance” strategy. The framework removes `WORLD_SIZE` temporarily in some paths to avoid this issue.

Practical guidance:

- Single-machine, single instance, multi-GPU (TP/auto sharding) → use `python run.py` and control `CUDA_VISIBLE_DEVICES`
- Single-machine, multiple instances in parallel (DP/multi-process) → use `torchrun --nproc-per-node=N`
- Multi-machine, multiple instances → deploy via a serving framework (e.g., vLLM) and evaluate via API calls

## 4. Dependency debugging checklist

When you hit import errors or version conflicts, check in order:

1. `python -c "import torch; print(torch.__version__)"`
2. `python -c "import transformers; print(transformers.__version__)"`
3. `vlmutil check {MODEL_NAME}` (minimal-input validation)
4. If only one model fails, consider switching to a dedicated environment for that model family
