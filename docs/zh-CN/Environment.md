# 环境与依赖建议

VLMEvalKit 覆盖的模型与数据集非常多，依赖矩阵会随着开源模型生态变化而变化。本页提供一套“可复现优先”的环境组织建议，帮助你在团队/多机环境里稳定跑通评测。

## 1. 基本原则

1. 按“模型家族”固化环境，而不是试图用一个环境跑所有模型。
2. 固化四件事：Python、PyTorch、CUDA、transformers（以及必要的 vision 依赖），请参考待评测模型的官方要求配置环境，以免产生非预期的结果。
3. 评测结果复现时，尽量同时记录 git commit (代码库默认已开启) 与环境版本。

## 2. 推荐的环境布局

### 2.1 多环境策略（推荐）

以 conda/venv 为单位，为不同模型家族创建不同环境：

- `vlmeval-tf433`：兼容较老的 Qwen/IDEFICS 等家族
- `vlmeval-tf437`：兼容部分 LLaVA/InternVL/DeepSeek-VL 等家族
- `vlmeval-latest`：跟随最新 transformers，用于新模型或你确认兼容的模型

### 2.2 单环境策略（谨慎）

如果你必须用单环境：

- 只选择你确定要评测的少量模型家族
- 在跑大规模组合前，用 `vlmutil check` 对每个模型做最小验证

## 3. torchrun / WORLD_SIZE 相关注意事项

在 `torchrun` 场景下，框架采用“多进程多实例”策略：每个进程一个模型实例、各自分配一部分 GPU。该模式对显存较小/可多实例并发的模型更友好。

但在较新的 transformers 中，某些 `device_map="auto"` 逻辑会在 torchrun 环境下自动启用 TP 并行，可能与“多实例”策略冲突。框架在部分路径会临时移除 `WORLD_SIZE` 以规避该问题。

实践建议：

- 想跑单机、单实例、多卡（TP/自动切分）→ 用 `python run.py` 并控制 `CUDA_VISIBLE_DEVICES`
- 想跑单机、多实例并发（DP/多进程）→ 用 `torchrun --nproc-per-node=N`
- 想跑多机、多实例并发 → 基于部署框架 (如 vLLM) 先在多机上完成部署，再利用 API 方式调用进行评估

## 4. 依赖排查建议

当你遇到导入错误或版本冲突时，建议按顺序排查：

1. `python -c "import torch; print(torch.__version__)"`
2. `python -c "import transformers; print(transformers.__version__)"`
3. `vlmutil check {MODEL_NAME}`（最小输入验证）
4. 若仅某个模型失败，考虑为该模型家族切换到独立环境
