# 常见问题与排障

本页收集 VLMEvalKit 使用中最常见的问题与排障路径，优先覆盖“能快速定位/能快速恢复”的场景。

## 1. 运行前的最小自检

### 1.1 确认工具链可用

```bash
python -c "import vlmeval; print(vlmeval.__version__)"
python -c "import vlmeval.tools as t; print(sorted(t.cli.commands.keys()))"
```

### 1.2 先检查模型本体

```bash
vlmutil check {MODEL_NAME}
```

如果 `check` 都无法通过，不建议直接跑大规模评测组合。

## 2. 数据集相关问题

### 2.1 数据集下载慢或失败

建议：

- 确认 `$LMUData` 指向有足够空间且可访问的路径
- 在特定网络环境下，可能需要设置代理（尤其评测阶段 LLM Judge）
- 先在单机单进程下跑一个小数据集确认可用

### 2.2 图片字段报错（路径不存在 / 解码失败）

排查要点：

- TSV 是否包含 `image`（base64）或 `image_path`
- `image_path` 是绝对路径还是相对路径；相对路径是否相对于 `LMUDataRoot()/images/...`
- 是否在 `build_prompt()` 过程中触发落盘；可通过查看 `LMUData/images/...` 下是否生成图片文件验证

更完整的图片落盘规则见 `docs/zh-CN/Dataset.md`。

## 3. 推理相关问题

### 3.1 CUDA OOM

常见解决方式：

- 改用 `python run.py`（单实例多卡）而不是 `torchrun`（多实例分卡）
- 降低 `--nproc-per-node` 或减少 `CUDA_VISIBLE_DEVICES` 可见卡数
- 若模型支持更省显存的后端（如某些模型支持 lmdeploy/vllm），考虑切换后端配置

### 3.2 torchrun 卡在 barrier / 超时

排查要点：

- 确认所有 rank 都能访问同一份数据与输出目录（尤其是网络文件系统场景）
- 增大分布式超时：

```bash
export DIST_TIMEOUT=7200
```

- 若某个 rank 提前异常退出，其它 rank 可能会卡在 barrier；优先查看最早异常的日志

### 3.3 API 模型限流/超时

建议：

- 调小并发：`--api-nproc`
- 增加 `--retry`（如模型 wrapper 支持 timeout 参数也可适当调大）
- 检查密钥与 Base URL 是否正确（`.env` 是否被加载）

## 4. 评测（Judge）相关问题

### 4.1 无法调用 Judge 或代理不生效

检查点：

- `.env` 中的 key 是否配置
- 是否设置了 `EVAL_PROXY`（评测阶段使用）并且网络允许
- 仅评测阶段需要代理时，可只配置 `EVAL_PROXY` 而不改推理阶段

### 4.2 评测结果异常偏低

常见原因：

- 模型输出格式不符合数据集抽取规则（例如 MCQ 需要干净的 A/B/C/D）
- 未使用 LLM Judge，落入“exact matching”导致抽取失败
- 数据集的 `build_prompt` 与模型提示策略不匹配（可考虑启用模型自定义 prompt）

建议先打开预测文件（tsv/xlsx）检查 `prediction` 字段的实际输出形式。
