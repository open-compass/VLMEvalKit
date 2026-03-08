# 工作流与产物说明（理解评测 / 生成评测）

本文档解释 VLMEvalKit 的端到端流程：从构建数据集与模型开始，到推理、评测、输出文件与复现实验的组织方式。它面向“跑通后想把流程讲清楚/自动化”的用户。

## 1. 两条主流程：run.py 与 run_gen.py

VLMEvalKit 把评测流程分为两类入口：

- 理解类评测：`run.py`
  - 适用：选择题、VQA、OCR、视频理解、多轮对话、工具调用类等（输出通常是文本）
- 生成类评测：`run_gen.py`
  - 适用：文生图（T2I）、图像编辑（TI2I）、图像编辑并输出文本（TI2TI）等（输出可能是图像或图文混合）

如果你尝试用 `run.py` 跑生成类数据集，入口会显式报错提示改用 `run_gen.py`。

## 2. run.py（理解类）执行顺序

对每个 `model_name` 与每个 `dataset_name`，`run.py` 会做：

1. 生成一次运行的 `eval_id = T{date}_G{git_hash}`（用于复现与隔离不同批次）
2. 构建本次运行目录：
   - `pred_root = {work_dir}/{model_name}/{eval_id}`
   - `pred_root_meta = {work_dir}/{model_name}`（用于软链接“最新结果”）
3. 构建数据集对象 `dataset = build_dataset(dataset_name, **kwargs)`
4. 推理：
   - 视频数据集：`vlmeval/inference_video.py`
   - 多轮数据集：`vlmeval/inference_mt.py`
   - 其它：`vlmeval/inference.py`
5. 评测：
   - 仅在 `RANK==0` 进行
   - 通过 `dataset.evaluate(result_file, **judge_kwargs)` 计算指标
6. 软链接与归档：将本次运行产物链接到 `pred_root_meta` 便于快速访问

## 3. run_gen.py（生成类）执行顺序

生成类流程与理解类类似，但推理入口为 `vlmeval/inference_gen.py`，并且：

- 模型与数据集必须满足 `SUPPORT_GEN=True`
- 常见会产生 sample/instance 粒度的中间记录（用于多次采样与聚合）

## 4. 推理阶段的核心产物

### 4.1 预测文件（Prediction）

预测文件是最核心的产物，典型命名为：

```
{model_name}_{dataset_name}.{PRED_FORMAT}
```

默认 `PRED_FORMAT=tsv`，也可通过环境变量切换为 `xlsx/json`。预测文件通常包含：

- 原始样本字段（index/question/answer/选项/元数据等）
- `prediction`：模型输出（字符串或可序列化结构）
- 对某些数据集还可能包含 `stats`、`raw_prediction`、`thinking` 等辅助字段

PS: 默认选择 `tsv` 格式存储的原因是：

- 表格格式易读性好，并确保每条样本包含相同字段，可利用 pandas 进行读取写入
- 不选 `xlsx`，是因为 excel 单个 cell 存在长度上限，模型回复过长时会超出这一上限，产生阶段

### 4.2 分布式中间文件（Pickle shards）

当使用 `torchrun` 多进程推理时，推理阶段往往会产生每个 rank 的中间 pkl 文件用于汇总。汇总完成后通常会清理这些文件。

当你在进行开源模型评测过程中，中断任务且重启时选用了不同的 `WORLD_SIZE`, 此时可能需要手动合并 pkl 文件，可使用：

```bash
vlmutil merge_pkl /path/to/pred_root --world-size {WORLD_SIZE}
```

### 4.3 断点续跑（PREV）

推理入口通常会使用 `*_PREV.pkl` 来缓存“已经完成且有效”的预测，从而实现断点续跑：

- 若发现已有 `result_file`，会提取其中有效预测写入 PREV
- 后续推理只对缺失或无效样本继续生成

## 5. 评测阶段的核心产物

评测阶段由数据集类的 `evaluate()` 负责，对于 MCQ 类评测集，常见产物包括：

- `*_score.csv`：汇总指标（overall 与分组指标）
- `*_result.tsv` / `*_openai_result.tsv`：逐条 judge 结果与日志（仅当需要 LLM Judge）

不同数据集实现差异较大，产物格式在数据集类中定义，具体以数据集类为准。

## 6. 复现实验的建议方式

为了让结果可复现，建议记录：

- 代码版本：git commit hash（框架会写入 eval_id）
- 运行环境：python/torch/cuda/transformers 版本
- 运行参数：`run.py`/`run_gen.py` 命令行
- 关键环境变量：`LMUData`、`MMEVAL_ROOT`、`PRED_FORMAT`、`EVAL_PROXY` 等

经验上，最稳妥的做法是用容器或 conda 环境固化依赖，并在团队内共享一套“模型家族 → 环境”的映射。

## 7. 常用调试手段

- `vlmutil check {MODEL}`：先验证模型最小可用性
- `--mode infer`：先只跑推理，确认预测文件内容合理再评测
- `vlmutil eval <pred_file>`：对单个预测文件进行评测打分
- `SKIP_ERR=1`：遇到个别样本报错时不中断整批任务 (针对开源模型)
- `--api-nproc`：调小并发排查 API 限流与超时
