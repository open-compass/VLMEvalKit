# VLMEvalKit-Lite 中文文档总览

本文档是 `docs/zh-CN/` 目录下中文文档的入口，目标是让你在“能跑起来 → 能解释清楚 → 能扩展”的路径上快速定位信息。

如果你是第一次使用，建议按以下顺序阅读：

1. `docs/zh-CN/Quickstart.md`：安装、配置与第一条评测命令
2. `docs/zh-CN/Workflow.md`：推理/评测整体流程、输出文件与复现方式
3. `docs/zh-CN/Model.md`：API / 开源模型接口、消息格式与最佳实践
4. `docs/zh-CN/Dataset.md`：数据集 TSV 规范、图片落盘、评测产物命名
5. `docs/zh-CN/Tools.md`：`vlmutil` 命令行工具（模型检查、评测、文件辅助）
6. `docs/zh-CN/Environment.md`：环境与依赖建议
7. `docs/zh-CN/Troubleshooting.md`：常见问题与排障

## 1. 这是什么

VLMEvalKit（Python 包名 `vlmeval`）是一套用于评测视觉语言模型（VLM）与生成类模型（ULM/UG）的工具链，主要提供：

- 统一的数据集管理：自动下载、缓存、图片落盘、预测文件规范
- 统一的模型输入格式：可交错的多模态 message（image/text/video）
- 推理入口：`run.py`（理解类）、`run_gen.py`（生成类）
- 评测入口：数据集类实现 `evaluate()`；需要 LLM Judge 时可自动调用
- 命令行工具：`vlmutil`（`ve`）辅助列举/检查/评测/整理输出

## 2. 关键概念（建议先理解）

### 2.1 模型分类：VLM / ULM / API

VLMEvalKit 在配置层把模型分为三类（见 `vlmeval/config/__init__.py`）：

- VLM：理解类模型（开源/本地或 API 均可），需要实现 `generate()` 产生文本回答
- ULM：生成类模型（如 T2I/TI2I/TI2TI），需要实现 `generate()` 产生图像或图文混合结果
- API：API 模型集合（很多也会同时出现在 VLM/ULM 中，取决于能力域）

一个直观的经验法则：

- 你在评测“看图回答/选择题/多轮对话/视频理解” → 用 `run.py`（VLM 流程）
- 你在评测“文生图/图像编辑/带图输出的生成任务” → 用 `run_gen.py`（ULM 流程）

### 2.2 数据集对象的职责

一个数据集（benchmark）在代码中对应一个 dataset class，其核心职责是：

- `build_prompt(line)`：把一条样本构建成统一 message 输入格式
- `evaluate(eval_file, **judge_kwargs)`：读取预测文件，计算指标，必要时调用 LLM Judge

更多细节见 `docs/zh-CN/Dataset.md`。

### 2.3 message（多模态输入）标准格式

VLMEvalKit 的核心约定是：模型输入统一表示为一个“可交错”的列表，每个元素是字典：

```python
[
  {"type": "image", "value": "/abs/path/to/img.png"},
  {"type": "text", "value": "Question ..."},
]
```

开源模型与 API 模型都会把用户传入的多种形式（字符串、路径列表、dict、带 role 的对话结构）归一化到该格式后再推理。详见 `docs/zh-CN/Model.md`。

## 3. 入口与工作流

### 3.1 理解类入口：run.py

`run.py` 会按 `(model, dataset)` 组合执行：

1. 构建数据集（支持自动下载/准备）
2. 推理：根据数据集类型分发到不同推理入口
   - 视频：`vlmeval/inference_video.py`
   - 多轮：`vlmeval/inference_mt.py`
   - 其它（图像/通用理解类）：`vlmeval/inference.py`
3. 评测：调用 dataset 的 `evaluate()`，可选 LLM Judge
4. 链接与归档：为“最新结果”建立软链接到 `outputs/{model_name}/`

### 3.2 生成类入口：run_gen.py

`run_gen.py` 针对 `SUPPORT_GEN=True` 的模型与数据集，按 `(model, dataset)` 执行生成与生成评测流程，推理入口在 `vlmeval/inference_gen.py`。

### 3.3 输出目录与复现

输出根目录默认为 `./outputs`，可通过 `MMEVAL_ROOT` 指定。每次运行会生成一个 `eval_id`（包含日期与 git hash），便于复现与隔离不同运行：

```
outputs/{model_name}/T{date}_G{git_hash}/...
outputs/{model_name}/...   # 软链接到最近一次运行产物
```

详细说明见 `docs/zh-CN/Workflow.md`。

## 4. 文档导航（按任务）

### 4.1 我只想跑一个模型

- `docs/zh-CN/Quickstart.md`
- `docs/zh-CN/Tools.md`（`vlmutil check/mlist/dlist`）
- `docs/zh-CN/Environment.md`

### 4.2 我想理解评测产物与如何复现

- `docs/zh-CN/Workflow.md`
- `docs/zh-CN/Dataset.md`（命名规范与 report）
- `docs/zh-CN/Troubleshooting.md`

### 4.3 我想接入一个新模型

- `docs/zh-CN/Model.md`
- `docs/zh-CN/Development.md`（更偏开发流程与示例）

### 4.4 我想接入一个新数据集

- `docs/zh-CN/Dataset.md`
- `docs/zh-CN/Development.md`

### 4.5 我想用配置系统批量跑组合

- `docs/zh-CN/ConfigSystem.md`

## 5. 常见约定（速查）

| 项目 | 建议 |
| --- | --- |
| 并行推理 | 开源模型优先用 `torchrun` 做数据并行；API 模型用 `--api-nproc` 并发 |
| 图片落盘 | 统一使用 dataset 的 `dump_image`（由推理侧注入到模型） |
| 输出格式 | `PRED_FORMAT` 控制预测文件扩展名；`EVAL_FORMAT` 控制评测结果扩展名 |
| 失败样本 | `SKIP_ERR=1` 时推理失败会写入 FAIL_MSG，便于后续统计与重跑 |
