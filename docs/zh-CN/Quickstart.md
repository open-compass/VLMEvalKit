# 快速开始

本文档面向首次使用 VLMEvalKit 的用户，覆盖安装、API 密钥配置、模型/数据集选择、推理与评测、输出目录结构与常见环境变量。更深入的结构与开发细节请参阅：

- 项目入口文档：`docs/zh-CN/README.md`
- 模型接口与最佳实践：`docs/zh-CN/Model.md`
- 数据集格式与规范：`docs/zh-CN/Dataset.md`
- 配置系统：`docs/zh-CN/ConfigSystem.md`
- CLI 工具：`docs/zh-CN/Tools.md`

## 1. 安装

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```

建议使用独立的 Python 环境，并保证与模型依赖（如 transformers）兼容。当前包要求 `python>=3.7`（详见 `setup.py`）。

## 2. 设置 API 密钥（仅 API 模型或 Judge 需要）

API 模型和 LLM Judge 需要密钥。你可以：

- 在仓库根目录创建 `.env` 文件
- 或直接设置环境变量

`.env` 示例（仅列出常见项，可按需补充）：

```bash
DASHSCOPE_API_KEY=
GOOGLE_API_KEY=
OPENAI_API_KEY=
OPENAI_API_BASE=
STEPAI_API_KEY=
GLMV_API_KEY=
CW_API_BASE=
CW_API_KEY=
SENSENOVA_API_KEY=
HUNYUAN_SECRET_KEY=
HUNYUAN_SECRET_ID=
EVAL_PROXY=
```

程序会在入口脚本中自动加载 `.env`。

## 3. 查看可用模型 / 数据集

`vlmutil` 是命令行工具入口（等价别名为 `ve`）：

```bash
vlmutil mlist VLM     # 可用理解类模型
vlmutil mlist ULM     # 可用生成类模型
vlmutil mlist API     # 可用 API 模型
vlmutil dlist all     # 所有支持数据集
vlmutil dlist MMBench # 数据集组（若存在）
```

如需快速检查模型可用性：

```bash
vlmutil check InternVL3-8B
```

## 4. 评测理解类模型（run.py）

理解类评测入口为 `run.py`，支持图像、视频、多轮对话数据集。

### 4.1 基本用法

```bash
python run.py --data MMBench_DEV_EN MME --model InternVL3-8B --verbose
```

仅推理不评测：

```bash
python run.py --data MMBench_DEV_EN --model InternVL3-8B --mode infer
```

仅评测（已有预测文件时）：

```bash
python run.py --data MMBench_DEV_EN --model InternVL3-8B --mode eval
```

### 4.2 多卡并行

`torchrun` 会启动多个进程，每个进程实例化一个模型实例。每个实例分配 `N_GPU // N_PROC` 张卡（详见 `run.py` 启动逻辑）。

```bash
torchrun --nproc-per-node=4 run.py --data MME --model InternVL3-8B
```

### 4.3 使用配置文件

当模型/数据集组合复杂时，推荐用 `--config`：

```bash
python run.py --config config.json
```

配置格式详见 `docs/zh-CN/ConfigSystem.md`。

## 5. 评测生成类模型（run_gen.py）

当数据集类型为生成类（如 T2I/TI2I/TI2TI），必须使用 `run_gen.py`：

```bash
python run_gen.py --data DPGBench --model Janus-Pro-7B
```

`run_gen.py` 支持 `--num-generations` 用于每条样本多次采样，详细说明见 `docs/zh-CN/Workflow.md`。

## 6. 输出目录结构

默认输出目录为 `./outputs`，也可通过 `MMEVAL_ROOT` 修改：

```bash
export MMEVAL_ROOT=/path/to/outputs
```

理解类评测的典型输出路径：

```
outputs/
  {model_name}/
    T{date}_G{git_hash}/
      {model_name}_{dataset}.tsv
      {model_name}_{dataset}_{judge}_score.csv
```

生成类评测输出路径结构类似，但会包含额外的 sample/instance 级记录文件。

## 7. 常用环境变量

| 变量名 | 作用 |
| --- | --- |
| `LMUData` | 数据集下载/缓存根目录，默认 `~/LMUData` |
| `MMEVAL_ROOT` | 推理与评测输出根目录 |
| `PRED_FORMAT` | 预测文件格式（默认 tsv，可选 xlsx/json） |
| `EVAL_FORMAT` | 评测结果格式（默认 csv，可选 json） |
| `EVAL_PROXY` | 评测阶段 API 代理 |
| `DIST_TIMEOUT` | 分布式超时（秒） |
| `SKIP_ERR=1` | 推理异常时跳过并记录错误 |

## 8. 常见问题

### 8.1 评测结果差异

不同运行环境（transformers、torch、CUDA 等版本）会带来一定差异。建议保留 `outputs` 中的中间记录以便复现与排查。

### 8.2 模型不支持图文交错

若模型 `INTERLEAVE=False`，框架会自动降级为“第一张图像 + 拼接文本”的输入形式。可在自定义模型中通过 `message_to_promptimg` 处理。
