# 命令行工具：vlmutil（ve）

VLMEvalKit 提供命令行工具 `vlmutil`（别名 `ve`），入口定义在 `setup.py`，实现位于 `vlmeval/tools/`。本节给出所有主要命令与使用场景。

## 1. 基本用法

```bash
vlmutil -h
vlmutil <command> -h
```

## 2. 模型与数据集列表

### 2.1 模型列表

```bash
vlmutil mlist VLM   # 理解类模型
vlmutil mlist ULM   # 生成类模型
vlmutil mlist API   # API 模型
```

### 2.2 数据集列表

```bash
vlmutil dlist all            # 所有数据集
vlmutil dlist MMBench        # 数据集组（若存在）
```

`dlist` 会优先查 `DATASET_GROUPS`；否则输出全部 `SUPPORTED_DATASETS`。

## 3. 模型最小可用性检查

```bash
vlmutil check InternVL3-8B
```

该命令会用 `assets/apple.jpg` 构造若干条最小输入：

- 单图单问
- 多图多问
- 生成类模型会追加 T2I / TI2I / TI2TI 相关检查（取决于 `EXPERTISE`）

用于快速验证模型是否可运行。

## 4. 评测与评测结果工具

### 4.1 直接评测打分一个预测文件

```bash
vlmutil eval /path/to/{model}_{dataset}.tsv --judge GPT4o --api-nproc 32
```

常用选项：

- `--judge`：指定评测用 LLM
- `--api-nproc`：并发数
- `--retry`：失败重试次数
- `--verbose`：输出更详细日志
- `--rerun`：删除已有 judge 产物后重新评测

### 4.2 快速读取历史结果

```bash
vlmutil print_acc {model_name} {dataset_name} --root /path/to/outputs
```

用于从已有结果文件中汇总/显示评分。

## 5. 运行入口代理

```bash
vlmutil run --data MMBench_DEV_EN --model InternVL3-8B --verbose
```

`vlmutil run` 会把参数透传给 `run.py`，便于在任何目录调用。

## 6. 数据与结果辅助命令

### 6.1 circular：选择题循环增强

```bash
vlmutil circular /path/to/data.tsv
```

对多选题数据集进行 “circular” 变换，输出 `*_circular.tsv`。

### 6.2 localize：大文件本地化

```bash
vlmutil localize /path/to/data.tsv
```

将 TSV 本地化为 `_local.tsv`，便于处理超大文件或路径问题。

### 6.3 merge_pkl：合并分布式推理结果

```bash
vlmutil merge_pkl /path/to/pkl_dir --world-size 4
```

用于合并多卡推理阶段产生的分片 pkl 文件。

### 6.4 missing：检查评测组缺失数据集

```bash
vlmutil missing /path/to/outputs/{model_name} --group MMBench
```

用于查看某个数据集组中哪些任务尚未完成。

### 6.5 fetch：提取某个数据集的所有结果

```bash
vlmutil fetch MMBench --source /path/to/outputs --target /path/to/debug
```

用于收集与指定数据集相关的预测与评测文件。

### 6.6 upload / upload_data：上传到 TOS

```bash
vlmutil upload /path/to/file --bucket vlmeval-data --prefix data
vlmutil upload_data /path/to/file
```

需要配置相应的 TOS 访问密钥（见环境变量或 `.env`）。

## 7. 命令速查表

| 命令 | 作用 |
| --- | --- |
| `mlist` | 列出模型 |
| `dlist` | 列出数据集 |
| `check` | 快速验证模型可用性 |
| `eval` | 评测一个预测文件 |
| `print_acc` | 从已有结果中汇总分数 |
| `run` | 代理执行 `run.py` |
| `circular` | 多选题数据集循环增强 |
| `localize` | TSV 本地化 |
| `merge_pkl` | 合并分布式推理中间文件 |
| `missing` | 检查评测组缺失任务 |
| `fetch` | 收集指定数据集结果 |
| `upload` | 上传文件到 TOS |
| `upload_data` | 上传数据文件到 TOS |
