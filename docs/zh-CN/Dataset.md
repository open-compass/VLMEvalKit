# 数据集格式与数据集类（VLMEvalKit）

本文档介绍 VLMEvalKit 中“数据集（benchmark）”的组织方式、TSV 打包规范、图片字段的处理规则、评测产物命名约定，并通过两个实例讲解如何落地（MathVista 与 ERIQ）。

## 总览：一个数据集是什么

在 VLMEvalKit 中，benchmark 以“数据集类（dataset class）”的形式呈现。数据集类的核心职责是：

- 读取并规范化数据（通常是一个 TSV，对应一个 `pandas.DataFrame`）
- 为每条样本构建模型输入（`build_prompt`）
- 对模型预测文件进行评测并输出指标（`evaluate`）
- （推荐）遵守统一的预测/评测中间文件命名规范，便于框架自动汇总与 `report`

大多数图像数据集继承自 [ImageBaseDataset](/vlmeval/dataset/image_base.py) 或其子类（例如 [ImageMCQDataset](/vlmeval/dataset/image_mcq.py)）。

## 必需接口：build_prompt / evaluate / supported_datasets

### 1) build_prompt(self, line)

基类默认实现见 [ImageBaseDataset.build_prompt](/vlmeval/dataset/image_base.py#L273-L292)。你可以直接复用，也可以覆写以支持更复杂的 interleave/模板。

- 输入 `line`：
  - `int`：表示数据在 `self.data` 中的行号，会被转换为 `pd.Series`
  - `pd.Series` 或 `dict`：一条样本记录
- 输出：VLMEvalKit 统一的“多模态消息格式”，即一个可任意交错的 list，每个元素是 `dict(type=..., value=...)`：

```python
[
    dict(type='text', value='...'),
    dict(type='image', value='/abs/path/to/img.png'),
    dict(type='video', value='/abs/path/to/video.mp4'),
]
```

说明：

- `type` 常见取值是 `image` 与 `text`
- `video` 主要用于 video 类模型（在图像基类里通常不会出现）
- `value` 对 `image` 来说通常是本地路径（也可以是 URL，取决于模型实现）

基类默认策略是：先放入图片（1 张或多张），再放入文本 `question`（见 [ImageBaseDataset.build_prompt](/vlmeval/dataset/image_base.py#L273-L292)）。

### 2) evaluate(self, eval_file, **judge_kwargs)

数据集类必须实现评测函数（抽象方法见 [ImageBaseDataset.evaluate](/vlmeval/dataset/image_base.py#L293-L296)）。

- 输入 `eval_file`：模型推理输出文件路径（通常是 `{model_name}_{dataset_name}.tsv`，具体扩展名受环境变量影响）
- 输入 `judge_kwargs`：当评测需要额外 LLM/Judge 时，用于传递 judge 模型参数（例如 judge 模型名称 `model`、并发 `nproc`、超时时间 `timeout` 等）
- 返回值：评测指标，通常是 `dict` 或 `pd.DataFrame`

实践上，`eval_file` 一般为原始数据集 TSV 文件的副本，新增 `prediction` 列。一般还会移除 `image` 列，以减小文件大小：

### 3) supported_datasets(cls)

用于声明该类“支持的 dataset name 列表”。框架侧会据此完成：

- `SUPPORTED_DATASETS` 汇总与文件名解析（[dataset/__init__.py](/vlmeval/dataset/__init__.py#L355-L363)，以及 [extract_model_dataset](/vlmeval/dataset/image_base.py#L110-L118)）
- `build_dataset(dataset_name, **kwargs)` 的构建（[build_dataset](/vlmeval/dataset/__init__.py#L406-L433)）

基类默认实现会返回 `DATASET_URL` 的 key 列表（[ImageBaseDataset.supported_datasets](/vlmeval/dataset/image_base.py#L256-L260)）。因此一种推荐写法是：只要维护好 `DATASET_URL` / `DATASET_MD5`，就可以复用基类的 `supported_datasets()` 与 `__init__()`。

## 推荐的 TSV 数据打包规范（Best Practice）

VLMEvalKit 推荐把一个 benchmark 打包为一个 TSV 文件，其内容为 `pandas.DataFrame`，并由数据集类负责下载/读取。

### 1) 通用字段（建议所有图像数据集具备）

- `index`：每条样本唯一标识。基类会将其强制转为 `str`（[ImageBaseDataset.__init__](/vlmeval/dataset/image_base.py#L55-L83)）
- `question`：文本问题（默认 `build_prompt` 直接读取该字段，见 [build_prompt](/vlmeval/dataset/image_base.py#L283-L291)）
- 图片字段二选一：
  - `image`：base64 编码（`str`）或 base64 列表（`list[str]`，也允许以 `"[...]"` 的字符串形式存储并在加载时解析）
  - `image_path`：本地路径（`str`）或路径列表（`list[str]` 或 `"[...]"` 字符串形式）
- `answer`：GT。对于 test-only split 可为空/缺失，但评测时通常需要
- 可选字段：`category`、`l2-category`、`split`、`hint`、任意自定义元数据

### 2) 选择题（MCQ）TSV 模板

以单选题为例，推荐字段如下（最小可用 + 常用增强）：

- 必需：`index`, `question`, `answer`, `image`/`image_path`
- 选项列：`A`, `B`, `C`, `D`, ...（可以扩展到更多选项）

重要约束：

- `answer` 直接用选项字母（`A/B/C/D/...`）
- 若你使用 [ImageMCQDataset](vlmeval/dataset/image_mcq.py)，在没有配置好用于 judge 的 API 时，默认 judge 可能是“exact matching”（不启用 LLM），此时模型输出如果不是干净的单字母，很容易抽取失败并判错（详见下文 ERIQ 实例与 FAQ）

### 3) 填空题/开放问答（VQA）TSV 模板

推荐字段如下：

- 必需：`index`, `question`, `answer`, `image`/`image_path`
- 推荐：`category`、子任务字段、难度字段等

开放问答的数据集经常需要更复杂的评测（正则/规则/外部 judge），这些逻辑应在 `evaluate()` 中实现。

## 图片字段与落盘规则（非常重要）

### 1) LMUDataRoot 与目录布局

数据文件与图片的默认根目录由 [LMUDataRoot](/vlmeval/smp/file.py#L84-L90) 决定：

- 默认：`$HOME/LMUData`
- 可通过环境变量覆盖：`LMUData=/path/to/LMUData`

图像基类会将图片落在：

- `LMUDataRoot()/images/<img_root_map(dataset_name)>/`

其中 `<img_root_map>` 在大部分情况下为 dataset_name, 但会对部分数据集名做归一（如 MMBench/OCRVQA 等），见 [img_root_map](/vlmeval/dataset/image_base.py#L6-L29)。

### 2) image: base64（或多图 base64 list）

当 TSV 存在 `image` 列时，基类会：

- 支持 `image` 为 base64 字符串或列表（列表也支持以 `"[...]"` 的字符串形式写入 TSV，最终会用 [toliststr](/vlmeval/smp/misc.py#L284-L291) 解析）
- 支持用“另一个样本的 index”来引用图片以节省空间：当某行 `image` 长度 ≤ 64 且恰好是另一个 `index` 时，会被替换为被引用样本的 base64（见 [ImageBaseDataset.__init__](/vlmeval/dataset/image_base.py#L59-L71)）
- 在构建 prompt 时按需将 base64 解码到 `self.img_root` 下（[dump_image](/vlmeval/dataset/image_base.py#L217-L248)）

落盘命名策略（摘要）：

- 多图：优先使用样本内 `image_path` 作为文件名，否则用 `{index}_{i}.png`
- 单图：若无 `image_path`，用 `{index}.png`

### 3) image_path: 直接引用本地图片路径

当 TSV 仅提供 `image_path`（不提供 `image`）时，基类会把数据集标记为 `meta_only=True`，后续 `build_prompt` 将直接使用路径而不做解码（[build_prompt](/vlmeval/dataset/image_base.py#L278-L282)）。

路径规则：

- `image_path` 可以是绝对路径
- 也可以是相对路径；若读取失败，会尝试拼到 `self.img_root` 下（[dump_image](/vlmeval/dataset/image_base.py#L238-L247)）

### 4) 图片太多：zip 预打包 + TSV 存相对路径

如果图片数量很多、base64 TSV 太大，推荐做法是：

- 图片单独打包成一个 zip
- TSV 中仅存 `image_path`（相对 `LMUDataRoot()/images` 的路径）
- 在数据集类中调用基类 `prepare_tsv(..., img_zip=..., img_zip_md5=...)`

基类会在首次运行时下载并解压 zip 到 `LMUDataRoot()/images/`，并把 TSV 中的 `image_path` 改写为绝对路径（见 [prepare_tsv unzip 分支](/vlmeval/dataset/image_base.py#L169-L199)）。`ERIQ` 数据集的打包就是这样实现的，可以参考。

## 数据集下载、MD5 与“大文件本地化”

推荐在数据集类中维护两张表：

- `DATASET_URL: dict[str, str]`
- `DATASET_MD5: dict[str, str]`

基类 `load_data()` 会根据 `DATASET_URL[dataset_name]` 自动下载 TSV 到 `LMUDataRoot()`，并用 `DATASET_MD5` 校验（见 [load_data/prepare_tsv](/vlmeval/dataset/image_base.py#L261-L207)）。

当 TSV 文件大于 1GB 时，框架会生成并优先使用 `*_local.tsv`（可通过环境变量 `FORCE_LOCAL=1` 强制刷新），见 [prepare_tsv 本地化逻辑](/vlmeval/dataset/image_base.py#L201-L206)。

## 预测/评测产物命名约定与 report

### 1) 命名格式（*_FORMAT）

基类定义了默认模板（可按数据集覆写）：

- `PRED_FORMAT = "{model_name}_{dataset_name}.tsv"`
- `JUDGE_FORMAT = "{model_name}_{dataset_name}_openai_result.tsv"`
- `RATING_FORMAT = "{model_name}_{dataset_name}_acc.csv"`

见 [ImageBaseDataset 类属性](/vlmeval/dataset/image_base.py#L34-L41) 与 `*_file_basename()`（[pred/judge/rating_file_basename](/vlmeval/dataset/image_base.py#L92-L109)）。

注意：框架会通过文件名反推 `(model_name, dataset_name)`（[extract_model_dataset](/vlmeval/dataset/image_base.py#L110-L118)），因此建议遵守 `{model}_{dataset}` 的约定且 `dataset` 必须是已注册支持的数据集名。

### 2) report：错误率 + 指标汇总

基类提供统一的 `report()`，会汇总：

- 推理输出 `prediction` 的错误率（空/NaN/包含 FAIL_MSG/“思考过程过长”/OpenAI error 等），见 [is_response_err](/vlmeval/dataset/image_base.py#L298-L305)
- judge 中间文件的错误率（若存在 `log` 列则统计 `log`；同时统计 `prediction`），见 [report_judge_err_rate](/vlmeval/dataset/image_base.py#L318-L339)
- rating 文件里的 overall 指标，见 [report_score](/vlmeval/dataset/image_base.py#L368-L387)

默认 outputs 根目录：

- `vlmeval/../outputs/`，可通过环境变量 `MMEVAL_ROOT=/path/to/outputs` 覆盖（[report](/vlmeval/dataset/image_base.py#L389-L401)）

### 3) PRED_FORMAT / EVAL_FORMAT 环境变量

推理与评测输出文件扩展名可通过环境变量控制（见 [get_pred_file_format/get_eval_file_format](/vlmeval/smp/file.py#L185-L202)）：

- `PRED_FORMAT`: `tsv`（默认）/ `xlsx` / `json`
- `EVAL_FORMAT`: `csv`（默认）/ `json`

## LLM Judge 的推荐约定

若评测需要外部 LLM/Judge：

- 在数据集类中设置 `DEFAULT_JUDGE`
- 在 `evaluate(..., **judge_kwargs)` 中使用 `build_judge(**judge_kwargs)` 构造 judge
- `judge_kwargs` 中的 `model` 一般支持简写别名（映射见 [JudgeAbbr_JudgeName](/vlmeval/dataset/utils/judge_util.py#L8-L26)）
- 可通过 `JUDGE_ROUTER` 选择 judge 路由（default/modelcard/openapi/openrouter），见 [build_judge](/vlmeval/dataset/utils/judge_util.py#L188-L200)

## 实例 1：MathVista（开放问答 + LLM 辅助抽取答案）

实现位于 [MathVista](/vlmeval/dataset/image_vqa.py#L313-L455)。

### 1) 数据文件字段（TSV）

除通用字段外，MathVista 的评测逻辑还会用到以下列（在 [utils/mathvista.py](/vlmeval/dataset/utils/mathvista.py#L148-L189) 里读取）：

- `question_type`：区分 `multi_choice` 与其他题型
- `choices`：当 `question_type == 'multi_choice'` 时需要，且要求是可被 `eval()` 的 Python 列表文本
- `task`：用于按任务分组统计准确率
- `skills`：用于按技能分组统计准确率（代码会尝试 `eval(skills)`，失败则降级为单元素列表）

一个示意行（非完整）：

```text
index: 12
image: "<base64...>"  # 或 image_path
question: "..."
answer: "..."
question_type: "multi_choice"
choices: "['A. ...', 'B. ...', 'C. ...', 'D. ...']"
task: "geometry"
skills: "['spatial', 'calculation']"
```

注意，这里由于 MathVista 并非纯选择题，所以采取了自己特有的格式存储选项和答案 (选项存为 `choices` 列表，答案直接用 `answer` 字段 (非 `A/B/C/D/...`))。数据集类会在 `build_prompt`, `evaluate` 中对这些情况针对处理。

### 2) build_prompt

MathVista 通常可以复用基类默认的“图片 + question” prompt（[ImageBaseDataset.build_prompt](/vlmeval/dataset/image_base.py#L273-L292)），除非你希望把更多字段（例如 hint、公式格式要求）拼入文本 prompt。

### 3) evaluate：必须提供 judge

MathVista 的 `evaluate_heuristic` 会强制要求 `judge_kwargs['model']` 存在且 judge 可用（[evaluate_heuristic](/vlmeval/dataset/image_vqa.py#L333-L345)）：

- judge 主要用于从模型 `prediction` 中“抽取最终答案”（并做必要的弱等价判断）
- 评测会为每条样本写回 `res/log/hit`，并输出按 `Overall/task/skill` 聚合的准确率表（[MathVista_auxeval/MathVista_acc](/vlmeval/dataset/utils/mathvista.py#L148-L189)）

产物文件（摘要）：

- 中间明细：`..._{judge}_*.tsv`（逐条含 hit/log/res）
- 汇总得分：`..._{judge}_score.csv`（并由 `report_score` 读取）

## 实例 2：ERIQ（多图 interleave + zip 图片包 + MCQ）

实现位于 [ERIQ](/vlmeval/dataset/spatial_easi.py#L8-L44)，继承自 [ImageMCQDataset](/vlmeval/dataset/image_mcq.py)。

### 1) 图片打包方式：TSV 存 image_path + 下载解压 zip

ERIQ 覆写了 `prepare_tsv`，调用基类的 zip 解压分支（[ERIQ.prepare_tsv](/vlmeval/dataset/spatial_easi.py#L28-L30) 与 [prepare_tsv unzip](/vlmeval/dataset/image_base.py#L169-L199)）：

- TSV 必须包含 `image_path`
- TSV 不能包含 `image`
- `image_path` 推荐是相对 `LMUDataRoot()/images` 的相对路径；解压后会被改写为绝对路径

### 2) interleave prompt：用 <image> 占位符插图

ERIQ 使用 `question_raw` 字段，并支持用 `<image>` 控制图文交错位置（[ERIQ.build_prompt/build_msgs](/vlmeval/dataset/spatial_easi.py#L31-L63)）。

简化示例：

```text
question_raw: "Consider the following. <image> Now look at another view. <image> What is the answer?"
image_path: ["eriq/0001_a.png", "eriq/0001_b.png"]
```

生成的消息大致形如 ($LMUData 为环境变量 `LMUData` 的值，绝对路径）：

```python
[
    dict(type='text', value='Consider the following. '),
    dict(type='image', value='$LMUData/images/eriq/0001_a.png'),
    dict(type='text', value=' Now look at another view. '),
    dict(type='image', value='$LMUData/images/eriq/0001_b.png'),
    dict(type='text', value=' What is the answer?'),
]
```

## FAQ

### 1) 图片太多，难以打包成单个 TSV（base64 太大）怎么办？

参考 ERIQ 的做法：

- 图片打包成 zip
- TSV 中只存相对路径 `image_path`
- 在数据集类中用 `prepare_tsv(..., img_zip=..., img_zip_md5=...)` 解压到 `LMUDataRoot()/images/`

### 2) 一个样本多张图怎么写？

两种方式都支持：

- `image` 写为 base64 list（或 `"[...]"` 字符串形式）；落盘时会按 `{index}_{i}.png` 命名，或用 `image_path` 提供文件名（[dump_image](/vlmeval/dataset/image_base.py#L217-L248)）
- `image_path` 写为路径 list（或 `"[...]"` 字符串形式）；prompt 会按顺序插入多张 `image`

### 3) 我没有在官方列表里注册数据集名，能直接跑吗？

可以。`build_dataset(dataset_name, **kwargs)` 会对“非官方支持”的 `dataset_name` 尝试从 `LMUDataRoot()/{dataset_name}.tsv` 构建 Custom 数据集（[build_dataset fallback](/vlmeval/dataset/__init__.py#L413-L433)）：

- 若存在 `A/B` 选项列且有图片字段，则推断为 Custom MCQ，支持推理与评估
- 否则推断为 Custom VQA，注意，这种情况下只支持推理，不支持评估

但注意：为了让 `{model}_{dataset}` 文件名解析稳定，仍建议将数据集名纳入 `supported_datasets()`（即注册到官方支持列表），或确保你的数据集名不会与其他已支持数据集名发生歧义。
