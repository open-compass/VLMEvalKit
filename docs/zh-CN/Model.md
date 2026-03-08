# 模型接口与最佳实践（API / 开源 / 生成类）

本文档说明 VLMEvalKit 中“模型”需要满足的最小接口、统一的输入 message 格式、推理侧如何调用模型、以及实现新模型时的最佳实践（含 transformers 版本建议与常见坑）。

## 1. 模型在框架中的位置

VLMEvalKit 以 `vlmeval.config.build_model(model_name)` 作为统一模型构建入口（见 `vlmeval/config/__init__.py`）。模型大致分为三类：

- 开源/本地理解模型：`vlmeval/vlm/*`，通常继承 `vlmeval.vlm.base.BaseModel`
- API 理解模型：`vlmeval/api/*`，通常继承 `vlmeval.api.base.BaseAPI`（`is_api=True`）
- 生成类模型（文生图/图像编辑等）：`vlmeval/ulm/*`（以及部分 API 生成模型），通常继承 `vlmeval.ulm.base.BaseGenModel`，并设置 `SUPPORT_GEN=True`、`EXPERTISE` 标识能力域

推理入口会按模型属性区分路径：

- API 模型：`getattr(model, "is_api", False) == True`，走并发 API 推理
- 开源模型：否则走本地循环推理

## 2. 统一输入：多模态 message 格式

VLMEvalKit 将模型输入统一抽象为“可交错 message 列表”，每一项是字典：

```python
[
  {"type": "image", "value": "/abs/path/to/img.png"},
  {"type": "text", "value": "Question ..."},
]
```

允许的 `type`：

- 理解模型通常支持：`text`、`image`（视频类模型额外支持 `video`）
- 生成类模型通常支持：`text`、`image`

允许的 `value`：

- `text`：纯字符串
- `image`/`video`：一般是本地文件路径；框架会尽量把 URL/相对路径归一化为本地路径

### 2.1 便捷输入形式

为了方便使用，`generate()` 通常也接受以下输入并自动归一化为 `list[dict]`：

```python
["assets/apple.jpg", "What is in this image?"]
```

或显式 dict：

```python
[{"type": "image", "value": "assets/apple.jpg"},
 {"type": "text", "value": "What is in this image?"}]
```

API 模型还常见支持带 role 的对话结构（会被展开与归一化）：

```python
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": ["assets/apple.jpg", "Describe the image."]},
]
```

## 3. 理解类开源模型（BaseModel）必备接口

开源/本地 VLM 的最小要求是实现 `generate_inner()`，其余由基类负责输入归一化与常见工具函数。

### 3.1 必备：generate_inner

你需要实现：

```python
def generate_inner(self, message: list[dict], dataset: str | None = None) -> str:
    ...
```

其中 `message` 已经被归一化为 `list[dict(type,value)]`，你只需要把它转成你模型后端需要的输入并返回字符串回答。

### 3.2 建议实现：INTERLEAVE

- `INTERLEAVE=True`：模型支持图文交错输入（多图、多段文本混排）
- `INTERLEAVE=False`：模型不支持交错时，推荐在 `generate_inner` 中调用 `message_to_promptimg()` 将其降级为 “第一张图 + 拼接文本”

### 3.3 可选：use_custom_prompt / build_prompt

推理侧会优先使用模型自定义 prompt（若启用）：

- `use_custom_prompt(dataset_name) -> bool`：返回 True 则启用模型自定义 prompt
- `build_prompt(line, dataset=dataset_name) -> message`：把数据集的一条样本行构造成 message

注意：只有当 `use_custom_prompt()` 返回 True 时才会调用 `build_prompt()`。否则框架调用 `dataset.build_prompt()`。

### 3.4 可选：set_dump_image

当模型包含自定义 prompt, 需要在 `build_prompt()` 中把 base64 图片落盘并返回路径时，应实现：

```python
def set_dump_image(self, dump_image_func):
    ...
```

推理侧会把 `dataset.dump_image` 注入进模型，模型即可在 `build_prompt()` 内调用 `self.dump_image(...)`。

## 4. API 模型（BaseAPI）必备接口

API 模型通常继承 `vlmeval.api.base.BaseAPI`，其约定与本地模型不同的地方在于：

- `is_api=True`：推理侧据此进入 API 并发路径
- `generate()` 会负责重试、日志、输入归一化，并调用你实现的 `generate_inner()`

### 4.1 必备：generate_inner（API 版本）

你需要实现：

```python
def generate_inner(self, inputs: list[dict], **kwargs) -> tuple[int, Any, Any]:
    # 返回 (ret_code, answer, log)
```

其中：

- `ret_code`：0 表示成功（框架会据此决定是否重试）
- `answer`：通常是字符串；生成类 API 可能返回 `PIL.Image` 或图文混合结果
- `log`：用于记录 token、时延、费用等统计信息；若用户启用 `keep_stats=True`，`generate()` 会返回 `{"response": answer, "stats": log}`

### 4.2 kwargs 与最佳实践

建议在 API wrapper 中：

- 通过环境变量读取密钥（并支持 `.env`）
- 提供 `retry/timeout/verbose` 等通用参数
- 在 `log` 中记录有用的统计字段（但不要记录敏感信息）

## 5. 生成类模型（BaseGenModel / SUPPORT_GEN）

生成类模型用于 T2I / TI2I / TI2TI 等任务，入口为 `run_gen.py`，推理实现位于 `vlmeval/inference_gen.py`。

### 5.1 关键字段：SUPPORT_GEN 与 EXPERTISE

生成类模型通常需要：

- `SUPPORT_GEN = True`
- `EXPERTISE`：能力域列表，例如：
  - `T2I`：文生图
  - `TI2I`：图像编辑
  - `TI2TI`：图像编辑并输出文字描述等混合任务

`vlmutil check` 会基于这些字段决定对模型做哪些最小可用性测试。

### 5.2 输出类型

生成类模型的 `generate()` 返回值可能是：

- `PIL.Image.Image`
- `str`（例如描述文本）
- `list[str|Image]`（混合输出）

不同数据集会在 `evaluate()` 中定义如何读取/解析这些输出。

## 6. 推理时会如何调用模型（理解路径）

理解类推理大体分两类：

- API 模型：构建 prompt 后并发调用 `model.generate(message=..., dataset=...)`
- 开源模型：逐条构建 prompt、串行/多进程按 rank 切片调用 `model.generate(...)`

当模型实现了 `use_custom_prompt()` 时，推理侧会优先调用 `model.build_prompt()`，否则调用 `dataset.build_prompt()`。

## 7. transformers 版本建议

VLMEvalKit 覆盖大量开源模型，不同模型对 transformers 版本要求不同。建议实践：

- 为“模型家族”固定一个可复现的环境（python/torch/cuda/transformers）
- 在 CI 或团队内部用容器/conda lock 固化依赖
- 接入新模型时，优先在该模型官方推荐的 transformers 版本上实现与验证

## 8. 新增模型实现清单（建议照做）

1. 确定模型类别：理解类 VLM 或生成类 ULM（或 API）
2. 实现最小接口：
   - VLM：`generate_inner()`
   - API：`generate_inner()` 返回 `(ret_code, answer, log)`
   - ULM：`generate_inner()` 并设置 `SUPPORT_GEN=True`
3. 明确 message 支持范围：
   - 是否支持多图、交错、视频
   - 不支持交错时如何降级
4. 如果需要自定义 prompt：
   - 实现 `use_custom_prompt()` + `build_prompt()`
5. 注册到配置：
   - 将类加入 `vlmeval/vlm/__init__.py` 或 `vlmeval/api/__init__.py` / `vlmeval/ulm/__init__.py`
   - 在 `vlmeval/config/*` 中加入到 `supported_VLM / supported_ULM / supported_APIs`
6. 用 `vlmutil check {MODEL_NAME}` 做最小验证
