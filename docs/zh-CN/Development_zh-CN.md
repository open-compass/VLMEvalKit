# 🛠️ 如何在 VLMEvalKit 中实现一个新的 Benchmark 或多模态模型（VLM）

## 实现一个新的 benchmark

示例 PR: **添加 Math-Vision Benchmark** ([#292](https://github.com/open-compass/VLMEvalKit/pull/292/files))

### 1. TSV 数据文件准备

目前，我们将每一个 benchmark 数据集设置为一个单独的 TSV 文件。在推理过程中，数据文件将自动下载到 `$LMUData`（如果没有明确设置的话，默认路径是 `$HOME/LMUData`）。你也可以在环境变量中自定义设置`LMUData=/path/to/your/data`。

TSV 文件中的内容组成为：

| 数据集名称 \ 字段  | index | image | image_path | question | hint | multi-choice<br>options | answer | category | l2-category | split |
| ---------------------- | ----- | ----- | ---------- | -------- | ---- | ----------------------- | ------ | -------- | ----------- | ----- |
| MMBench_DEV_[CN/EN]    | ✅     | ✅     |            | ✅        | ✅    | ✅                       | ✅      | ✅        | ✅           | ✅     |
| MMBench_TEST_[CN/EN]   | ✅     | ✅     |            | ✅        | ✅    | ✅                       |        | ✅        | ✅           | ✅     |
| CCBench                | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| SEEDBench_IMG          | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| MME                    | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| CORE_MM                | ✅     | ✅     | ✅          | ✅        |      |                         |        | ✅        |             |       |
| MMVet                  | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMMU_DEV_VAL           | ✅     | ✅     | ✅          | ✅        |      | ✅                       | ✅      | ✅        | ✅           | ✅     |
| COCO_VAL               | ✅     | ✅     |            |          |      |                         | ✅      |          |             |       |
| OCRVQA_[TEST/TESTCORE] | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| TextVQA_VAL            | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| VCR_[EN/ZH]\_[EASY/HARD]_[ALL/500/100]            | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |

<div align="center"><b>表 1. 支持的数据集的 TSV 字段。</b></div>

**TSV 中一些字段的介绍：**

- **index:** 一个整数，`tsv` 中每一行的唯一标识
- **image:** 图片的 base64 编码，你可以使用 `vlmeval/smp.py` 中实现的API进行编码和解码：
    - 编码：`encode_image_to_base64`（对于PIL Image）/ `encode_image_file_to_base64`（对于图片文件路径）
    - 解码：`decode_base64_to_image`（对于PIL Image）/ `decode_base64_to_image_file`（对于图片文件路径）

### 2. 自定义数据集的指标实现

增加对新 benchmark 的评测需要在 `vlmeval/dataset/utils` 中自定义一个该数据集的 class 对象，从而实现数据集的指标计算。图文多模态数据集均继承自 `vlmeval/dataset/image_base.py` 中的 `ImageBaseDataset` 对象。其中 `TYPE` 定义了数据集的类型；`DATASET_URL` 为数据集的下载地址；`DATASET_MD5` 为数据集文件的 md5 一致性编码检查。

在 class 中需要实现 `evaluate(eval_file, **judge_kwargs)` 类函数，对自定义的数据集结果进行指标计算和结果输出。如果需要针对数据集添加 prompt，可以通过实现 `build_prompt(line)` 函数进行实现。给定 TSV 文件中的一行作为 line，该函数生成一个多模态消息 `msg` 的字典 `dict(image=image_path, text=prompt)`，包括图片路径和将被输入到 VLMs 的文本 prompt。

## 实现一个新的模型

示例 PR: **支持 LLaVA-Next-Interleave** ([#294](https://github.com/open-compass/VLMEvalKit/pull/294))

现有所有的模型都在 `vlmeval/vlm` 中实现。对于一个最基本的模型，你的模型类**应该实现方法** `generate_inner(msgs, dataset=None)`。这个函数将向 VLM 输入一个多模态数据，并返回 VLM 的预测（一个字符串）。可选参数 `dataset` 可以用作模型在不同推理策略之间切换的标志。

其中多模态消息 `msgs` 是一个字典列表，每个字典有两个键：类型和值：
- `type`：我们目前支持两种类型，选项是 ["image", "text"]。
- `value`：当类型为 `text` 时，值是文本消息（一个字符串）；当类型为 `image` 时，值可以是图像文件的本地路径，或者是图像的URL。

> 目前，一个多模态消息可能包含任意交错的图像和文本。如果你的模型不支持这一点，我们推荐的做法是取第一张图像和连接的文本消息作为模型的输入。

一些多模态消息的例子:

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='What is in this image?')
]
msg2 = [
    dict(type='image', value=IMAGE_URL),
    dict(type='image', value=IMAGE_URL),
    dict(type='text', value='How many apples are there in these images?')
]
response = model.generate(msg1)
```

为了方便起见，我们还支持接受字符串列表作为输入。在这种情况下，我们将检查一个字符串是图像路径还是图像 URL，并自动将其转换为 `list[dict]` 格式：

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [IMAGE_PTH, 'What is in this image?']
msg2 = [IMAGE_URL, IMAGE_URL,  'How many apples are there in these images?']
response = model.generate(msg1)
```

此外，你的模型可以通过实现两个可选方法来支持自定义提示构建：`use_custom_prompt(dataset)` 和 `build_prompt(line, dataset=None)`。这两个函数都将数据集名称作为输入。`use_custom_prompt` 将返回一个布尔值，指示模型是否应使用自定义提示构建策略。如果它返回 True，`build_prompt` 应该为相应的数据集返回一个自定义构建的多模态消息，line 数据是一个包含数据样本所需信息的字典。如果它返回False，则将使用默认的 prompt 构建策略。

### 示例 PRs：

- 不支持交错的图像和文本，且不使用自定义提示的VLM：[[模型] 支持 glm-4v-9b](https://github.com/open-compass/VLMEvalKit/pull/221)
- 支持交错的图像和文本及自定义提示的VLM：[添加 MiniCPM-Llama3-V-2.5](https://github.com/open-compass/VLMEvalKit/pull/205)
- VLM API：[特征添加 glmv](https://github.com/open-compass/VLMEvalKit/pull/201)

## 为 VLMEvalKit 贡献代码

如果你想为 **VLMEvalKit** 贡献代码，请在提交PR之前进行预提交检查。这有助于保持代码整洁。

```bash
# 在VLMEvalKit的目录下，安装预提交 hook:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# 然后提交你的代码。
```
