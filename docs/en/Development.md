# 🛠️ How to Implement a New Benchmark or Multimodal Model (VLM) in VLMEvalKit

## Implementing a new benchmark

Example PR: **Add Math-Vision Benchmark** ([#292](https://github.com/open-compass/VLMEvalKit/pull/292/files))

In VLMEvalKit, benchmarks are implemented as dataset classes. When adding a new benchmark, you can either reuse an existing dataset class (e.g., reuse `ImageMCQDataset` for single-choice MCQ benchmarks) or implement a new dataset class. Your dataset class must support the following two methods (either inherited from a base class or implemented yourself):

- `build_prompt(self, line)`: input `line` is an `int` (row index in data) or a `pd.Series` (raw record). Output is one “multi-modal message” as the model input. A “multi-modal message” is an interleaved image/text list, for example (one image + one text): `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`.
- `evaluate(self, eval_file, **judge_kwargs)`: input `eval_file` is the model prediction file (often `.xlsx`), and `judge_kwargs` provides judge model parameters when evaluation needs an LLM (often GPT). Output is evaluation results in `dict` or `pd.DataFrame`.

Below is a typical workflow for adding a dataset:

### 1. Prepare the TSV data file (image-text benchmarks)

Currently, each benchmark is packaged as a single TSV file. During inference, the data file is automatically downloaded to `$LMUData` from the dataset’s `DATASET_URL` (default path is `$HOME/LMUData` if not explicitly set). You can upload the TSV to a downloadable location (e.g., HuggingFace), or email it to <opencompass@pjlab.org.cn> for help uploading to the server. You can also customize the download path via `LMUData=/path/to/your/data`.

TSV fields typically look like:

| Dataset \\ Field  | index | image | image_path | question | hint | multi-choice<br>options | answer | category | l2-category | split |
| ---------------------- | ----- | ----- | ---------- | -------- | ---- | ----------------------- | ------ | -------- | ----------- | ----- |
| MMBench_DEV_[CN/EN]    | ✅     | ✅     |            | ✅        | ✅    | ✅                       | ✅      | ✅        | ✅           | ✅     |
| MMBench_TEST_[CN/EN]   | ✅     | ✅     |            | ✅        | ✅    | ✅                       |        | ✅        | ✅           | ✅     |
| CCBench                | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| SEEDBench_IMG          | ✅     | ✅     |            | ✅        |      | ✅                       | ✅      | ✅        |             |       |
| MME                    | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMVet                  | ✅     | ✅     |            | ✅        |      |                         | ✅      | ✅        |             |       |
| MMMU_DEV_VAL           | ✅     | ✅     | ✅          | ✅        |      | ✅                       | ✅      | ✅        | ✅           | ✅     |
| COCO_VAL               | ✅     | ✅     |            |          |      |                         | ✅      |          |             |       |
| OCRVQA_[TEST/TESTCORE] | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| TextVQA_VAL            | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |
| VCR_[EN/ZH]_[EASY/HARD]_[ALL/500/100]            | ✅     | ✅     |            | ✅        |      |                         | ✅      |          |             |       |

<div align="center"><b>Table 1. TSV fields for supported datasets.</b></div>

Required fields:

- **index:** an integer unique identifier for each row in TSV
- **image:** base64-encoded image; you can use helper APIs in `vlmeval/smp/vlm.py`:
   - Encode: `encode_image_to_base64` (PIL Image) / `encode_image_file_to_base64` (image file path)
   - Decode: `decode_base64_to_image` (PIL Image) / `decode_base64_to_image_file` (image file path)
- **question:** the question string
- **answer:** the answer string; may be missing for test-only splits

### 2. Build custom prompts for the dataset

`ImageBaseDataset` defines a default prompt format. If you need extra prompt fields or an interleaved input format, implement `build_prompt(line)`. The input is a row from the TSV containing fields like index/image/question, and the output is a list of dicts such as `[dict(type='image', value=IMAGE_PTH), dict(type='text', value=prompt)]`, containing the image path and the text prompt passed to VLMs. For interleaved inputs, place the image dict at the image token position.

### 3. Implement dataset metrics / evaluation

To support evaluation, define a dataset class that computes metrics. Image-text multimodal datasets inherit from `ImageBaseDataset` in `vlmeval/dataset/image_base.py`. `TYPE` defines dataset type; `DATASET_URL` is the download URL; `DATASET_MD5` is used to verify file integrity.

You **must implement** `evaluate(eval_file, **judge_kwargs)` to compute metrics and output results. Input `eval_file` is the model prediction file path `{model_name}_{dataset}.xlsx`. You can load it into a pandas DataFrame via `load(eval_file)`, which typically includes fields like index/question/answer/category/prediction. `judge_kwargs` provides evaluation-related parameters (judge model name, API threads, etc.). The return value is a set of metrics (often a dict of lists) and can be organized into a pandas DataFrame.

## Implementing a new model

Example PR: **Support LLaVA-Next-Interleave** ([#294](https://github.com/open-compass/VLMEvalKit/pull/294))

### 1. Support `generate_inner` API (required)

All models are implemented under `vlmeval/vlm`. For a minimal model, your model class **should implement** `generate_inner(msgs, dataset=None)`. This function sends a multimodal message to the VLM and returns the prediction (a string). Optional `dataset` can be used as a flag to switch strategies for different datasets.

The multimodal message `msgs` is a list of dicts. Each dict has two keys:

- `type`: supported types currently include `image` and `text`
- `value`: for `text`, a string; for `image`, a local image path or an image URL

Currently, a message can contain arbitrary interleavings of images and text. If your model does not support that, the recommended approach is to use the first image plus concatenated text. You can set `INTERLEAVE = False` in the class and call `self.message_to_promptimg(message, dataset=dataset)` to obtain the prompt and the first image path.

Examples:

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

For convenience, string lists are also supported. In that case, the framework checks whether a string is an image path or an image URL and normalizes it into `list[dict]`:

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [IMAGE_PTH, 'What is in this image?']
msg2 = [IMAGE_URL, IMAGE_URL,  'How many apples are there in these images?']
response = model.generate(msg1)
```

### 2. Support custom prompt construction (optional)

You can support custom prompting by implementing two optional methods: `use_custom_prompt(dataset)` and `build_prompt(line, dataset=None)`.

- `use_custom_prompt(dataset)` returns a boolean indicating whether the model should use its own prompt strategy.
- If `use_custom_prompt(dataset)` returns True, `build_prompt(line, dataset)` should return a custom multimodal message for the dataset row. If False, the default dataset prompt strategy is used.

### 3. Support multi-turn chat (optional)

To support multi-turn dialogue evaluation, implement `chat_inner(message, dataset)` and use the model’s `chat()` interface. This API returns a string response. The input `message` is a chat history list:

```python
# Assume msg1, msg2, msg3, ... are multi-modal messages following the previously described format
# `chat_inner` takes the following chat history list as input:
message = [
    dict(role='user', content=msg1),
    dict(role='assistant', content=msg2),
    dict(role='user', content=msg3),
    dict(role='assistant', content=msg4),
    ......
    dict(role='user', content=msgn),
]
# `message` should contain an odd number of utterances, roles should alternate between
# "user" and "assistant", and the last utterance must be "user".
# The chat function will call `chat_inner`.
response = model.chat(message)
```

Example PRs:

- A VLM that does not support interleaving and does not use custom prompts: [[Model] Support glm-4v-9b](https://github.com/open-compass/VLMEvalKit/pull/221)
- A VLM that supports interleaving and custom prompts: [Add MiniCPM-Llama3-V-2.5](https://github.com/open-compass/VLMEvalKit/pull/205)
- A VLM API: [Feature: add glmv](https://github.com/open-compass/VLMEvalKit/pull/201)

## Contributing code to VLMEvalKit

If you want to contribute to **VLMEvalKit**, run pre-commit checks before submitting PRs to keep the codebase clean:

```bash
# Under the VLMEvalKit directory, install pre-commit hooks:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# Then submit your PR.
```
