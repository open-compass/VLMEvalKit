# 🛠️ How to implement a new Benchmark / VLM in VLMEvalKit?

## Implement a new benchmark

Example PR: **Math-Vision Benchmark** ([#292](https://github.com/open-compass/VLMEvalKit/pull/292/files))

### 1. Prepare your benchmark tsv file

Currently, we organize a benchmark as one single TSV file. During inference, the data file will be automatically downloaded to `$LMUData` (default path is `$HOME/LMUData`, if not set explicitly). You can also customize it in the environment variable `LMUData=/path/to/your/data`.

The contents of the TSV file consist of:

| Dataset Name \ Fields  | index | image | image_path | question | hint | multi-choice<br>options | answer | category | l2-category | split |
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

<div align="center"><b>Table 1. TSV fields of supported datasets.</b></div>

**Intro to some fields:**

- **index:** Integer, Unique for each line in `tsv`
- **image:** the base64 of the image, you can use APIs implemented in `vlmeval/smp.py` for encoding and decoding:
  - Encoding: `encode_image_to_base64 `(for PIL Image) / `encode_image_file_to_base64` (for image file path)
  - Decoding: `decode_base64_to_image`(for PIL Image) / `decode_base64_to_image_file` (for image file path)

### 2. Cutomize your benchmark metrics

To add evaluation for a new benchmark, you need to customize a class object for the dataset in `vlmeval/dataset/utils` to implement the dataset’s metrics calculation. Multimodal datasets inherit from the `ImageBaseDataset` object in `vlmeval/dataset/image_base.py`. The TYPE defines the type of dataset, `DATASET_URL` is the download address of the dataset, and `DATASET_MD5` is the MD5 checksum for consistency checking of the dataset file.

In this class, you need to implement the `evaluate(eval_file, **judge_kwargs)` class function to calculate metrics and output results for the custom dataset. If you need to add prompts for the dataset, you can do so by implementing the `build_prompt(line)` function. Given a line from the TSV file as line, this function generates a dictionary `dict(image=image_path, text=prompt)` representing a multimodal message `msg`, including the image path and the text prompt to be input into the VLMs.

## Implement a new model

Example PR: **Support LLaVA-Next-Interleave** ([#294](https://github.com/open-compass/VLMEvalKit/pull/294))

All existing models are implemented in `vlmeval/vlm`. For a minimal model, your model class **should implement the method** `generate_inner(msgs, dataset=None)`. In this function, you feed a multi-modal message to your VLM and return the VLM prediction (which is a string). The optional argument `dataset` can be used as the flag for the model to switch among various inference strategies.

The multi-modal messages `msgs` is a list of dictionaries, each dictionary has two keys: type and value:
- `type`: We currently support two types, choices are ["image", "text"].
- `value`: When type=='text' , the value is the text message (a single string); when type=='image', the value can be the local path of an image file, or the image URL.

Currently a multi-modal message may contain arbitarily interleaved images and texts. If your model do not support that, our recommended practice is to take the first image and concatenated text messages as the input to the model.

Here are some examples of multi-modal messages:

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

For convenience sake, we also support to take a list of string as inputs. In that case, we will check if a string is an image path or image URL and automatically convert it to the list[dict] format:

```python
IMAGE_PTH = 'assets/apple.jpg'
IMAGE_URL = 'https://raw.githubusercontent.com/open-compass/VLMEvalKit/main/assets/apple.jpg'
msg1 = [IMAGE_PTH, 'What is in this image?']
msg2 = [IMAGE_URL, IMAGE_URL,  'How many apples are there in these images?']
response = model.generate(msg1)
```

Besides, your model can support custom prompt building by implementing two optional methods: `use_custom_prompt(dataset)` and `build_prompt(line, dataset=None)`. Both functions take the dataset name as the input. `use_custom_prompt` will return a boolean flag, indicating whether the model should use the custom prompt building strategy. If it returns True, `build_prompt` should return a customly bulit multimodal message for the corresponding `dataset`, given `line`, which is a dictionary that includes the necessary information of a data sample. If it returns False, the default prompt building strategy will be used.

### Example PRs:

- VLM that doesn't support interleaved images and texts, and does not use custom prompts: [[Model] Support glm-4v-9b](https://github.com/open-compass/VLMEvalKit/pull/221)
- VLM that supports interleaved images and texts and custom prompts: [Add MiniCPM-Llama3-V-2.5](https://github.com/open-compass/VLMEvalKit/pull/205)
- VLM API: [Feature add glmv](https://github.com/open-compass/VLMEvalKit/pull/201)

## Contribute to VLMEvalKit

If you want to contribute codes to **VLMEvalKit**, please do the pre-commit check before you submit a PR. That helps to keep the code tidy.

```bash
# Under the directory of VLMEvalKit, install the pre-commit hook:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# Then you can commit your code.
```
