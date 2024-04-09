# 🛠️ How to implement a new Benchmark / VLM in VLMEvalKit?

## Implement a new benchmark

Example PR: **Add OCRBench** ([#91](https://github.com/open-compass/VLMEvalKit/pull/91/files))

Currently, we organize a benchmark as one single TSV file. During inference, the data file will be automatically downloaded to `$LMUData` (default path is `$HOME/LMUData`, if not set explicitly). All existing benchmark TSV files are handled by `TSVDataset` implemented in `vlmeval/utils/dataset_config.py`.

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

<div align="center"><b>Table 1. TSV fields of supported datasets.</b></div>

**Intro to some fields:**

- **index:** Integer, Unique for each line in `tsv`
- **image:** the base64 of the image, you can use APIs implemented in `vlmeval/smp.py` for encoding and decoding:
  - Encoding: `encode_image_to_base64 `(for PIL Image) / `encode_image_file_to_base64` (for image file path)
  - Decoding: `decode_base64_to_image`(for PIL Image) / `decode_base64_to_image_file` (for image file path)

Besides, your dataset class **should implement the method `build_prompt(self, line, dataset=None)`**. Given line as a line number or one line in the TSV file, the function yields a dictionary `dict(image=image_path, text=prompt)`, including the image path and the prompt that will be fed to the VLMs.

## Implement a new model

Example PR: **Support Monkey** ([#45](https://github.com/open-compass/VLMEvalKit/pull/45/files))

All existing models are implemented in `vlmeval/vlm`. For a minimal model, your model class **should implement the method** `generate(image_path, prompt, dataset=None)`. In this function, you feed the image and prompt to your VLM and return the VLM prediction (which is a string). The optional argument `dataset` can be used as the flag for the model to switch among various inference strategies.

Besides, your model can support custom prompt building by implementing an optional method `build_prompt(line, dataset=None)`. In this function, the line is a dictionary that includes the necessary information of a data sample, while `dataset` can be used as the flag for the model to switch among various prompt building strategies.

## Contribute to VLMEvalKit

If you want to contribute codes to **VLMEvalKit**, please do the pre-commit check before you submit a PR. That helps to keep the code tidy.

```bash
# Under the directory of VLMEvalKit, install the pre-commit hook:
pip install pre-commit
pre-commit install
pre-commit run --all-files
# Then you can commit your code.
```
