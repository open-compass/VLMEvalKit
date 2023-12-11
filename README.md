# VLMEvalKit

![LOGO](assets/LOGO.svg)

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs (obtain the answer via `generate` / `chat`  interface), and provide the evaluation results obtained with both **exact matching** and **LLM(ChatGPT)-based answer extraction**. 

**The codebase can:**

1. Provide an **easy-to-use**, **opensource evaluation toolkit** to make it convenient for researchers & developers to evaluate existing LVLMs and make evaluation results **easy to reproduce**.
2. Make it easy for VLM developers to evaluate their own models. To evaluate the VLM on multiple supported benchmarks, one just need to **implement a single `generate` function**, all other workloads (data downloading, data preprocessing, prediction inference, metric calculation) are handled by the codebase. 

**The codebase can not:**

1. Reproduce the exact accuracy number reported in the original papers of all **3rd party benchmarks**. The reason can be two-fold:
   1. VLMEvalKit **mainly** uses **generation-based evaluation** for all VLMs (and optionally with **LLM-based answer extraction**). Meanwhile, some benchmarks may use different metrics (SEEDBench uses PPL-based evaluation, *eg.*). For these benchmarks, we will compare both scores in the corresponding README file. We encourage developers to support other different evaluation paradigms in the codebase. 
   2. By default, we use the same prompt template for all VLMs to evaluate on a benchmark. Meanwhile, **some VLMs may have their specific prompt templates** (some may not covered by the codebase at this time). We encourage VLM developers to implement their own prompt template in VLMEvalKit, if that is not covered currently. That will help to improve the reproducibility. 

## 📊 Datasets, Models, and Evaluation Results

**Supported Dataset**

| Dataset                                                      | Inference | Evaluation | Results                                                      |
| ------------------------------------------------------------ | --------- | ---------- | ------------------------------------------------------------ |
| [**MMBench Series**](https://github.com/open-compass/mmbench/): MMBench, MMBench-CN, CCBench | √         | √          | [**MMBench Series**](https://mmbench.opencompass.org.cn/leaderboard) |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | √         | √          | [**MME**](results/MME.md)                                    |
| [**SEEDBench_IMG**](https://github.com/AILab-CVC/SEED-Bench) | √         | √          | [**SEEDBench_IMG**](results/SEEDBench_IMG.md)                |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | √         | √          | [**MM-Vet**](results/MMVet.md)                               |
| [**Core-MM**](https://github.com/core-mm/core-mm)            | √         |            |                                                              |

**Supported Models**

| [**IDEFICS-9B-Instruct**🎞️🚅](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct), [**IDEFICS-80B-Instruct**🎞️🚅](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B🚅/v1.5-13B🚅]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) | [**mPLUG-Owl2**🎞️](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**🎞️🚅](https://huggingface.co/Qwen/Qwen-VL), [**Qwen-VL-Chat**🎞️🚅](https://huggingface.co/Qwen/Qwen-VL-Chat) | [**VisualGLM-6B**🚅](https://huggingface.co/THUDM/visualglm-6b) | [**InternLM-XComposer-7B**🎞️🚅](https://huggingface.co/internlm/internlm-xcomposer-7b) |
| [**ShareGPT4V-7B**🚅](https://sharegpt4v.github.io)           | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |                                                              |                                                              |                                                              |

🎞️: Support multiple images as inputs, via the `multi_generate` interface. 

🚅: Model can be used without any additional configuration / operation. 

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate('assets/apple.jpg', 'What is in this image?')
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.multi_generate(['assets/apple.jpg', 'assets/apple.jpg'], 'How many apples are there in the provided images? ')
print(ret)  # There are two apples in the provided images.
```

## 🏗️ How to run the evaluation?

Basically, there are two stages for evaluating an VLM on a benchmark, namely **prediction inference** and **metric calculation**. 

Besides, before running the evaluation script, you need to **configure** the VLMs and set the model_paths properly. 

### Step1. Configuration

**VLM Configuration**: All VLMs are configured in `vlmeval/config.py`, for some VLMs, you need to configure the code root (MiniGPT-4, PandaGPT, etc.) or the model_weight root (LLaVA-v1-7B, etc.) before conducting the evaluation. During evaluation, you should use the model name specified in `supported_VLM` in `vlmeval/config.py` to select the VLM. For MiniGPT-4 and InstructBLIP, you also need to modify the config files in `vlmeval/vlm/misc` to configure LLM path and ckpt path.

### Step2. Prediction Inference

We use `vlmeval/infer/inference.py` for prediction inference. To use the script, you can go into the directory `vlmeval/infer/` or create a soft-link of the script (to use the script anywhere):

**Arguments**

- For `--data`, you can only set the dataset names that are supported in VLMEvalKit (defined in `vlmeval/utils/data_util.py`). 
  - including: `MME, SEEDBench_IMG, MMBench_DEV_EN, MMBench_TEST_EN, MMBench_DEV_CN, MMBench_TEST_CN, CCBench, Core_MM, MMVet`

- For `--model`, you can only set the VLM names that are supported in VLMEvalKit (defined in `supported_VLM` in `vlmeval/config.py`). 
- `--data` and `--model` receive list as arguments, you can select multiple benchmarks and VLMs for prediction inference at a time. 

**Command**

You can run the script with `python` or `torchrun`:

```bash
# When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior). 
# That is recommended for evaluating very large VLMs (like IDEFICS-80B-Instruct).

# Inference IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG
python inference.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose

# When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference. 
# However, that is only suitable for VLMs that consume small amounts of GPU memory. 

# Inference IDEFICS-9B-Instruct, Qwen-VL-Chat, mPLUG-Owl2 on MMBench_DEV_EN, MME, and SEEDBench_IMG. On a node with 8 GPU. 
torchrun --nproc-per-node=8 inference.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
# Inference Qwen-VL-Chat on MME. On a node with 2 GPU. 
torchrun --nproc-per-node=2 inference.py --data MME --model qwen_chat --verbose
```

**Generated Files**

Two files will be generated with `inference.py`:
- `{model_name}/{model_name}_{dataset_name}.xlsx`: The file that contain the VLM predictions. 
- `{model_name}/{model_name}_{dataset_name}_prefetch.xlsx`: The file that contain the basic statistics of the VLM predictions. For example, the score / accuracy based on exact matching. 

### Step3. Metric Calculation

**Multiple Choice Problems (MMBench Series, SEEDBench_IMG)** 

> The following script use ChatGPT / Exact Matching for answer matching and calculate the accuracy (**CircularEval** for MMBench, **VanillaEval** for SEEDBench_IMG).
>
> ```bash
> python vlmeval/eval/multiple_choice.py {MMBench_or_SEEDBench_prediction_file.xlsx} --model {llm_for_choice_matching} --dataset {dataset_name} --nproc {num_process_for_API_calling}
> ```
>
> - `--model`: supported choices are `gpt-3.5-turbo-0613 (default)` and `exact_matching`. 

**Yes / No Problems (MME)**

> The following script use ChatGPT for answer matching and calculate the MME score (The score of Exact matching is already calculated in `*_prefetch.xlsx`).
>
> ```bash
> python vlmeval/eval/mme_eval.py {MME_prediction_file.xlsx} --nproc {num_process_for_API_calling}
> ```

**GPT-based Scoring for Open-ended Problems (MMVet)**

> The following script use GPT-4-Turbo for scoring and calculate MMVet accuracies.
>
> ```bash
> python vlmeval/eval/mmvet_eval.py {MMVet_prediction_file.xlsx} --model {llm_for_scoring} --nproc {num_process_for_API_calling}
> ```
>
> `--model`: supported choices are `gpt-4-turbo (default), gpt-4-0613, chatgpt-1106, chatgpt-0613`. 

## 🛠️ How to implement a new Benchmark / VLM in VLMEvalKit? 

### Implement a new benchmark

Currently, we organize a benchmark as one single TSV file. During inference, the data file will be automatically downloaded to `$LMUData` (default path is `$HOME/LMUData`, if not set explicitly). All existing benchmark TSV files are handled by `TSVDataset` implemented in `vlmeval/utils/data_util.py`. 

| Dataset Name \ Fields | index | image | image_path | question | hint | A    | B    | C    | D    | answer | category | l2-category | split |
| --------------------- | ----- | ----- | ---------- | -------- | ---- | ---- | ---- | ---- | ---- | ------ | -------- | ----------- | ----- |
| MMBench_DEV_CN/EN     | √     | √     |            | √        | √    | √    | √    | √    | √    | √      | √        | √           | √     |
| MMBench_TEST_CN/EN    | √     | √     |            | √        | √    | √    | √    | √    | √    |        | √        | √           | √     |
| CCBench               | √     | √     |            | √        |      | √    | √    | √    | √    | √      | √        |             |       |
| SEEDBench_IMG         | √     | √     |            | √        |      | √    | √    | √    | √    | √      | √        |             |       |
| MME                   | √     | √     |            | √        |      |      |      |      |      | √      | √        |             |       |
| CORE_MM               | √     | √     | √          | √        |      |      |      |      |      |        | √        |             |       |
| MMVet                 | √     | √     |            | √        |      |      |      |      |      | √      | √        |             |       |

<div align="center"><b>Table 1. TSV fields of supported datasets.</b></div>

**Intro to some fields:**

- **index:** Integer, Unique for each line in `tsv`
- **image:** the base64 of the image, you can use APIs implemented in `vlmeval/smp.py` for encoding and decoding: 
  - Encoding: `encode_image_to_base64 `(for PIL Image) / `encode_image_file_to_base64` (for image file path)
  - Decoding: `decode_base64_to_image`(for PIL Image) / `decode_base64_to_image_file` (for image file path)

Besides, your dataset class **should implement the method `build_prompt(self, line, dataset=None)`**. Given line as a line number or one line in the TSV file, the function yields a dictionary `dict(image=image_path, text=prompt)`, including the image path and the prompt that will be fed to the VLMs.

### Implement a new model

All existing models are implemented in `vlmeval/vlm`. For a minimal model, your model class **should implement the method** `generate(image_path, prompt, dataset=None)`. In this function, you feed the image and prompt to your VLM and return the VLM prediction (which is a string). The optional argument `dataset` can be used as the flag for the model to switch among various inference strategies. 

Besides, your model can support custom prompt building by implementing an optional method `build_prompt(line, dataset=None)`. In this function, the line is a dictionary that includes the necessary information of a data sample, while `dataset` can be used as the flag for the model to switch among various prompt building strategies. 


## 🖊️ Citation

If you use VLMEvalKit in your research or wish to refer to the published OpenSource evaluation results, please use the following BibTeX entry and the BibTex entry corresponding to the specific VLM / benchmark you used.

```bib
@article{MMBench,
    author  = {Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, Dahua Lin},
    journal = {arXiv:2307.06281},
    title   = {MMBench: Is Your Multi-modal Model an All-around Player?},
    year    = {2023},
}

@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

## 💻 Other Projects in OpenCompass

- [opencompass](https://github.com/open-compass/opencompass/): An LLM evaluation platform, supporting a wide range of models (LLaMA, LLaMa2, ChatGLM2, ChatGPT, Claude, etc) over 50+ datasets.
- [MMBench](https://github.com/open-compass/MMBench/): Official Repo of "MMBench: Is Your Multi-modal Model an All-around Player?"
- [BotChat](https://github.com/open-compass/BotChat/): Evaluating LLMs' multi-round chatting capability.
- [LawBench](https://github.com/open-compass/LawBench): Benchmarking Legal Knowledge of Large Language Models. 
