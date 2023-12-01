# MMCompass

![LOGO](assets/LOGO.svg)

**MMCompass** is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In MMCompass, we adopt **generation-based evaluation** for all LVLMs (obtain the answer via `generate` / `chat`  interface), and provide the evaluation results obtained with both **exact matching** and **LLM(ChatGPT)-based answer extraction**. 

**The codebase can:**

1. Provide an **easy-to-use**, **opensource evaluation toolkit** to make it convenient for researchers & developers to evaluate existing LVLMs and make evaluation results **easy to reproduce**.
2. Make it easy for VLM developers to evaluate their own models. To evaluate the VLM on multiple MMCompass supported benchmarks, one just need to **implement a single `generate` function**, all other workloads (data downloading, data preprocessing, prediction inference, metric calculation) are handled by MMCompass. 

**The codebase can not: **

1. Reproduce the exact accuracy number reported in the original papers of all **3rd party benchmarks**. The reason can be two-fold:
   1. MMCompass **mainly** uses **generation-based evaluation** for all VLMs (and optionally with **LLM-based answer extraction**). Meanwhile, some benchmarks may use different metrics (SEEDBench uses PPL-based evaluation, *eg.*). For these benchmarks, we will compare both scores in the corresponding README file. We encourage developers to support other different evaluation paradigms in MMCompass. 
   2. By default, we use the same prompt template for all VLMs to evaluate on a benchmark. Meanwhile, **some VLMs may have their specific prompt templates** (some may not covered by MMCompass at this time). We encourage VLM developers to implement their own prompt template in MMCompass, if that is not covered currently. That will help to improve the reproducibility. 

## Supported Datasets and Models

**Supported Dataset**

- [x] [MMBench Series](https://github.com/open-compass/mmbench/): MMBench, MMBench-CN, CCBench
- [x] [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
- [x] [SEEDBench_IMG](https://github.com/AILab-CVC/SEED-Bench)
- [ ] [MM-VET]()
- [ ] ......

**Supported Models**

- [x] [IDEFICS-9B-Instruct](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct), [IDEFICS-80B-Instruct](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct)
- [x] [InstructBLIP-[7B/13B]](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md)
- [x] [LLaVA-[v1-7B/v1.5-7B/v1.5-13B]](https://github.com/haotian-liu/LLaVA)
- [x] [MiniGPT-4-[v1-7B/v1-13B/v2-7B]](https://github.com/Vision-CAIR/MiniGPT-4)
- [x] [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)
- [x] [OpenFlamingo-v2](https://github.com/mlfoundations/open_flamingo)
- [x] [PandaGPT-13B](https://github.com/yxuansu/PandaGPT)
- [x] [Qwen-VL](https://huggingface.co/Qwen/Qwen-VL), [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat)
- [x] [VisualGLM-6B](https://huggingface.co/THUDM/visualglm-6b)
- [x] [InternLM-XComposer-7B](https://huggingface.co/internlm/internlm-xcomposer-7b)
- [ ] ......

## How to run the evaluation?

Basically, there are two stages for evaluating an VLM on a benchmark, namely **prediction inference** and **metric calculation**. Besides, before running the evaluation script, you need to **configure** the VLMs and set the model_paths properly. 

### Configuration

**VLM Configuration**: All VLMs are configured in `mmcompass/config.py`, for some VLMs, you need to configure the code root (MiniGPT-4, PandaGPT, etc.) or the model_weight root (LLaVA-v1-7B, etc.) before conducting the evaluation. During evaluation, you should use the model name specified in `supported_VLM` in `mmcompass/config.py` to select the VLM. For MiniGPT-4 and InstructBLIP, you also need to modify the config files in `mmcompass/vlm/misc` to configure LLM path and ckpt path.

### Prediction Inference

We use the script `mmcompass/infer/inference.py` for prediction inference. To use the script, you can go into the directory `mmcompass/infer/` or create a soft-link of the script (so that you can use the script anywhere):

**Arguments**

- For `--data`, you can only set the dataset names that are supported in MMCompass (defined in `mmcompass/utils/data_util.py`), including: MME, SEEDBench_IMG, MMBench_DEV_EN, MMBench_TEST_EN, MMBench_DEV_CN, MMBench_TEST_CN, CCBench
- For `--model`, you can only set the VLM names that are supported in MMCompass (defined in `supported_VLM` in `mmcompass/config.py`). 
- `--data` and `--model` receive list as arguments, you can select multiple benchmarks and VLMs for prediction inference at a time. 

**Command**

- You can run the script with `python` or `torchrun`:
  - When running with `python`, only one VLM instance is instantiated, and it might use multiple GPUs (depending on its default behavior). That is recommended for evaluating very large VLMs (like IDEFICS-80B-Instruct):

    ```bash
    # Inference IDEFICS-80B-Instruct on MMBench_DEV_EN, MME, and SEEDBench_IMG
    python inference.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct --verbose
    ```

  - When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference. However, that is only suitable for VLMs that consume small amounts of GPU memory. 

    ```bash
    # Inference IDEFICS-9B-Instruct, Qwen-VL-Chat, mPLUG-Owl2 on MMBench_DEV_EN, MME, and SEEDBench_IMG. On a node with 8 GPU. 
    torchrun --nproc-per-node=8 inference.py --data MMBench_DEV_EN MME SEEDBench_IMG --model idefics_80b_instruct qwen_chat mPLUG-Owl2 --verbose
    # Inference Qwen-VL-Chat on MME. On a node with 2 GPU. 
    torchrun --nproc-per-node=2 inference.py --data MME --model qwen_chat --verbose
    ```

**Generated Files**

- Two files will be generated with `inference.py`:
  - `{model_name}/{model_name}_{dataset_name}.xlsx`: The file that contain the VLM predictions. 
  - `{model_name}/{model_name}_{dataset_name}_prefetch.xlsx`: The file that contain the basic statistics of the VLM predictions. For example, the score / accuracy based on exact matching. 

### Metric Calculation

 **Multiple Choice Problems (MMBenchs, SEEDBench_IMG)** 

The following script use ChatGPT

- **Script**: `mmcompass/eval/multiple_choice.py` 
- **Usage**: `python multiple_choice.py {MMBench_or_SEEDBench_prediction_file.xlsx} --model {llm_for_choice_matching} --dataset {dataset_name} --nproc {num_process_for_API_calling}`

**Yes / No Problems (MME)**



## How to implement a new Benchmark / VLM in MMCompass? 





