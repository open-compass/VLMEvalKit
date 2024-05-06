![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)
<div align="center"><b>A Toolkit for Evaluating Large Vision-Language Models. </b></div>

<div align="center">
English | [<a href="README_zh-CN.md">简体中文</a>]
</div>

<div align="center">
<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 Learderboard </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>
</div>

<div align="center">
<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 Leaderboard</a>
<a href="https://openxlab.org.cn/apps/detail/kennyutc/open_mllm_leaderboard">(🤖 OpenXlab Mirror)</a>
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord Channel</a>
</div>


**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

## 🆕 News

- **[2024-05-06]** We have launched a discord channel for VLMEvalKit users: https://discord.gg/evDT4GZmxN. Latest updates and discussion will be posted here
- **[2024-05-06]** We have supported 2 VLMs based on Llama3 🔥🔥🔥: Bunny-llama3-8B (SigLIP, image size 384) and llava-llama-3-8b (CLIP-L, image size 336), you can now evaluate both models on dozens of datasets we supported
- **[2024-04-28]** We have supported [**MMBench V1.1**](https://arxiv.org/pdf/2307.06281), the new version  has better data quality and improved vision indispensability. To obtain the test-set performance, please submit the inference result to the [**evaluation service**](https://mmbench.opencompass.org.cn/mmbench-submission) 🔥🔥🔥
- **[2024-04-28]** We have supported [**POPE**](https://github.com/AoiDragon/POPE), a benchmark for object hallucination evaluation 🔥🔥🔥
- **[2024-04-25]** We have supported [**Reka API**](https://www.reka.ai), the API model ranked first in [**Vision-Arena**](https://huggingface.co/spaces/WildVision/vision-arena) 🔥🔥🔥
- **[2024-04-21]** We have notices a minor issue of the MathVista evaluation script (which may negatively affect the performance). We have fixed it and updated the leaderboard accordingly
- **[2024-04-17]** We have supported [**InternVL-Chat-V1.5**](https://github.com/OpenGVLab/InternVL/) 🔥🔥🔥
- **[2024-04-15]** We have supported [**RealWorldQA**](https://x.ai/blog/grok-1.5v), a multimodal benchmark for real-world spatial understanding 🔥🔥🔥
- **[2024-04-09]** We have refactored the inference interface of VLMs to a more unified version, check [**#140**](https://github.com/open-compass/VLMEvalKit/pull/140) for more details
- **[2024-04-09]** We have supported [**MMStar**](https://github.com/MMStar-Benchmark/MMStar), a challenging vision-indispensable multimodal benchmark. The full evaluation results will be released soon 🔥🔥🔥


## 📊 Datasets, Models, and Evaluation Results

**The performance numbers on our official multi-modal leaderboards can be downloaded from here!**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [Download All DETAILED Results](http://opencompass.openxlab.space/utils/OpenVLM.json).

**Supported Dataset**

- By default, all evaluation results are presented in [**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard).

| Dataset                                                      | Dataset Names (for run.py)                             | Task | Dataset | Dataset Names (for run.py) | Task |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | --------- | --------- | --------- |
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench_DEV_[EN/CN]<br>MMBench_TEST_[EN/CN]<br>MMBench_DEV_[EN/CN]_V11<br>MMBench_TEST_[EN/CN]_V11<br>CCBench | Multi-choice <br>Question (MCQ) | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | MCQ |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | Yes or No (Y/N)                                         | [**SEEDBench_IMG**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG                              | MCQ                                                |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | [**MMMU**](https://mmmu-benchmark.github.io)  | MMMU_DEV_VAL/MMMU_TEST                        | MCQ                                |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | [**ScienceQA_IMG**](https://scienceqa.github.io) | ScienceQA_[VAL/TEST]                     | MCQ                        |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | Caption                                              | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                | Y/N                             |
| [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST] | VQA                                 | [**TextVQA**](https://textvqa.org)* | TextVQA_VAL                      | VQA                              |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST | VQA                                          | [**AI2D**](https://allenai.org/data/diagrams) | AI2D_TEST                                 | MCQ                         |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | [**DocVQA**](https://www.docvqa.org)+       | DocVQA_[VAL/TEST]                           | VQA                                         |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST] | VQA | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | MCQ                                          | [**POPE**](https://github.com/AoiDragon/POPE)+ | POPE                                           | Y/N                                            |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM | VQA                                               |                                                |                                                |                                                |

**\*** We only provide a subset of the evaluation results, since some VLMs do not yield reasonable results under the zero-shot setting

**\+** The evaluation results are not available yet

**\-** Only inference is supported in VLMEvalKit

VLMEvalKit will use an **judge LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**

**Supported API Models**

| [**GPT-4V (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision)🎞️🚅 | [**GeminiProVision**](https://platform.openai.com/docs/guides/vision)🎞️🚅 | [**QwenVLPlus**](https://huggingface.co/spaces/Qwen/Qwen-VL-Plus)🎞️🚅 | [**QwenVLMax**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max)🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v)🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| [**Reka**](https://www.reka.ai)🚅                             |                                                              |                                                              |                                                              |                                                  |

**Supported PyTorch / HF Models**

| [**IDEFICS-[9B/80B/v2-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🎞️🚅 | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl2**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🎞️🚅, [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🎞️**🚅** |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-7B**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅🎞️ | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-17B-Chat**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**SharedCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅         | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️  | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer2-[1.8B/7B]**](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b)🚅🎞️ | [**MiniCPM-[V1/V2]**](https://huggingface.co/openbmb/MiniCPM-V)🚅   | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat Series**](https://github.com/OpenGVLab/InternVL)🚅 |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅 | [**Bunny-Llama3**](https://huggingface.co/BAAI/Bunny-Llama-3-8B-V)🚅  |                                                              |

🎞️: Support multiple images as inputs.

🚅: Model can be used without any additional configuration / operation.

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `SharedCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-V series`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`.
- **Please use** `transformers==4.39.0` **for**: `LLaVA-Next series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`.

```python
# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# Forward Single Image
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # The image features a red apple with a leaf on it.
# Forward Multiple Images
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # There are two apples in the provided images.
```

## 🏗️ QuickStart

See [QuickStart](/Quickstart.md) for a quick start guide.

## 🛠️ Development Guide

To develop custom benchmarks, VLMs, or simply contribute other codes to **VLMEvalKit**, please refer to [Development_Guide](/Development.md).

## 🎯 The Goal of VLMEvalKit

**The codebase is designed to:**

1. Provide an **easy-to-use**, **opensource evaluation toolkit** to make it convenient for researchers & developers to evaluate existing LVLMs and make evaluation results **easy to reproduce**.
2. Make it easy for VLM developers to evaluate their own models. To evaluate the VLM on multiple supported benchmarks, one just need to **implement a single `generate_inner()` function**, all other workloads (data downloading, data preprocessing, prediction inference, metric calculation) are handled by the codebase.

**The codebase is not designed to:**

1. Reproduce the exact accuracy number reported in the original papers of all **3rd party benchmarks**. The reason can be two-fold:
   1. VLMEvalKit uses **generation-based evaluation** for all VLMs (and optionally with **LLM-based answer extraction**). Meanwhile, some benchmarks may use different approaches (SEEDBench uses PPL-based evaluation, *eg.*). For those benchmarks, we compare both scores in the corresponding result. We encourage developers to support other evaluation paradigms in the codebase.
   2. By default, we use the same prompt template for all VLMs to evaluate on a benchmark. Meanwhile, **some VLMs may have their specific prompt templates** (some may not covered by the codebase at this time). We encourage VLM developers to implement their own prompt template in VLMEvalKit, if that is not covered currently. That will help to improve the reproducibility.

## 🖊️ Citation

If you use VLMEvalKit in your research or wish to refer to the published OpenSource evaluation results, please use the following BibTeX entry and the BibTex entry corresponding to the specific VLM / benchmark you used.

```bib
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
- [Ada-LEval](https://github.com/open-compass/Ada-LEval): Length Adaptive Evaluation, measures the long-context modeling capability of language models.
