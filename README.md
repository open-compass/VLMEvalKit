![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)
<div align="center"><b>A Toolkit for Evaluating Large Vision-Language Models. </b></div>

<div align="center">
English | [<a href="/docs/zh-CN/README_zh-CN.md">简体中文</a>] | [<a href="/docs/ja/README_ja.md">日本語</a>]
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

- **[2024-06-26]** We have supported the evaluation of [**CongRong**](https://mllm.cloudwalk.com/web), it ranked **3rd** on the [**Open VLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 🔥🔥🔥
- **[2024-06-26]** We firstly support a video understanding benchmark: [**MMBench-Video**](https://mmbench-video.github.io), Image LVLMs that accept multiple images as inputs can be evaluated on the video understanding benchmarks. Check [**QuickStart**](/docs/en/Quickstart.md) to learn how to perform the evaluation 🔥🔥🔥
- **[2024-06-24]** We have supported the evaluation of [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet), it ranked the **2nd** on the [**Open VLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 🔥🔥🔥
- **[2024-06-22]** Since GPT-3.5-Turbo-0613 is no longer supported yet, we switch to GPT-3.5-Turbo-0125 for choice extraction
- **[2024-06-18]** We have supported [**SEEDBench2**](https://arxiv.org/abs/2311.17092),  thanks to [**Bohao-Lee**](https://github.com/Bohao-Lee)🔥🔥🔥
- **[2024-06-18]** We have supported [**MMT-Bench**](https://mmt-bench.github.io), thanks to [**KainingYing**](https://github.com/KainingYing)🔥🔥🔥
- **[2024-06-12]** We have supported [**GLM-4v-9B**](https://huggingface.co/THUDM/glm-4v-9b)🔥🔥🔥
- **[2024-06-05]** We have supported [**WeMM**](https://github.com/scenarios/WeMM), thanks to [**scenarios**](https://github.com/scenarios)🔥🔥🔥
- **[2024-05-27]** We have supported [**Mini InternVL**](https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5), thanks to [**czczup**](https://github.com/czczup)🔥🔥🔥
- **[2024-05-25]** We have supported [**SEEDBench2_Plus**](https://arxiv.org/abs/2404.16790),  thanks to [**Bohao-Lee**](https://github.com/Bohao-Lee)🔥🔥🔥

## 📊 Datasets, Models, and Evaluation Results

**The performance numbers on our official multi-modal leaderboards can be downloaded from here!**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [Download All DETAILED Results](http://opencompass.openxlab.space/utils/OpenVLM.json).

**Supported Image Understanding Dataset**

- By default, all evaluation results are presented in [**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard).

| Dataset                                                      | Dataset Names (for run.py)                             | Task | Dataset | Dataset Names (for run.py) | Task |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | --------- | --------- | --------- |
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN]<br>MMBench\_DEV\_[EN/CN]\_V11<br>MMBench\_TEST\_[EN/CN]\_V11<br>CCBench | Multi-choice <br>Question (MCQ) | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | MCQ |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | Yes or No (Y/N)                                         | [**SEEDBench Series**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus | MCQ                                                |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | [**MMMU**](https://mmmu-benchmark.github.io)  | MMMU_[DEV_VAL/TEST]                      | MCQ                                |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | [**ScienceQA_IMG**](https://scienceqa.github.io) | ScienceQA_[VAL/TEST]                     | MCQ                        |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | Caption                                              | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                | Y/N                             |
| [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST] | VQA                                 | [**TextVQA**](https://textvqa.org)* | TextVQA_VAL                      | VQA                              |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST | VQA                                          | [**AI2D**](https://allenai.org/data/diagrams) | AI2D_TEST                                 | MCQ                         |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | [**DocVQA**](https://www.docvqa.org)+       | DocVQA_[VAL/TEST]                           | VQA                                         |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST] | VQA | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | MCQ                                          | [**POPE**](https://github.com/AoiDragon/POPE) | POPE                                           | Y/N                                            |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM | VQA                                               | [**MMT-Bench**](https://mmt-bench.github.io)                 | MMT-Bench_[VAL/VAL_MI/ALL/ALL_MI]                | MCQ  |
| [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS | VQA |  |  |  |

**\*** We only provide a subset of the evaluation results, since some VLMs do not yield reasonable results under the zero-shot setting

**\+** The evaluation results are not available yet

**\-** Only inference is supported in VLMEvalKit

VLMEvalKit will use an **judge LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**

**Supported Video Understanding Dataset**

| Dataset                                              | Dataset Names (for run.py) | Task | Dataset | Dataset Names (for run.py) | Task |
| ---------------------------------------------------- | -------------------------- | ---- | ------- | -------------------------- | ---- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video              | VQA  |         |                            |      |

**Supported API Models**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude3-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 |                                                              |                                                              |                                                              |                                                   |

**Supported PyTorch / HF Models**

| [**IDEFICS-[9B/80B/v2-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🎞️🚅 | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl2**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🎞️🚅, [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🎞️**🚅** |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-7B**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅🎞️ | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-[Chat/Llama3]**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**ShareCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅, [**Monkey-Chat**](https://github.com/Yuliang-Liu/Monkey)🚅 | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️         | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer2-[1.8B/7B]**](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b)🚅🎞️ | [**MiniCPM-[V1/V2/V2.5]**](https://huggingface.co/openbmb/MiniCPM-V)🚅 | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat-[V1-1/V1-2/V1-2-Plus/V1-5]**](https://github.com/OpenGVLab/InternVL)🚅, <br>[**Mini-InternVL-Chat-2B-V1-5**](https://github.com/OpenGVLab/InternVL)🚅 |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅 | [**Bunny-Llama3**](https://huggingface.co/BAAI/Bunny-Llama-3-8B-V)🚅 | [**XVERSE-V-13B**](https://github.com/xverse-ai/XVERSE-V-13B/blob/main/vxverse/models/vxverse.py) |
| [**PaliGemma-3B**](https://huggingface.co/google/paligemma-3b-pt-448) 🚅 | [**360VL-70B**](https://huggingface.co/qihoo360/360VL-70B) 🚅 | [**Phi-3-Vision**](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)🚅 | [**WeMM**](https://github.com/scenarios/WeMM)🚅               |
| [**GLM-4v-9B**](https://huggingface.co/THUDM/glm-4v-9b) 🚅    |                                                              |                                                              |                                                              |

🎞️: Support multiple images as inputs.

🚅: Models can be used without any additional configuration/operation.

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`, `GLM-4v-9B`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `LLaVA-Next series`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **Please use** `transformers==latest` **for**: `PaliGemma-3B`.

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

See [[QuickStart](/docs/en/Quickstart.md) | [快速开始](/docs/zh-CN/Quickstart_zh-CN.md)] for a quick start guide.

## 🛠️ Development Guide

To develop custom benchmarks, VLMs, or simply contribute other codes to **VLMEvalKit**, please refer to [[Development_Guide](/docs/en/Development.md) | [开发指南](/docs/zh-CN/Development_zh-CN.md)].

## 🎯 The Goal of VLMEvalKit

**The codebase is designed to:**

1. Provide an **easy-to-use**, **opensource evaluation toolkit** to make it convenient for researchers & developers to evaluate existing LVLMs and make evaluation results **easy to reproduce**.
2. Make it easy for VLM developers to evaluate their own models. To evaluate the VLM on multiple supported benchmarks, one just need to **implement a single `generate_inner()` function**, all other workloads (data downloading, data preprocessing, prediction inference, metric calculation) are handled by the codebase.

**The codebase is not designed to:**

1. Reproduce the exact accuracy number reported in the original papers of all **3rd party benchmarks**. The reason can be two-fold:
   1. VLMEvalKit uses **generation-based evaluation** for all VLMs (and optionally with **LLM-based answer extraction**). Meanwhile, some benchmarks may use different approaches (SEEDBench uses PPL-based evaluation, *eg.*). For those benchmarks, we compare both scores in the corresponding result. We encourage developers to support other evaluation paradigms in the codebase.
   2. By default, we use the same prompt template for all VLMs to evaluate on a benchmark. Meanwhile, **some VLMs may have their specific prompt templates** (some may not covered by the codebase at this time). We encourage VLM developers to implement their own prompt template in VLMEvalKit, if that is not covered currently. That will help to improve the reproducibility.

## 🖊️ Citation

If you find this work helpful, please consider to **star🌟** this repo. Thanks for your support!

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

If you use VLMEvalKit in your research or wish to refer to published OpenSource evaluation results, please use the following BibTeX entry and the BibTex entry corresponding to the specific VLM / benchmark you used.

```bib
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

## 💻 Other Projects in OpenCompass

- [Opencompass](https://github.com/open-compass/opencompass/): An LLM evaluation platform, supporting a wide range of models (LLaMA, LLaMa2, ChatGLM2, ChatGPT, Claude, etc) over 50+ datasets.
- [MMBench](https://github.com/open-compass/MMBench/): Official Repo of "MMBench: Is Your Multi-modal Model an All-around Player?"
- [BotChat](https://github.com/open-compass/BotChat/): Evaluating LLMs' multi-round chatting capability.
- [LawBench](https://github.com/open-compass/LawBench): Benchmarking Legal Knowledge of Large Language Models.
- [Ada-LEval](https://github.com/open-compass/Ada-LEval): Length Adaptive Evaluation, measures the long-context modeling capability of language models.
