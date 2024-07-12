![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)
<div align="center"><b> VLMEvalKit—多模态大模型评测工具 </b></div>

<div align="center">
[<a href="/README.md">English</a>] | 简体中文 | [<a href="/docs/ja/README_ja.md">日本語</a>]
</div>

<div align="center">
<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 性能榜单 </a> •
<a href="#data-model-results">📊 数据集和模型 </a> •
<a href="#quickstart">🏗️ 快速开始 </a> •
<a href="#development">🛠️ 开发 </a> •
<a href="#goal-of-vlmevalkit">🎯 我们的目标 </a> •
<a href="#citation">🖊️ 引用 </a>
</div>

<div align="center">
<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 排行榜 </a>
<a href="https://openxlab.org.cn/apps/detail/kennyutc/open_mllm_leaderboard">(🤖 OpenXlab 镜像)</a>
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord 频道</a>
</div>
**VLMEvalKit** (python 包名为 **vlmeval**) 是一款专为大型视觉语言模型 (Large Vision-Language Models， LVLMs) 评测而设计的开源工具包。该工具支持在各种基准测试上对大型视觉语言模型进行**一键评估**，无需进行繁重的数据准备工作，让评估过程更加简便。在 VLMEvalKit 中，我们对所有大型视觉语言模型生成的结果进行评测，并提供基于**精确匹配**与基于 **LLM 的答案提取**两种评测结果。

## 🆕 更新

- **[2024-07-12]** 支持了多模态长文档内容理解基准 [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/), 感谢 [**mayubo2333**](https://github.com/mayubo2333) 🔥🔥🔥
- **[2024-07-12]** 支持了 [**VCR**](https://github.com/tianyu-z/vcr), 一个视觉 caption 修复测试基准, 感谢 [**tianyu-z**](https://github.com/tianyu-z) and [**sheryc**](https://github.com/sheryc) 🔥🔥🔥
- **[2024-07-08]** 支持了 [**InternLM-XComposer-2.5**](https://github.com/InternLM/InternLM-XComposer), 感谢 [**LightDXY**](https://github.com/LightDXY) 🔥🔥🔥
- **[2024-07-08]** 支持了 [**InternVL2**](https://huggingface.co/OpenGVLab/InternVL2-26B), 感谢 [**czczup**](https://github.com/czczup) 🔥🔥🔥
- **[2024-06-27]** 支持了 [**Cambrian**](https://cambrian-mllm.github.io/) 🔥🔥🔥
- **[2024-06-27]** 支持了 [**AesBench**](https://github.com/yipoh/AesBench)，感谢 [**Yipo Huang**](https://github.com/yipoh) 与 [**Quan Yuan**](https://github.com/dylanqyuan)🔥🔥🔥
- **[2024-06-26]** 支持了 [**CongRong**](https://mllm.cloudwalk.com/web) 的评测，该模型在 [**Open VLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 上**排名第三** 🔥🔥🔥
- **[2024-06-26]** 首次支持了视频理解评测基准 [**MMBench-Video**](https://mmbench-video.github.io)，可以用于测试支持多图输入的图文多模态大模型的。[**快速开始**](/docs/zh-CN/Quickstart.md) 中提供了启动 MMBench-Video 测试的方式 🔥🔥🔥
- **[2024-06-24]** 支持了 [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet) 的评测，该模型在 [**Open VLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 上**排名第二** 🔥🔥🔥
- **[2024-06-22]** 由于 GPT-3.5-Turbo-0613 已被 OpenAI 废弃，我们改为使用 GPT-3.5-Turbo-0125 辅助答案提取

## 📊 评测结果，支持的数据集和模型 <a id="data-model-results"></a>
### 评测结果

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [下载全部细粒度测试结果](http://opencompass.openxlab.space/assets/OpenVLM.json)。

### 支持的图文多模态评测集

- 默认情况下，我们在 [**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 提供全部测试结果

| 数据集                                                      | 数据集名称 (用于 run.py)                  | 任务类型 | 数据集                                                       | 数据集名称 (用于 run.py) | 任务类型 |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | --------- | --------- | --------- |
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN] <br>MMBench\_DEV\_[EN/CN]\_V11 <br>MMBench\_TEST\_[EN/CN]\_V11 <br>CCBench | Multi-choice <br>Question (MCQ) | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | MCQ |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | Yes or No (Y/N)                                         | [**SEEDBench Series**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus | MCQ      |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | [**MMMU**](https://mmmu-benchmark.github.io)  | MMMU_DEV_VAL/MMMU_TEST                        | MCQ                                |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | [**ScienceQA_IMG**](https://scienceqa.github.io) | ScienceQA_[VAL/TEST]                     | MCQ                        |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | Caption                                              | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                | Y/N                             |
| [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST] | VQA                                 | [**TextVQA**](https://textvqa.org)* | TextVQA_VAL                      | VQA                              |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST | VQA                                          | [**AI2D**](https://allenai.org/data/diagrams) | AI2D_TEST                                 | MCQ                         |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | [**DocVQA**](https://www.docvqa.org)+       | DocVQA_[VAL/TEST]                           | VQA                                         |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST] | VQA | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | MCQ                                          | [**POPE**](https://github.com/AoiDragon/POPE) | POPE                                           | Y/N                                            |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM | VQA                                               | [**MMT-Bench**](https://mmt-bench.github.io)                 | MMT-Bench_[VAL/VAL_MI/ALL/ALL_MI]                | MCQ      |
| [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS | VQA | [**AesBench**](https://github.com/yipoh/AesBench) | AesBench_[VAL/TEST] | MCQ |
| [**VCR-wiki**](https://huggingface.co/datasets/vcr-org/)+ | VCR_[EN/ZH]_[EASY/HARD]_[ALL/500/100] | VCR | [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/)+ | MMLongBench_DOC | VQA |

**\*** 我们只提供了部分模型上的测试结果，剩余模型无法在 zero-shot 设定下测试出合理的精度

**\+** 我们尚未提供这个评测集的测试结果

**\-** VLMEvalKit 仅支持这个评测集的推理，无法输出最终精度

如果您设置了 API KEY，VLMEvalKit 将使用一个 **LLM** 从输出中提取答案进行匹配判断，否则它将使用**精确匹配**模式 (直接在输出字符串中查找“yes”，“no”，“A”，“B”，“C”等)。**精确匹配只能应用于是或否任务和多选择任务**

### 支持的视频多模态评测集

| Dataset                                              | Dataset Names (for run.py) | Task | Dataset | Dataset Names (for run.py) | Task |
| ---------------------------------------------------- | -------------------------- | ---- | ------- | -------------------------- | ---- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video              | VQA  |         |                            |      |

### 支持的模型
**API 模型**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude3-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 |                                                              |                                                              |                                                              |                                                   |

**基于 PyTorch / HF 的开源模型**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude3-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 |                                                              |                                                              |                                                              |                                                   |

**Supported PyTorch / HF Models**

| [**IDEFICS-[9B/80B/v2-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🎞️🚅 | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl2**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🎞️🚅, [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🎞️**🚅** |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-[1/2]**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅 | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-[Chat/Llama3]**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**ShareCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅, [**Monkey-Chat**](https://github.com/Yuliang-Liu/Monkey)🚅 | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️         | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer-2.5**](https://github.com/InternLM/InternLM-XComposer)🚅🎞️ | [**MiniCPM-[V1/V2/V2.5]**](https://huggingface.co/openbmb/MiniCPM-V)🚅 | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat-[V1-1/V1-2/V1-2-Plus/V1-5/V2]**](https://github.com/OpenGVLab/InternVL)🚅🎞️, <br>[**Mini-InternVL-Chat-[2B/4B]-V1-5**](https://github.com/OpenGVLab/InternVL)🚅🎞️ |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅 | [**Bunny-Llama3**](https://huggingface.co/BAAI/Bunny-Llama-3-8B-V)🚅 | [**XVERSE-V-13B**](https://github.com/xverse-ai/XVERSE-V-13B/blob/main/vxverse/models/vxverse.py) |
| [**PaliGemma-3B**](https://huggingface.co/google/paligemma-3b-pt-448) 🚅 | [**360VL-70B**](https://huggingface.co/qihoo360/360VL-70B) 🚅 | [**Phi-3-Vision**](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)🚅 | [**WeMM**](https://github.com/scenarios/WeMM)🚅               |
| [**GLM-4v-9B**](https://huggingface.co/THUDM/glm-4v-9b) 🚅    |  [**Cambrian-[8B/13B/34B]**](https://cambrian-mllm.github.io/) |                                                              |                                                              |

🎞️ 表示支持多图片输入。

🚅 表示模型可以被直接使用，不需任何额外的配置。

### 其他

**Transformers 的版本推荐:**

**请注意**，某些 VLM 可能无法在某些特定的 transformers 版本下运行，我们建议使用以下设置来评估对应的VLM:

- **请用** `transformers==4.33.0` **来运行**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `SharedCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`, `GLM-4v-9B`.
- **请用** `transformers==4.37.0 ` **来运行**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-V (v1, v2)`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`.
- **请用** `transformers==4.40.0 ` **来运行**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `LLaVA-Next series`, `360VL-70B`， `Phi-3-Vision`，`WeMM`.
- **请用** `transformers==latest` **来运行**: `PaliGemma-3B`.

**如何测试一个 VLM 是否可以正常运行:**

```python
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# 前向单张图片
ret = model.generate(['assets/apple.jpg', 'What is in this image?'])
print(ret)  # 这张图片上有一个带叶子的红苹果
# 前向多张图片
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', 'How many apples are there in the provided images? '])
print(ret)  # 提供的图片中有两个苹果
```

## 🏗️ 快速开始 <a id="quickstart"></a>

请参阅[**快速开始**](/docs/zh-CN/Quickstart_zh-CN.md)获取入门指南。

## 🛠️ 开发指南 <a id="development"></a>

要开发自定义评测数据集，支持其他 VLMs，或为 VLMEvalKit 贡献代码，请参阅[**开发指南**](/docs/zh-CN/Development_zh-CN.md)。

## 🎯 VLMEvalKit 的目标 <a id="goal-of-vlmevalkit"></a>

**该代码库的设计目标是：**

1. 提供一个**易于使用**的**开源评估工具包**，方便研究人员和开发人员评测现有的多模态大模型，并使评测结果**易于复现**。
2. 使 VLM 开发人员能够轻松地评测自己的模型。在多个支持的基准测试上评估 VLM，只需实现一个 `generate_inner()` 函数，所有其他工作负载（数据下载、数据预处理、预测推理、度量计算）都由代码库处理。

**该代码库的设计目标不是:**

复现所有**第三方基准测试**原始论文中报告的准确数字。有两个相关的原因:
1. VLMEvalKit 对所有 VLMs 使用基于生成的评估（可选使用基于 LLM 的答案提取）。同时，一些基准测试可能官方使用不同的方法（*例如，SEEDBench 使用基于 PPL 的评估*）。对于这些基准测试，我们在相应的结果中比较两个得分。我们鼓励开发人员在代码库中支持其他评估范式。
2. 默认情况下，我们对所有多模态模型使用相同的提示模板来评估基准测试。同时，**一些多模态模型可能有他们特定的提示模板**（目前可能未在代码库中涵盖）。我们鼓励 VLM 的开发人员在 VLMEvalKit 中实现自己的提示模板，如果目前未覆盖。这将有助于提高可复现性。

## 🖊️ 引用 <a id="citation"></a>

如果我们的工作对您有所帮助，请考虑 **star🌟** VLMEvalKit。感谢支持！

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

如果您在研究中使用了 VLMEvalKit，或希望参考已发布的开源评估结果，请使用以下 BibTeX 条目以及与您使用的特定 VLM / 基准测试相对应的 BibTex 条目。

```bib
@misc{2023opencompass,
    title={OpenCompass: A Universal Evaluation Platform for Foundation Models},
    author={OpenCompass Contributors},
    howpublished = {\url{https://github.com/open-compass/opencompass}},
    year={2023}
}
```

## 💻 OpenCompass 的其他项目

- [Opencompass](https://github.com/open-compass/opencompass/): 一个大模型评测平台，支持广泛的模型 (LLaMA, LLaMa2, ChatGLM2, ChatGPT, Claude等) 覆盖 50 多个数据集。
- [MMBench](https://github.com/open-compass/MMBench/): 官方代码库 "MMBench: Is Your Multi-modal Model an All-around Player?"
- [BotChat](https://github.com/open-compass/BotChat/): 评测大模型多轮对话能力。
- [LawBench](https://github.com/open-compass/LawBench): 对大语言模型的法律知识进行测试。
- [Ada-LEval](https://github.com/open-compass/ada-leval): 对大语言模型的长文本建模能力进行测试。
