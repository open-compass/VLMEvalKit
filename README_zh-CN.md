![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)
<div align="center"><b> VLMEvalKit—多模态大模型评测工具 </b></div>

<div align="center">
[<a href="README.md">English</a>] | 简体中文
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
<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HuggingFace排行榜 </a>
<a href="https://openxlab.org.cn/apps/detail/kennyutc/open_mllm_leaderboard">🤖 OpenXlab排行榜 </a>
</div>
**VLMEvalKit** (python 包名为 **vlmeval**) 是一款专为大型视觉语言模型 (Large Vision-Language Models， LVLMs) 评测而设计的开源工具包。该工具支持在各种基准测试上对大型视觉语言模型进行**一键评估**，无需进行繁重的数据准备工作，让评估过程更加简便。在 VLMEvalKit 中，我们对所有大型视觉语言模型生成的结果进行评测，并提供基于**精确匹配**与基于 **LLM 的答案提取**两种评测结果。

## 🆕 更新

- **[2024-04-28]** 支持了 [**MMBench V1.1**](https://arxiv.org/pdf/2307.06281)，在这个新版本中，我们提升了评测题目的质量与视觉的不可或缺性 🔥🔥🔥
- **[2024-04-28]** 支持 [**POPE**](https://github.com/AoiDragon/POPE), 这是一个目标幻觉问题检测的数据集 🔥🔥🔥
- **[2024-04-25]** 支持了 [**Reka**](https://www.reka.ai), 这个 API 模型在 [**Vision-Arena**](https://huggingface.co/spaces/WildVision/vision-arena) 排名第一 🔥🔥🔥
- **[2024-04-21]** 修复了 MathVista 评估脚本的一个小问题（可能会对性能产生较小的负面影响），并相应更新了排行榜
- **[2024-04-17]** 支持 [**InternVL-Chat-V1.5**](https://github.com/OpenGVLab/InternVL/) 🔥🔥🔥
- **[2024-04-15]** 支持 [**RealWorldQA**](https://x.ai/blog/grok-1.5v)， 这是一个用于真实世界空间理解的多模态基准测试  🔥🔥🔥
- **[2024-04-09]** 将 VLMs 推理接口重构为更统一的版本，请查看 [**#140**](https://github.com/open-compass/VLMEvalKit/pull/140) 获取更多详细信息。
- **[2024-04-09]** 支持 [**MMStar**](https://github.com/MMStar-Benchmark/MMStar)，这是一个具有挑战性的视觉不可或缺的多模态基准测试 🔥🔥🔥
- **[2024-04-08]** 支持 [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa) 和 [**DocVQA**](https://www.docvqa.org) 的测试集，特别感谢 [**DLight**](https://github.com/LightDXY) 🔥🔥🔥
- **[2024-03-28]** 现在您可以使用本地的开源LLMs作为答案提取器或判断器 (请参阅 [**#132**](https://github.com/open-compass/VLMEvalKit/pull/132) 获取详细信息)，特别感谢 [**StarCycle**](https://github.com/StarCycle) 🔥🔥🔥

## 📊 评测结果，支持的数据集和模型 <a id="data-model-results"></a>
### 评测结果

[**OpenCompass 多模态排行榜**](https://rank.opencompass.org.cn/leaderboard-multimodal): [下载全部细粒度测试结果](http://opencompass.openxlab.space/utils/OpenVLM.json)。

### 支持的数据集

| 数据集                                                      | 数据集名称 (对应run.py文件)                             | 任务类型 | 推理 | 评测 | 结果                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | ---------- | ------------------------------------------------------------ |-----|
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench_DEV_[EN/CN]<br/>MMBench_TEST_[EN/CN]<br/>MMBench_DEV_[EN/CN]_V11<br/>MMBench_TEST_[EN/CN]_V11<br/>CCBench | Multi-choice | ✅         | ✅          | [**MMBench Leaderboard**](https://mmbench.opencompass.org.cn/leaderboard) |
| [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | Multi-choice   | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | Yes or No                                                   | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**SEEDBench_IMG**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG | Multi-choice                                         | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**MMMU**](https://mmmu-benchmark.github.io)                 | MMMU_DEV_VAL/MMMU_TEST | Multi-choice                                | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**ScienceQA_IMG**](https://scienceqa.github.io)             | ScienceQA_[VAL/TEST] | Multi-choice                                   | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | Caption                                              | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench | Yes or No                                         | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**OCRVQA**](https://ocr-vqa.github.io)                      | OCRVQA_[TESTCORE/TEST] | VQA                                 | ✅         | ✅          | **TBD.**                                                     |
| [**TextVQA**](https://textvqa.org)                           | TextVQA_VAL | VQA                                           | ✅         | ✅          | **TBD.**                                                     |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)            | ChartQA_TEST | VQA                                          | ✅         | ✅          | **TBD.**                                                     |
| [**AI2D**](https://allenai.org/data/diagrams)                | AI2D_TEST | Multi-choice                                             | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**DocVQA**](https://www.docvqa.org)                         | DocVQA_[VAL/TEST] | VQA                                            | ✅         | ✅          | **TBD.**                                                     |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa) | InfoVQA_[VAL/TEST] | VQA | ✅ | ✅ | **TBD.** |
| [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA                                              | ✅         | ✅          | [**Open_VLM_Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| [**Core-MM**](https://github.com/core-mm/core-mm)            | CORE_MM | VQA                                               | ✅         |            | **N/A**                                                      |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | VQA                                               | ✅         | ✅           | **TBD.**                                                      |
| [**POPE**](https://github.com/AoiDragon/POPE)            | POPE | Yes or No                                               | ✅         | ✅           | **TBD.**                                                      |

如果您设置了 API KEY，VLMEvalKit 将使用一个 **LLM** 从输出中提取答案进行匹配判断，否则它将使用**精确匹配**模式 (直接在输出字符串中查找“yes”，“no”，“A”，“B”，“C”等)。**精确匹配只能应用于是或否任务和多选择任务**

**OCRVQA, TextVQA, ChartQA 等 VQA 任务存在一些已知的问题，我们将尽快修正。**

### 支持的模型
**API 模型**

| [**GPT-4V (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision)🎞️🚅 | [**GeminiProVision**](https://platform.openai.com/docs/guides/vision)🎞️🚅 | [**QwenVLPlus**](https://huggingface.co/spaces/Qwen/Qwen-VL-Plus)🎞️🚅 | [**QwenVLMax**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max)🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v)🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------ |
| [**Reka**](https://www.reka.ai)🚅                             |                                                              |                                                              |                                                              |                                                  |

**基于 PyTorch / HF 的开源模型**

| [**IDEFICS-[9B/80B/v2-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🎞️🚅 | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl2**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🎞️🚅, [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🎞️**🚅** |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-7B**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅🎞️ | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-17B-Chat**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**SharedCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅         | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️  | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer2-[1.8B/7B]**](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b)🚅🎞️ | [**MiniCPM-[V1/V2]**](https://huggingface.co/openbmb/MiniCPM-V)🚅   | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat Series**](https://github.com/OpenGVLab/InternVL)🚅 |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅 |   |                                                              |

🎞️ 表示支持多图片输入。

🚅 表示模型可以被直接使用，不需任何额外的配置。

### 其他

**Transformers 的版本推荐:**

**请注意**，某些 VLM 可能无法在某些特定的 transformers 版本下运行，我们建议使用以下设置来评估对应的VLM:

- **请用** `transformers==4.33.0` **来运行**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `SharedCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`.
- **请用** `transformers==4.37.0 ` **来运行**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-V series`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`.
- **请用** `transformers==4.39.0 ` **来运行**: `LLaVA-Next series`.
- **请用** `transformers==4.40.0 ` **来运行**: `IDEFICS2`.

**如何测试一个 VLM 是否可以正常运行**

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

请参阅[**快速开始**](/Quickstart.md)获取入门指南。

## 🛠️ 开发指南 <a id="development"></a>

要开发自定义评测数据集，支持其他 VLMs，或为 VLMEvalKit 贡献代码，请参阅[**开发指南**](/Development.md)。

## 🎯 VLMEvalKit 的目标 <a id="goal-of-vlmevalkit"></a>

**该代码库的设计目标是：**

1. 提供一个**易于使用**的**开源评估工具包**，方便研究人员和开发人员评测现有的多模态大模型，并使评测结果**易于复现**。
2. 使 VLM 开发人员能够轻松地评测自己的模型。在多个支持的基准测试上评估 VLM，只需实现一个 `generate_inner()` 函数，所有其他工作负载（数据下载、数据预处理、预测推理、度量计算）都由代码库处理。

**该代码库的设计目标不是:**

复现所有**第三方基准测试**原始论文中报告的准确数字。有两个相关的原因:
1. VLMEvalKit 对所有 VLMs 使用基于生成的评估（可选使用基于 LLM 的答案提取）。同时，一些基准测试可能官方使用不同的方法（*例如，SEEDBench 使用基于 PPL 的评估*）。对于这些基准测试，我们在相应的结果中比较两个得分。我们鼓励开发人员在代码库中支持其他评估范式。
2. 默认情况下，我们对所有多模态模型使用相同的提示模板来评估基准测试。同时，**一些多模态模型可能有他们特定的提示模板**（目前可能未在代码库中涵盖）。我们鼓励 VLM 的开发人员在 VLMEvalKit 中实现自己的提示模板，如果目前未覆盖。这将有助于提高可复现性。

## 🖊️ 引用 <a id="citation"></a>

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

- [opencompass](https://github.com/open-compass/opencompass/): 一个大模型评测平台，支持广泛的模型 (LLaMA, LLaMa2, ChatGLM2, ChatGPT, Claude等) 覆盖 50 多个数据集。
- [MMBench](https://github.com/open-compass/MMBench/): 官方代码库 "MMBench: Is Your Multi-modal Model an All-around Player?"
- [BotChat](https://github.com/open-compass/BotChat/): 评测大模型多轮对话能力。
- [LawBench](https://github.com/open-compass/LawBench): 对大语言模型的法律知识进行测试。
- [Ada-LEval](https://github.com/open-compass/ada-leval): 对大语言模型的长文本建模能力进行测试。
