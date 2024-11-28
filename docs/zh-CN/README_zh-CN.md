<div align="center">

![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>VLMEvalKit: 一种多模态大模型评测工具 </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

[English](/README.md) | 简体中文 | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OpenCompass 排行榜 </a> •
<a href="#%EF%B8%8F-quickstart">🏗️ 快速开始 </a> •
<a href="#-datasets-models-and-evaluation-results">📊 数据集和模型 </a> •
<a href="#%EF%B8%8F-development-guide">🛠️ 开发指南 </a> •
<a href="#-the-goal-of-vlmevalkit">🎯 我们的目标 </a> •
<a href="#%EF%B8%8F-citation">🖊️ 引用 </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HuggingFace 排行榜 (存档全部性能) </a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 原始评测记录</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 技术报告 </a>
</div>

**VLMEvalKit** (python 包名为 **vlmeval**) 是一款专为大型视觉语言模型 (Large Vision-Language Models， LVLMs) 评测而设计的开源工具包。该工具支持在各种基准测试上对大型视觉语言模型进行**一键评估**，无需进行繁重的数据准备工作，让评估过程更加简便。在 VLMEvalKit 中，我们对所有大型视觉语言模型生成的结果进行评测，并提供基于**精确匹配**与基于 **LLM 的答案提取**两种评测结果。

## 🆕 更新

- **[2024-11-21]** 集成了一个新的配置系统，以实现更灵活的评估设置。查看[文档](/docs/zh-CN/ConfigSystem.md)或运行`python run.py --help`了解更多详情 🔥🔥🔥
- **[2024-11-21]** 支持 **[QSpatial](https://andrewliao11.github.io/spatial_prompt/)**，一个用于定量空间推理的多模态基准（例如，确定大小/距离），感谢 **[andrewliao11](https://github.com/andrewliao11)** 提供官方支持 🔥🔥🔥
- **[2024-11-21]** 支持 **[MM-Math](https://github.com/kge-sun/mm-math)**，一个包含约6K初中多模态推理数学问题的新多模态数学基准。GPT-4o-20240806在该基准上达到了22.5%的准确率 🔥🔥🔥
- **[2024-11-16]** 支持 **[OlympiadBench](https://github.com/OpenBMB/OlympiadBench)**，一个多模态基准，包含奥林匹克级别的数学和物理问题 🔥🔥🔥
- **[2024-11-16]** 支持 **[WildVision](https://huggingface.co/datasets/WildVision/wildvision-bench)**，一个基于多模态竞技场数据的主观多模态基准 🔥🔥🔥
- **[2024-11-13]** 支持 **[MIA-Bench](https://arxiv.org/abs/2407.01509)**，一个多模态指令跟随基准 🔥🔥🔥
- **[2024-11-08]** 支持 **[Aria](https://arxiv.org/abs/2410.05993)**，一个多模态原生 MoE 模型，感谢 **[teowu](https://github.com/teowu)** 🔥🔥🔥
- **[2024-11-04]** 支持 **[WorldMedQA-V](https://www.arxiv.org/abs/2410.12722)**，该基准包含 1000 多个医学 VQA 问题，涵盖巴西、以色列、日本、西班牙等四个国家的语言，以及它们的英文翻译 🔥🔥🔥
- **[2024-11-01]** 支持 `AUTO_SPLIT` 标志 (https://github.com/open-compass/VLMEvalKit/pull/566)，用于在低配置 GPU 上进行评估。设置后，模型将自动拆分到多个 GPU（流水线并行）以减少 GPU 内存使用（目前仅支持部分 VLMs：Qwen2-VL、Llama-3.2、LLaVA-OneVision 等） 🔥🔥🔥
- **[2024-10-30]** 支持评估 **[MLVU](https://github.com/JUNJIE99/MLVU)** 和 **[TempCompass](https://arxiv.org/abs/2403.00476v1)**。这两个基准将很快被纳入 **[OpenVLM 视频排行榜](https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard)** 🔥🔥🔥

## 🏗️ 快速开始 <a id="quickstart"></a>

请参阅[**快速开始**](/docs/zh-CN/Quickstart.md)获取入门指南。

## 📊 评测结果，支持的数据集和模型 <a id="data-model-results"></a>

### 评测结果

**[OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)**: **[下载全部细粒度测试结果](http://opencompass.openxlab.space/assets/OpenVLM.json)**.

### 支持的图文多模态评测集

- 默认情况下，我们在 [**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 提供全部测试结果
- 使用的缩写：`MCQ`: 单项选择题; `Y/N`: 正误判断题; `MTT`: 多轮对话评测; `MTI`: 多图输入评测
-
- | Dataset                                                      | Dataset Names (for run.py)                                   | Task      | Dataset                                                      | Dataset Names (for run.py)                                   | Task           |
  | ------------------------------------------------------------ | ------------------------------------------------------------ | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------- |
  | [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN]<br>MMBench\_DEV\_[EN/CN]\_V11<br>MMBench\_TEST\_[EN/CN]\_V11<br>CCBench | MCQ       | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar)     | MMStar                                                       | MCQ            |
  | [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME                                                          | Y/N       | [**SEEDBench Series**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus             | MCQ            |
  | [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet                                                        | VQA       | [**MMMU**](https://mmmu-benchmark.github.io)                 | MMMU_[DEV_VAL/TEST]                                          | MCQ            |
  | [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI                                               | VQA       | [**ScienceQA_IMG**](https://scienceqa.github.io)             | ScienceQA_[VAL/TEST]                                         | MCQ            |
  | [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL                                                     | Caption   | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                               | Y/N            |
  | [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST]                                       | VQA       | [**TextVQA**](https://textvqa.org)*                          | TextVQA_VAL                                                  | VQA            |
  | [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST                                                 | VQA       | [**AI2D**](https://allenai.org/data/diagrams)                | AI2D_[TEST/TEST_NO_MASK]                                     | MCQ            |
  | [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench                                                   | VQA       | [**DocVQA**](https://www.docvqa.org)+                        | DocVQA_[VAL/TEST]                                            | VQA            |
  | [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST]                                           | VQA       | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench                                                     | VQA            |
  | [**RealWorldQA**](https://x.ai/blog/grok-1.5v)               | RealWorldQA                                                  | MCQ       | [**POPE**](https://github.com/AoiDragon/POPE)                | POPE                                                         | Y/N            |
  | [**Core-MM**](https://github.com/core-mm/core-mm)-           | CORE_MM (MTI)                                                | VQA       | [**MMT-Bench**](https://mmt-bench.github.io)                 | MMT-Bench\_[VAL/ALL]<br>MMT-Bench\_[VAL/ALL]_MI              | MCQ (MTI)      |
  | [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS                                                 | VQA       | [**AesBench**](https://github.com/yipoh/AesBench)+           | AesBench_[VAL/TEST]                                          | MCQ            |
  | [**VCR-wiki**](https://huggingface.co/vcr-org/) +            | VCR\_[EN/ZH]\_[EASY/HARD]_[ALL/500/100]                      | VQA       | [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/)+ | MMLongBench_DOC                                              | VQA (MTI)      |
  | [**BLINK**](https://zeyofu.github.io/blink/)                 | BLINK                                                        | MCQ (MTI) | [**MathVision**](https://mathvision-cuhk.github.io)+         | MathVision<br>MathVision_MINI                                | VQA            |
  | [**MT-VQA**](https://github.com/bytedance/MTVQA)             | MTVQA_TEST                                                   | VQA       | [**MMDU**](https://liuziyu77.github.io/MMDU/)+               | MMDU                                                         | VQA (MTT, MTI) |
  | [**Q-Bench1**](https://github.com/Q-Future/Q-Bench)          | Q-Bench1_[VAL/TEST]                                          | MCQ       | [**A-Bench**](https://github.com/Q-Future/A-Bench)           | A-Bench_[VAL/TEST]                                           | MCQ            |
  | [**DUDE**](https://arxiv.org/abs/2305.08455)+                | DUDE                                                         | VQA (MTI) | [**SlideVQA**](https://arxiv.org/abs/2301.04883)+            | SLIDEVQA<br>SLIDEVQA_MINI                                    | VQA (MTI)      |
  | [**TaskMeAnything ImageQA Random**](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random)+ | TaskMeAnything_v1_imageqa_random                             | MCQ       | [**MMMB and Multilingual MMBench**](https://sun-hailong.github.io/projects/Parrot/)+ | MMMB\_[ar/cn/en/pt/ru/tr]<br>MMBench_dev\_[ar/cn/en/pt/ru/tr]<br>MMMB<br>MTL_MMBench_DEV<br>PS: MMMB & MTL_MMBench_DEV <br>are **all-in-one** names for 6 langs | MCQ            |
  | [**A-OKVQA**](https://arxiv.org/abs/2206.01718)+             | A-OKVQA                                                      | MCQ       | [**MuirBench**](https://muirbench.github.io)+                | MUIRBench                                                    | MCQ            |
  | [**GMAI-MMBench**](https://huggingface.co/papers/2408.03361)+ | GMAI-MMBench_VAL                                             | MCQ       | [**TableVQABench**](https://arxiv.org/abs/2404.19205)+       | TableVQABench                                                | VQA            |
  | [**MME-RealWorld**](https://arxiv.org/abs/2408.13257)+       | MME-RealWorld[-CN]                                           | MCQ       | [**HRBench**](https://arxiv.org/abs/2408.15556)+             | HRBench[4K/8K]                                               | MCQ            |
  | [**MathVerse**](https://mathverse-cuhk.github.io/)+          | MathVerse_MINI<br/>MathVerse_MINI_Vision_Only <br/>MathVerse_MINI_Vision_Dominant<br/>MathVerse_MINI_Vision_Intensive<br/>MathVerse_MINI_Text_Lite<br/>MathVerse_MINI_Text_Dominant | VQA       | [**AMBER**](https://github.com/junyangwang0410/AMBER)+       | AMBER                                                        | Y/N            |
  | [**CRPE**](https://huggingface.co/datasets/OpenGVLab/CRPE)+  | CRPE_[EXIST/RELATION]                                        | VQA       | **[MMSearch](https://mmsearch.github.io/)**$$^1$$            | -                                                            | **-**          |
  | **[R-Bench](https://arxiv.org/abs/2410.05474)**+             | R-Bench-[Dis/Ref]                                            | MCQ       | **[WorldMedQA-V](https://www.arxiv.org/abs/2410.12722)**+    | WorldMedQA-V                                                 | MCQ            |
  | **[GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)**+ | GQA_TestDev_Balanced | VQA | **[MIA-Bench](https://arxiv.org/abs/2407.01509)**+ | MIA-Bench | VQA |
  | **[WildVision](https://huggingface.co/datasets/WildVision/wildvision-bench)**+ | WildVision | VQA | **[OlympiadBench](https://github.com/OpenBMB/OlympiadBench)** | OlympiadBench | VQA |

**\*** 我们只提供了部分模型上的测试结果，剩余模型无法在 zero-shot 设定下测试出合理的精度

**\+** 我们尚未提供这个评测集的测试结果

**\-** VLMEvalKit 仅支持这个评测集的推理，无法输出最终精度

$$^1$$ VLMEvalKit 在评测集的官方代码库中被使用

如果您设置了 API KEY，VLMEvalKit 将使用一个 **LLM** 从输出中提取答案进行匹配判断，否则它将使用**精确匹配**模式 (直接在输出字符串中查找“yes”，“no”，“A”，“B”，“C”等)。**精确匹配只能应用于是或否任务和多选择任务**

### 支持的视频多模态评测集

| Dataset                                                      | Dataset Names (for run.py) | Task                | Dataset                                       | Dataset Names (for run.py) | Task      |
| ------------------------------------------------------------ | -------------------------- | ------------------- | --------------------------------------------- | -------------------------- | --------- |
| [**MMBench-Video**](https://mmbench-video.github.io)         | MMBench-Video              | VQA                 | [**Video-MME**](https://video-mme.github.io/) | Video-MME                  | MCQ       |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | MVBench/MVBench_MP4        | MCQ                 | **[MLVU](https://github.com/JUNJIE99/MLVU)**  | MLVU                       | MCQ & VQA |
| **[TempCompass](https://arxiv.org/abs/2403.00476)**          | TempCompass                | MCQ & Y/N & Caption |                                               |                            |           |

### 支持的模型

**API 模型**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅<br>[**Qwen-VL-[Plus / Max]-0809**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude3-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet (20240620, 20241022)**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 | [**GPT-4o-Mini**](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 🎞️🚅 | [**Yi-Vision**](https://platform.lingyiwanwu.com)🎞️🚅          | [**Hunyuan-Vision**](https://cloud.tencent.com/document/product/1729)🎞️🚅 | [**BlueLM-V**](https://developers.vivo.com/) 🎞️🚅   |

**基于 PyTorch / HF 的开源模型**

| [**IDEFICS-[9B/80B/v2-8B/v3-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🚅🎞️ | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl[2/3]**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🚅🎞️ <br>[**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🚅🎞️ |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-[1/2]**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅 | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-[Chat/Llama3]**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**ShareCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅<br>[**Monkey-Chat**](https://github.com/Yuliang-Liu/Monkey)🚅 | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️         | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer-2.5**](https://github.com/InternLM/InternLM-XComposer)🚅🎞️ | [**MiniCPM-[V1/V2/V2.5/V2.6]**](https://github.com/OpenBMB/MiniCPM-V)🚅🎞️ | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat-[V1-1/V1-2/V1-5/V2]**](https://github.com/OpenGVLab/InternVL)🚅🎞️ |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅🎞️ | [**Bunny-Llama3**](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)🚅 | [**XVERSE-V-13B**](https://github.com/xverse-ai/XVERSE-V-13B/blob/main/vxverse/models/vxverse.py) |
| [**PaliGemma-3B**](https://huggingface.co/google/paligemma-3b-pt-448) 🚅 | [**360VL-70B**](https://huggingface.co/qihoo360/360VL-70B) 🚅 | [**Phi-3-Vision**](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)🚅🎞️<br>[**Phi-3.5-Vision**](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)🚅🎞️ | [**WeMM**](https://github.com/scenarios/WeMM)🚅               |
| [**GLM-4v-9B**](https://huggingface.co/THUDM/glm-4v-9b) 🚅    | [**Cambrian-[8B/13B/34B]**](https://cambrian-mllm.github.io/) | [**LLaVA-Next-[Qwen-32B]**](https://huggingface.co/lmms-lab/llava-next-qwen-32b) 🎞️ | [**Chameleon-[7B/30B]**](https://huggingface.co/facebook/chameleon-7b)🚅🎞️ |
| [**Video-LLaVA-7B-[HF]**](https://github.com/PKU-YuanGroup/Video-LLaVA) 🎬 | [**VILA1.5-[3B/8B/13B/40B]**](https://github.com/NVlabs/VILA/)🎞️ | [**Ovis[1.5-Llama3-8B/1.5-Gemma2-9B/1.6-Gemma2-9B/1.6-Llama3.2-3B]**](https://github.com/AIDC-AI/Ovis) 🚅🎞️ | [**Mantis-8B-[siglip-llama3/clip-llama3/Idefics2/Fuyu]**](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) 🎞️ |
| [**Llama-3-MixSenseV1_1**](https://huggingface.co/Zero-Vision/Llama-3-MixSenseV1_1)🚅 | [**Parrot-7B**](https://github.com/AIDC-AI/Parrot) 🚅         | [**OmChat-v2.0-13B-sinlge-beta**](https://huggingface.co/omlab/omchat-v2.0-13B-single-beta_hf)  🚅 | [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT) 🎬 |
| [**Chat-UniVi-7B[-v1.5]**](https://github.com/PKU-YuanGroup/Chat-UniVi) 🎬 | [**LLaMA-VID-7B**](https://github.com/dvlab-research/LLaMA-VID) 🎬 | [**VideoChat2-HD**](https://huggingface.co/OpenGVLab/VideoChat2_HD_stage4_Mistral_7B) 🎬 | [**PLLaVA-[7B/13B/34B]**](https://huggingface.co/ermu2001/pllava-7b) 🎬 |
| [**RBDash_72b**](https://github.com/RBDash-Team/RBDash) 🚅🎞️   | [**xgen-mm-phi3-[interleave/dpo]-r-v1.5**](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5) 🚅🎞️ | [**Qwen2-VL-[2B/7B/72B]**](https://github.com/QwenLM/Qwen2-VL)🚅🎞️ | [**slime_[7b/8b/13b]**](https://github.com/yfzhang114/SliME)🎞️ |
| [**Eagle-X4-[8B/13B]**](https://github.com/NVlabs/EAGLE)🚅🎞️, <br>[**Eagle-X5-[7B/13B/34B]**](https://github.com/NVlabs/EAGLE)🚅🎞️ | [**Moondream1**](https://github.com/vikhyat/moondream)🚅, <br>[**Moondream2**](https://github.com/vikhyat/moondream)🚅 | [**XinYuan-VL-2B-Instruct**](https://huggingface.co/Cylingo/Xinyuan-VL-2B)🚅🎞️ | [**Llama-3.2-[11B/90B]-Vision-Instruct**](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)🚅 |
| [**Kosmos2**](https://huggingface.co/microsoft/kosmos-2-patch14-224)🚅 | [**H2OVL-Mississippi-[0.8B/2B]**](https://huggingface.co/h2oai/h2ovl-mississippi-2b)🚅🎞️ | **[Pixtral-12B](https://huggingface.co/mistralai/Pixtral-12B-2409)**🎞️ | **[Falcon2-VLM-11B](https://huggingface.co/tiiuae/falcon-11B-vlm)**🚅 |
| **[MiniMonkey](https://huggingface.co/mx262/MiniMonkey)**🚅🎞️  | **[LLaVA-OneVision](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-sft)**🚅🎞️ | **[LLaVA-Video](https://huggingface.co/collections/lmms-lab/llava-video-661e86f5e8dabc3ff793c944)**🚅🎞️ | **[Aquila-VL-2B](https://huggingface.co/BAAI/Aquila-VL-2B-llava-qwen)**🚅🎞️ |
| [**Mini-InternVL-Chat-[2B/4B]-V1-5**](https://github.com/OpenGVLab/InternVL)🚅🎞️ | **[InternVL2 Series](https://huggingface.co/OpenGVLab/InternVL2-8B)** 🚅🎞️ | **[Janus-1.3B](https://huggingface.co/deepseek-ai/Janus-1.3B)**🚅🎞️ | **[molmoE-1B/molmo-7B/molmo-72B](https://huggingface.co/allenai/Molmo-7B-D-0924)**🚅 |
| **[Points-[Yi-1.5-9B/Qwen-2.5-7B]](https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat)**🚅 | **[NVLM](https://huggingface.co/nvidia/NVLM-D-72B)**🚅        | **[VIntern](https://huggingface.co/5CD-AI/Vintern-3B-beta)**🚅🎞️ | **[Aria](https://huggingface.co/rhymes-ai/Aria)**🚅🎞️ |

🎞️ 表示支持多图片输入。

🚅 表示模型可以被直接使用，不需任何额外的配置。

🎬 表示支持视频输入。

### 其他

**Transformers 的版本推荐:**

**请注意**，某些 VLM 可能无法在某些特定的 transformers 版本下运行，我们建议使用以下设置来评估对应的VLM:

- **请用** `transformers==4.33.0` **来运行**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **请用** `transformers==4.37.0 ` **来运行**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **请用** `transformers==4.40.0 ` **来运行**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **请用** `transformers==latest` **来运行**: `LLaVA-Next series`, `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`.

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

## 🛠️ 开发指南 <a id="development"></a>

要开发自定义评测数据集，支持其他 VLMs，或为 VLMEvalKit 贡献代码，请参阅[**开发指南**](/docs/zh-CN/Development_zh-CN.md)。

为激励来自社区的共享并分享相应的 credit，在下一次 report 更新中，我们将：

- 致谢所有的 contribution
- 具备三个或以上主要贡献 (支持新模型、评测集、或是主要特性) 的贡献者将可以加入技术报告的作者列表 。合条件的贡献者可以创建 issue 或是在 [VLMEvalKit Discord Channel](https://discord.com/invite/evDT4GZmxN) 私信 kennyutc，我们将进行跟进

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
@misc{duan2024vlmevalkit,
      title={VLMEvalKit: An Open-Source Toolkit for Evaluating Large Multi-Modality Models},
      author={Haodong Duan and Junming Yang and Yuxuan Qiao and Xinyu Fang and Lin Chen and Yuan Liu and Xiaoyi Dong and Yuhang Zang and Pan Zhang and Jiaqi Wang and Dahua Lin and Kai Chen},
      year={2024},
      eprint={2407.11691},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11691},
}
```

<p align="right"><a href="#top">🔝回到顶部</a></p>

[github-contributors-link]: https://github.com/open-compass/VLMEvalKit/graphs/contributors
[github-contributors-shield]: https://img.shields.io/github/contributors/open-compass/VLMEvalKit?color=c4f042&labelColor=black&style=flat-square
[github-forks-link]: https://github.com/open-compass/VLMEvalKit/network/members
[github-forks-shield]: https://img.shields.io/github/forks/open-compass/VLMEvalKit?color=8ae8ff&labelColor=black&style=flat-square
[github-issues-link]: https://github.com/open-compass/VLMEvalKit/issues
[github-issues-shield]: https://img.shields.io/github/issues/open-compass/VLMEvalKit?color=ff80eb&labelColor=black&style=flat-square
[github-license-link]: https://github.com/open-compass/VLMEvalKit/blob/main/LICENSE
[github-license-shield]: https://img.shields.io/github/license/open-compass/VLMEvalKit?color=white&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/open-compass/VLMEvalKit/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/open-compass/VLMEvalKit?color=ffcb47&labelColor=black&style=flat-square
