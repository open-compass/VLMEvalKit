<div align="center">

![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>A Toolkit for Evaluating Large Vision-Language Models. </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

English | [简体中文](/docs/zh-CN/README_zh-CN.md) | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OC Learderboard </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Report</a>
</div>

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

## 🆕 News
- **[2024-09-03]** We have supported [**RBDash**](https://huggingface.co/RBDash-Team/RBDash-v1.2-72b), thanks to [**anzhao920**](https://github.com/anzhao920) 🔥🔥🔥
- **[2024-09-03]** In the previous 2 months, 7 new contributors have made 3+ major contributions to the project: [amitbcp](https://github.com/amitbcp), [czczup](https://github.com/czczup), [DseidLi](https://github.com/DseidLi), [mayubo233](https://github.com/mayubo233), [sun-hailong](https://github.com/sun-hailong), [PhoenixZ810](https://github.com/PhoenixZ810), [Cuiunbo](https://github.com/Cuiunbo). We will update the report accordingly in the coming weeks. Check [contributor list](/docs/en/advanced_guides/Contributors.md) for their detailed contributions 🔥🔥🔥
- **[2024-09-03]** We have supported [**MME-RealWorld**](https://arxiv.org/abs/2408.13257), thanks to [**Yifan zhang**](https://github.com/yfzhang114) 🔥🔥🔥
- **[2024-09-02]** We have supported [**TableVQABench**](https://arxiv.org/abs/2404.19205), thanks to [**hkunzhe**](https://github.com/hkunzhe) 🔥🔥🔥
- **[2024-08-29]** We have supported [**PLLaVA**](https://arxiv.org/abs/2404.16994), including three different sizes (7B/13B/34B)  🔥🔥🔥
- **[2024-08-29]** We have supported [**GMAI-MMBench**](https://huggingface.co/papers/2408.03361), thanks to [**TousenKaname**](https://github.com/TousenKaname). Reference Performance: GPT-4o-MINI achieves 42.2% Overall accuracy 🔥🔥🔥
- **[2024-08-29]** We have supported [**MUIRBench**](https://muirbench.github.io), thanks to **amitbcp**. Reference Performance: GPT-4o-MINI achieves 64.6% Overall accuracy  🔥🔥🔥
- **[2024-08-28]** We have supported [**VideoChat2-HD**](https://huggingface.co/OpenGVLab/VideoChat2_HD_stage4_Mistral_7B)  🔥🔥🔥
- **[2024-08-25]** We have supported [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) with its [**video**](https://huggingface.co/datasets/OpenGVLab/MVBench/tree/video) version 🔥🔥🔥
- **[2024-08-25]** We have supported [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT), [**Chat-UniVi**](https://github.com/PKU-YuanGroup/Chat-UniVi) and [**LLaMA-VID**](https://github.com/dvlab-research/LLaMA-VID), thanks to [**Xiangyu Zhao**](https://github.com/PhoenixZ810)🔥🔥🔥
- **[2024-08-20]** We optimized the evaluation pipeline of [**MMMB and Multilingual MMBench**](https://arxiv.org/abs/2406.02539), now you can use the name `MMMB` and `MTL_MMBench_DEV` to obtain the results of 6 langs at the a time

## 📊 Datasets, Models, and Evaluation Results

**The performance numbers on our official multi-modal leaderboards can be downloaded from here!**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [Download All DETAILED Results](http://opencompass.openxlab.space/assets/OpenVLM.json).

**Supported Image Understanding Dataset**

- By default, all evaluation results are presented in [**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard).
- Abbrs: `MCQ`: Multi-choice question; `Y/N`: Yes-or-No Questions; `MTT`: Benchmark with Multi-turn Conversations; `MTI`: Benchmark with Multi-Image as Inputs.

| Dataset                                                      | Dataset Names (for run.py)                             | Task | Dataset | Dataset Names (for run.py) | Task |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | --------- | --------- | --------- |
| [**MMBench Series**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN]<br>MMBench\_DEV\_[EN/CN]\_V11<br>MMBench\_TEST\_[EN/CN]\_V11<br>CCBench | MCQ | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | MCQ |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | Y/N                                       | [**SEEDBench Series**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus | MCQ                                                |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | [**MMMU**](https://mmmu-benchmark.github.io)  | MMMU_[DEV_VAL/TEST]                      | MCQ                                |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | [**ScienceQA_IMG**](https://scienceqa.github.io) | ScienceQA_[VAL/TEST]                     | MCQ                        |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | Caption                                              | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                | Y/N                             |
| [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST] | VQA                                 | [**TextVQA**](https://textvqa.org)* | TextVQA_VAL                      | VQA                              |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST | VQA                                          | [**AI2D**](https://allenai.org/data/diagrams) | AI2D_[TEST/TEST_NO_MASK]                                 | MCQ                         |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | [**DocVQA**](https://www.docvqa.org)+       | DocVQA_[VAL/TEST]                           | VQA                                         |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST] | VQA | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | MCQ                                          | [**POPE**](https://github.com/AoiDragon/POPE) | POPE                                           | Y/N                                            |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM (MTI) | VQA                                               | [**MMT-Bench**](https://mmt-bench.github.io)                 | MMT-Bench\_[VAL/ALL]<br>MMT-Bench\_[VAL/ALL]_MI | MCQ (MTI) |
| [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS | VQA | [**AesBench**](https://github.com/yipoh/AesBench)+ | AesBench_[VAL/TEST] | MCQ |
| [**VCR-wiki**](https://huggingface.co/vcr-org/) + | VCR\_[EN/ZH]\_[EASY/HARD]_[ALL/500/100] | VQA | [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/)+ | MMLongBench_DOC | VQA (MTI) |
| [**BLINK**](https://zeyofu.github.io/blink/) | BLINK | MCQ (MTI) | [**MathVision**](https://mathvision-cuhk.github.io)+ | MathVision<br>MathVision_MINI | VQA |
| [**MT-VQA**](https://github.com/bytedance/MTVQA) | MTVQA_TEST | VQA | [**MMDU**](https://liuziyu77.github.io/MMDU/)+ | MMDU | VQA (MTT, MTI) |
| [**Q-Bench1**](https://github.com/Q-Future/Q-Bench) | Q-Bench1_[VAL/TEST] | MCQ | [**A-Bench**](https://github.com/Q-Future/A-Bench) | A-Bench_[VAL/TEST] | MCQ |
| [**DUDE**](https://arxiv.org/abs/2305.08455)+ | DUDE | VQA (MTI) | [**SlideVQA**](https://arxiv.org/abs/2301.04883)+ | SLIDEVQA<br>SLIDEVQA_MINI | VQA (MTI) |
| [**TaskMeAnything ImageQA Random**](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random)+ | TaskMeAnything_v1_imageqa_random | MCQ  | [**MMMB and Multilingual MMBench**](https://sun-hailong.github.io/projects/Parrot/)+ | MMMB\_[ar/cn/en/pt/ru/tr]<br>MMBench_dev\_[ar/cn/en/pt/ru/tr]<br>MMMB<br>MTL_MMBench_DEV<br>PS: MMMB & MTL_MMBench_DEV <br>are **all-in-one** names for 6 langs | MCQ  |
| [**A-OKVQA**](https://arxiv.org/abs/2206.01718)+ | A-OKVQA | MCQ | [**MuirBench**](https://muirbench.github.io)+ | MUIRBench | MCQ |
| [**GMAI-MMBench**](https://huggingface.co/papers/2408.03361)+ | GMAI-MMBench_VAL | MCQ | [**TableVQABench**](https://arxiv.org/abs/2404.19205)+ | TableVQABench | VQA |
| [**MME-RealWorld**](https://arxiv.org/abs/2408.13257)+ | MME-RealWorld[-CN] | MCQ |  |  |  |

**\*** We only provide a subset of the evaluation results, since some VLMs do not yield reasonable results under the zero-shot setting

**\+** The evaluation results are not available yet

**\-** Only inference is supported in VLMEvalKit

VLMEvalKit will use a **judge LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**

**Supported Video Understanding Dataset**

| Dataset                                              | Dataset Names (for run.py) | Task | Dataset | Dataset Names (for run.py) | Task |
| ---------------------------------------------------- | -------------------------- | ---- | ------- | -------------------------- | ---- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video              | VQA  | [**Video-MME**](https://video-mme.github.io/)        |    Video-MME                        | MCQ     |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md)| MVBench/MVBench_MP4              | MCQ  |       |                |  |

**Supported API Models**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude3-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 | [**GPT-4o-Mini**](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 🎞️🚅 | [**Yi-Vision**](https://platform.lingyiwanwu.com)🎞️🚅          | [**Hunyuan-Vision**](https://cloud.tencent.com/document/product/1729)🎞️🚅 |                                                   |

**Supported PyTorch / HF Models**

| [**IDEFICS-[9B/80B/v2-8B/v3-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🚅🎞️ | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl2**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🚅🎞️ , [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🚅🎞️ |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-[1/2]**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅 | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-[Chat/Llama3]**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**ShareCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅, [**Monkey-Chat**](https://github.com/Yuliang-Liu/Monkey)🚅 | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️         | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer-2.5**](https://github.com/InternLM/InternLM-XComposer)🚅🎞️ | [**MiniCPM-[V1/V2/V2.5/V2.6]**](https://github.com/OpenBMB/MiniCPM-V)🚅🎞️ | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat-[V1-1/V1-2/V1-5/V2]**](https://github.com/OpenGVLab/InternVL)🚅🎞️, <br>[**Mini-InternVL-Chat-[2B/4B]-V1-5**](https://github.com/OpenGVLab/InternVL)🚅🎞️ |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅🎞️ | [**Bunny-Llama3**](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)🚅 | [**XVERSE-V-13B**](https://github.com/xverse-ai/XVERSE-V-13B/blob/main/vxverse/models/vxverse.py) |
| [**PaliGemma-3B**](https://huggingface.co/google/paligemma-3b-pt-448) 🚅 | [**360VL-70B**](https://huggingface.co/qihoo360/360VL-70B) 🚅 | [**Phi-3-Vision**](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)/[**Phi-3.5-Vision**](https://huggingface.co/microsoft/Phi-3.5-vision-instruct)🚅🎞️ | [**WeMM**](https://github.com/scenarios/WeMM)🚅               |
| [**GLM-4v-9B**](https://huggingface.co/THUDM/glm-4v-9b) 🚅    | [**Cambrian-[8B/13B/34B]**](https://cambrian-mllm.github.io/) | [**LLaVA-Next-[Qwen-32B]**](https://huggingface.co/lmms-lab/llava-next-qwen-32b) 🎞️ | [**Chameleon-[7B/30B]**](https://huggingface.co/facebook/chameleon-7b)🚅🎞️ |
| [**Video-LLaVA-7B-[HF]**](https://github.com/PKU-YuanGroup/Video-LLaVA) 🎬 | [**VILA1.5-[3B/8B/13B/40B]**](https://github.com/NVlabs/VILA/)🎞️ | [**Ovis1.5-[Llama3-8B/Gemma2-9B]**](https://github.com/AIDC-AI/Ovis) 🚅🎞️ | [**Mantis-8B-[siglip-llama3/clip-llama3/Idefics2/Fuyu]**](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) 🎞️ |
| [**Llama-3-MixSenseV1_1**](https://huggingface.co/Zero-Vision/Llama-3-MixSenseV1_1)🚅 | [**Parrot-7B**](https://github.com/AIDC-AI/Parrot) 🚅         | [**OmChat-v2.0-13B-sinlge-beta**](https://huggingface.co/omlab/omchat-v2.0-13B-single-beta_hf)  🚅 | [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT) 🎬 |
| [**Chat-UniVi**](https://github.com/PKU-YuanGroup/Chat-UniVi) 🎬 | [**LLaMA-VID-7B**](https://github.com/dvlab-research/LLaMA-VID) 🎬 | [**VideoChat2-HD**](https://huggingface.co/OpenGVLab/VideoChat2_HD_stage4_Mistral_7B) 🎬 | [**PLLaVA-[7B/13B/34B]**](https://huggingface.co/ermu2001/pllava-7b) 🎬 |
| [**RBDash_72b**](https://github.com/RBDash-Team/RBDash) 🚅🎞️   | [mPLUG                                                       |                                                              |                                                              |

🎞️: Support multiple images as inputs.

🚅: Models can be used without any additional configuration/operation.

🎬: Support Video as inputs.

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **Please use** `transformers==latest` **for**: `LLaVA-Next series`, `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`, `RBDash_72b`.

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

See [[QuickStart](/docs/en/get_started/Quickstart.md) | [快速开始](/docs/zh-CN/get_started/Quickstart.md)] for a quick start guide.

## 🛠️ Development Guide

To develop custom benchmarks, VLMs, or simply contribute other codes to **VLMEvalKit**, please refer to [[Development_Guide](/docs/en/advanced_guides/Development.md) | [开发指南](/docs/zh-CN/advanced_guides/Development.md)].

**Call for contributions**

To promote the contribution from the community and share the corresponding credit (in the next report update):

- All Contributions will be acknowledged in the report.
- Contributors with 3 or more major contributions (implementing an MLLM, benchmark, or major feature) can join the author list of [VLMEvalKit Technical Report](https://www.arxiv.org/abs/2407.11691) on ArXiv. Eligible contributors can create an issue or dm kennyutc in [VLMEvalKit Discord Channel](https://discord.com/invite/evDT4GZmxN).

Here is a [contributor list](/docs/en/advanced_guides/Contributors.md) we curated based on the records.

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

<p align="right"><a href="#top">🔝Back to top</a></p>

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
