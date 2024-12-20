![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>A Toolkit for Evaluating Large Vision-Language Models. </b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

English | [简体中文](/docs/zh-CN/README_zh-CN.md) | [日本語](/docs/ja/README_ja.md)

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OC Learderboard </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://huggingface.co/spaces/opencompass/openvlm_video_leaderboard">🤗 HF Video Leaderboard</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Report</a>
</div>

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

## 🆕 News

> We have presented a [**comprehensive survey**](https://arxiv.org/pdf/2411.15296) on the evaluation of large multi-modality models, jointly with [**MME Team**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models) and [**LMMs-Lab**](https://lmms-lab.github.io) 🔥🔥🔥

- **[2024-12-11]** Supported [**NaturalBench**](https://huggingface.co/datasets/BaiqiL/NaturalBench), a vision-centric VQA benchmark (NeurIPS'24) that challenges vision-language models with simple questions about natural imagery.
- **[2024-12-02]** Supported [**VisOnlyQA**](https://github.com/psunlpgroup/VisOnlyQA/), a benchmark for evaluating the visual perception capabilities 🔥🔥🔥
- **[2024-11-26]** Supported [**Ovis1.6-Gemma2-27B**](https://huggingface.co/AIDC-AI/Ovis1.6-Gemma2-27B), thanks to [**runninglsy**](https://github.com/runninglsy) 🔥🔥🔥
- **[2024-11-25]** Create a new flag `VLMEVALKIT_USE_MODELSCOPE`. By setting this environment variable, you can download the video benchmarks supported from [**modelscope**](https://www.modelscope.cn) 🔥🔥🔥
- **[2024-11-25]** Supported [**VizWiz**](https://vizwiz.org/tasks/vqa/) benchmark 🔥🔥🔥
- **[2024-11-22]** Supported the inference of [**MMGenBench**](https://mmgenbench.alsoai.com), thanks [**lerogo**](https://github.com/lerogo) 🔥🔥🔥
- **[2024-11-22]** Supported [**Dynamath**](https://huggingface.co/datasets/DynaMath/DynaMath_Sample), a multimodal math benchmark comprising of 501 SEED problems and 10 variants generated based on random seeds. The benchmark can be used to measure the robustness of MLLMs in multi-modal math solving 🔥🔥🔥
- **[2024-11-21]** Integrated a new config system to enable more flexible evaluation settings. Check the [Document](/docs/en/ConfigSystem.md) or run `python run.py --help` for more details 🔥🔥🔥
- **[2024-11-21]** Supported [**QSpatial**](https://andrewliao11.github.io/spatial_prompt/), a multimodal benchmark for Quantitative Spatial Reasoning (determine the size / distance, e.g.), thanks [**andrewliao11**](https://github.com/andrewliao11)  for providing the official support 🔥🔥🔥
- **[2024-11-21]** Supported [**MM-Math**](https://github.com/kge-sun/mm-math), a new multimodal math benchmark comprising of ~6K middle school multi-modal reasoning math problems. GPT-4o-20240806 achieces 22.5% accuracy on this benchmark 🔥🔥🔥

## 🏗️ QuickStart

See [[QuickStart](/docs/en/Quickstart.md) | [快速开始](/docs/zh-CN/Quickstart.md)] for a quick start guide.

## 📊 Datasets, Models, and Evaluation Results

### Evaluation Results

**The performance numbers on our official multi-modal leaderboards can be downloaded from here!**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [**Download All DETAILED Results**](http://opencompass.openxlab.space/assets/OpenVLM.json).

### Supported Benchmarks

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
| [**MME-RealWorld**](https://arxiv.org/abs/2408.13257)+ | MME-RealWorld[-CN]<br/>[MME-RealWorld-Lite](https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite) | MCQ | [**HRBench**](https://arxiv.org/abs/2408.15556)+ | HRBench[4K/8K] | MCQ |
| [**MathVerse**](https://mathverse-cuhk.github.io/)+ | MathVerse_MINI<br/>MathVerse_MINI_Vision_Only <br/>MathVerse_MINI_Vision_Dominant<br/>MathVerse_MINI_Vision_Intensive<br/>MathVerse_MINI_Text_Lite<br/>MathVerse_MINI_Text_Dominant | VQA | [**AMBER**](https://github.com/junyangwang0410/AMBER)+ | AMBER | Y/N |
| [**CRPE**](https://huggingface.co/datasets/OpenGVLab/CRPE)+ | CRPE_[EXIST/RELATION] | VQA | [**MMSearch**](https://mmsearch.github.io/)$$^1$$ | - | **-** |
| [**R-Bench**](https://arxiv.org/abs/2410.05474)+ | R-Bench-[Dis/Ref] | MCQ | [**WorldMedQA-V**](https://www.arxiv.org/abs/2410.12722)+ | WorldMedQA-V | MCQ |
| [**GQA**](https://cs.stanford.edu/people/dorarad/gqa/about.html)+ | GQA_TestDev_Balanced | VQA | [**MIA-Bench**](https://arxiv.org/abs/2407.01509)+ | MIA-Bench | VQA |
| [**WildVision**](https://huggingface.co/datasets/WildVision/wildvision-bench)+ | WildVision | VQA | [**OlympiadBench**](https://github.com/OpenBMB/OlympiadBench)+ | OlympiadBench | VQA |
| [**MM-Math**](https://github.com/kge-sun/mm-math)+ | MM-Math | VQA | [**Dynamath**](https://huggingface.co/datasets/DynaMath/DynaMath_Sample) | DynaMath | VQA |
| [**MMGenBench**](https://mmgenbench.alsoai.com/)- | MMGenBench-Test<br>MMGenBench-Domain | - | [**QSpatial**](https://andrewliao11.github.io/spatial_prompt/)+ | QSpatial_[plus/scannet] | VQA |
| [**VizWiz**](https://vizwiz.org/tasks/vqa/)+ | VizWiz | VQA | [**VisOnlyQA**](https://github.com/psunlpgroup/VisOnlyQA/)+ | VisOnlyQA-VLMEvalKit | MCQ |

**\*** We only provide a subset of the evaluation results, since some VLMs do not yield reasonable results under the zero-shot setting

**\+** The evaluation results are not available yet

**\-** Only inference is supported in VLMEvalKit (That includes the `TEST` splits of some benchmarks that do not include the ground truth answers).

$$^1$$ VLMEvalKit is integrated in its official repository.

VLMEvalKit will use a **judge LLM** to extract answer from the output if you set the key, otherwise it uses the **exact matching** mode (find "Yes", "No", "A", "B", "C"... in the output strings). **The exact matching can only be applied to the Yes-or-No tasks and the Multi-choice tasks.**

**Supported Video Understanding Dataset**

| Dataset                                              | Dataset Names (for run.py) | Task | Dataset | Dataset Names (for run.py) | Task |
| ---------------------------------------------------- | -------------------------- | ---- | ------- | -------------------------- | ---- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video              | VQA  | [**Video-MME**](https://video-mme.github.io/)        |    Video-MME                        | MCQ     |
| [**MVBench**](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md) | MVBench/MVBench_MP4              | MCQ  | [**MLVU**](https://github.com/JUNJIE99/MLVU) | MLVU | MCQ & VQA |
| [**TempCompass**](https://arxiv.org/abs/2403.00476) | TempCompass | MCQ & Y/N & Caption | [**LongVideoBench**](https://longvideobench.github.io/) | LongVideoBench | MCQ |

### Supported Models

**Supported API Models**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅<br>[**Qwen-VL-[Plus / Max]-0809**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude3-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet (20240620, 20241022)**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 | [**GPT-4o-Mini**](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 🎞️🚅 | [**Yi-Vision**](https://platform.lingyiwanwu.com)🎞️🚅          | [**Hunyuan-Vision**](https://cloud.tencent.com/document/product/1729)🎞️🚅 | [**BlueLM-V**](https://developers.vivo.com/) 🎞️🚅   |
| [**TeleMM**](https://cloud.siliconflow.cn/playground/chat/17885302607)🎞️🚅                                                 |

**Supported PyTorch / HF Models**

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
| [**Video-LLaVA-7B-[HF]**](https://github.com/PKU-YuanGroup/Video-LLaVA) 🎬 | [**VILA1.5-[3B/8B/13B/40B]**](https://github.com/NVlabs/VILA/)🎞️ | [**Ovis[1.5-Llama3-8B/1.5-Gemma2-9B/1.6-Gemma2-9B/1.6-Llama3.2-3B/1.6-Gemma2-27B]**](https://github.com/AIDC-AI/Ovis) 🚅🎞️ | [**Mantis-8B-[siglip-llama3/clip-llama3/Idefics2/Fuyu]**](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) 🎞️ |
| [**Llama-3-MixSenseV1_1**](https://huggingface.co/Zero-Vision/Llama-3-MixSenseV1_1)🚅 | [**Parrot-7B**](https://github.com/AIDC-AI/Parrot) 🚅         | [**OmChat-v2.0-13B-sinlge-beta**](https://huggingface.co/omlab/omchat-v2.0-13B-single-beta_hf)  🚅 | [**Video-ChatGPT**](https://github.com/mbzuai-oryx/Video-ChatGPT) 🎬 |
| [**Chat-UniVi-7B[-v1.5]**](https://github.com/PKU-YuanGroup/Chat-UniVi) 🎬 | [**LLaMA-VID-7B**](https://github.com/dvlab-research/LLaMA-VID) 🎬 | [**VideoChat2-HD**](https://huggingface.co/OpenGVLab/VideoChat2_HD_stage4_Mistral_7B) 🎬 | [**PLLaVA-[7B/13B/34B]**](https://huggingface.co/ermu2001/pllava-7b) 🎬 |
| [**RBDash_72b**](https://github.com/RBDash-Team/RBDash) 🚅🎞️   | [**xgen-mm-phi3-[interleave/dpo]-r-v1.5**](https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5) 🚅🎞️ | [**Qwen2-VL-[2B/7B/72B]**](https://github.com/QwenLM/Qwen2-VL)🚅🎞️ | [**slime_[7b/8b/13b]**](https://github.com/yfzhang114/SliME)🎞️ |
| [**Eagle-X4-[8B/13B]**](https://github.com/NVlabs/EAGLE)🚅🎞️, <br>[**Eagle-X5-[7B/13B/34B]**](https://github.com/NVlabs/EAGLE)🚅🎞️ | [**Moondream1**](https://github.com/vikhyat/moondream)🚅, <br>[**Moondream2**](https://github.com/vikhyat/moondream)🚅 | [**XinYuan-VL-2B-Instruct**](https://huggingface.co/Cylingo/Xinyuan-VL-2B)🚅🎞️ | [**Llama-3.2-[11B/90B]-Vision-Instruct**](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)🚅 |
| [**Kosmos2**](https://huggingface.co/microsoft/kosmos-2-patch14-224)🚅 | [**H2OVL-Mississippi-[0.8B/2B]**](https://huggingface.co/h2oai/h2ovl-mississippi-2b)🚅🎞️ | [**Pixtral-12B**](https://huggingface.co/mistralai/Pixtral-12B-2409)🎞️ | [**Falcon2-VLM-11B**](https://huggingface.co/tiiuae/falcon-11B-vlm)🚅 |
| [**MiniMonkey**](https://huggingface.co/mx262/MiniMonkey)🚅🎞️  | [**LLaVA-OneVision**](https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-sft)🚅🎞️ | [**LLaVA-Video**](https://huggingface.co/collections/lmms-lab/llava-video-661e86f5e8dabc3ff793c944)🚅🎞️ | [**Aquila-VL-2B**](https://huggingface.co/BAAI/Aquila-VL-2B-llava-qwen)🚅🎞️ |
| [**Mini-InternVL-Chat-[2B/4B]-V1-5**](https://github.com/OpenGVLab/InternVL)🚅🎞️ | [**InternVL2 Series**](https://huggingface.co/OpenGVLab/InternVL2-8B) 🚅🎞️ | [**Janus-1.3B**](https://huggingface.co/deepseek-ai/Janus-1.3B)🚅🎞️ | [**molmoE-1B/molmo-7B/molmo-72B**](https://huggingface.co/allenai/Molmo-7B-D-0924)🚅 |
| [**Points-[Yi-1.5-9B/Qwen-2.5-7B]**](https://huggingface.co/WePOINTS/POINTS-Yi-1-5-9B-Chat)🚅 | [**NVLM**](https://huggingface.co/nvidia/NVLM-D-72B)🚅        | [**VIntern**](https://huggingface.co/5CD-AI/Vintern-3B-beta)🚅🎞️ |  [**Aria**](https://huggingface.co/rhymes-ai/Aria)🚅🎞️ |


🎞️: Support multiple images as inputs.

🚅: Models can be used without any additional configuration/operation.

🎬: Support Video as inputs.

**Transformers Version Recommendation:**

Note that some VLMs may not be able to run under certain transformer versions, we recommend the following settings to evaluate each VLM:

- **Please use** `transformers==4.33.0` **for**: `Qwen series`, `Monkey series`, `InternLM-XComposer Series`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICS series`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4 series`, `InstructBLIP series`, `PandaGPT`, `VXVERSE`.
- **Please use** `transformers==4.36.2` **for**: `Moondream1`.
- **Please use** `transformers==4.37.0` **for**: `LLaVA series`, `ShareGPT4V series`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLM Series`, `EMU2 Series`, `Yi-VL Series`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VL series`, `InternVL series`, `Cambrian Series`, `VILA Series`, `Llama-3-MixSenseV1_1`, `Parrot-7B`, `PLLaVA Series`.
- **Please use** `transformers==4.40.0` **for**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **Please use** `transformers==4.44.0` **for**: `Moondream2`, `H2OVL series`.
- **Please use** `transformers==4.45.0` **for**: `Aria`.
- **Please use** `transformers==latest` **for**: `LLaVA-Next series`, `PaliGemma-3B`, `Chameleon series`, `Video-LLaVA-7B-HF`, `Ovis series`, `Mantis series`, `MiniCPM-V2.6`, `OmChat-v2.0-13B-sinlge-beta`, `Idefics-3`, `GLM-4v-9B`, `VideoChat2-HD`, `RBDash_72b`, `Llama-3.2 series`, `Kosmos series`.

**Torchvision Version Recommendation:**

Note that some VLMs may not be able to run under certain torchvision versions, we recommend the following settings to evaluate each VLM:

- **Please use** `torchvision>=0.16` **for**: `Moondream series` and `Aria`

**Flash-attn Version Recommendation:**

Note that some VLMs may not be able to run under certain flash-attention versions, we recommend the following settings to evaluate each VLM:

- **Please use** `pip install flash-attn --no-build-isolation` **for**: `Aria`

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

## 🛠️ Development Guide

To develop custom benchmarks, VLMs, or simply contribute other codes to **VLMEvalKit**, please refer to [[Development_Guide](/docs/en/Development.md) | [开发指南](/docs/zh-CN/Development.md)].

**Call for contributions**

To promote the contribution from the community and share the corresponding credit (in the next report update):

- All Contributions will be acknowledged in the report.
- Contributors with 3 or more major contributions (implementing an MLLM, benchmark, or major feature) can join the author list of [VLMEvalKit Technical Report](https://www.arxiv.org/abs/2407.11691) on ArXiv. Eligible contributors can create an issue or dm kennyutc in [VLMEvalKit Discord Channel](https://discord.com/invite/evDT4GZmxN).

Here is a [contributor list](/docs/en/Contributors.md) we curated based on the records.

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
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
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
