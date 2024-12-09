<div align="center">

![LOGO](http://opencompass.openxlab.space/utils/MMLB.jpg)

<b>VLMEvalKit: 大規模視覚言語モデルの評価ツールキット</b>

[![][github-contributors-shield]][github-contributors-link] • [![][github-forks-shield]][github-forks-link] • [![][github-stars-shield]][github-stars-link] • [![][github-issues-shield]][github-issues-link] • [![][github-license-shield]][github-license-link]

[English](/README.md) | [简体中文](/docs/zh-CN/README_zh-CN.md) | 日本語

<a href="https://rank.opencompass.org.cn/leaderboard-multimodal">🏆 OpenCompass Learderboard </a> •
<a href="#-datasets-models-and-evaluation-results">📊Datasets & Models </a> •
<a href="#%EF%B8%8F-quickstart">🏗️Quickstart </a> •
<a href="#%EF%B8%8F-development-guide">🛠️Development </a> •
<a href="#-the-goal-of-vlmevalkit">🎯Goal </a> •
<a href="#%EF%B8%8F-citation">🖊️Citation </a>

<a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard">🤗 HF Leaderboard</a> •
<a href="https://huggingface.co/datasets/VLMEval/OpenVLMRecords">🤗 Evaluation Records</a> •
<a href="https://discord.gg/evDT4GZmxN">🔊 Discord Channel</a> •
<a href="https://www.arxiv.org/abs/2407.11691">📝 Technical Report</a>
</div>

**VLMEvalKit**（pythonパッケージ名は**vlmeval**）は、**大規模視覚言語モデル（LVLMs）**の**オープンソース評価ツールキット**です。このツールキットは、複数のリポジトリでのデータ準備という重労働なしに、さまざまなベンチマークでLVLMsの**ワンコマンド評価**を可能にします。VLMEvalKitでは、すべてのLVLMsに対して**生成ベースの評価**を採用し、**正確なマッチング**と**LLMベースの回答抽出**の両方で得られた評価結果を提供します。

PS: 日本語の README には最新のアップデートがすべて含まれていない場合があります。英語版をご確認ください。

## 📊 データセット、モデル、および評価結果

**公式のマルチモーダルリーダーボードでのパフォーマンス数値は、ここからダウンロードできます！**

[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard): [すべての詳細な結果をダウンロード](http://opencompass.openxlab.space/assets/OpenVLM.json)。

**Supported Image Understanding Dataset**

- デフォルトでは、すべての評価結果は[**OpenVLM Leaderboard**](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)に表示されます。

| データセット                                                      | データセット名 (run.py用)                             | タスク | データセット | データセット名 (run.py用) | タスク |
| ------------------------------------------------------------ | ------------------------------------------------------ | --------- | --------- | --------- | --------- |
| [**MMBench シリーズ**](https://github.com/open-compass/mmbench/): <br>MMBench, MMBench-CN, CCBench | MMBench\_DEV\_[EN/CN] <br>MMBench\_TEST\_[EN/CN]<br>MMBench\_DEV\_[EN/CN]\_V11<br>MMBench\_TEST\_[EN/CN]\_V11<br>CCBench | 多肢選択問題 (MCQ) | [**MMStar**](https://github.com/MMStar-Benchmark/MMStar) | MMStar | MCQ |
| [**MME**](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) | MME | はい/いいえ (Y/N)                                         | [**SEEDBench シリーズ**](https://github.com/AILab-CVC/SEED-Bench) | SEEDBench_IMG <br>SEEDBench2 <br>SEEDBench2_Plus | MCQ                                                |
| [**MM-Vet**](https://github.com/yuweihao/MM-Vet)             | MMVet  | VQA                                              | [**MMMU**](https://mmmu-benchmark.github.io)  | MMMU_[DEV_VAL/TEST]                      | MCQ                                |
| [**MathVista**](https://mathvista.github.io)                 | MathVista_MINI | VQA                                         | [**ScienceQA_IMG**](https://scienceqa.github.io) | ScienceQA_[VAL/TEST]                     | MCQ                        |
| [**COCO Caption**](https://cocodataset.org)                  | COCO_VAL | キャプション                                              | [**HallusionBench**](https://github.com/tianyi-lab/HallusionBench) | HallusionBench                                | Y/N                             |
| [**OCRVQA**](https://ocr-vqa.github.io)*                     | OCRVQA_[TESTCORE/TEST] | VQA                                 | [**TextVQA**](https://textvqa.org)* | TextVQA_VAL                      | VQA                              |
| [**ChartQA**](https://github.com/vis-nlp/ChartQA)*           | ChartQA_TEST | VQA                                          | [**AI2D**](https://allenai.org/data/diagrams) | AI2D_[TEST/TEST_NO_MASK]                                 | MCQ                         |
| [**LLaVABench**](https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild) | LLaVABench | VQA                                            | [**DocVQA**](https://www.docvqa.org)+       | DocVQA_[VAL/TEST]                           | VQA                                         |
| [**InfoVQA**](https://www.docvqa.org/datasets/infographicvqa)+ | InfoVQA_[VAL/TEST] | VQA | [**OCRBench**](https://github.com/Yuliang-Liu/MultimodalOCR) | OCRBench | VQA |
| [**RealWorldQA**](https://x.ai/blog/grok-1.5v)            | RealWorldQA | MCQ                                          | [**POPE**](https://github.com/AoiDragon/POPE) | POPE                                           | Y/N                                            |
| [**Core-MM**](https://github.com/core-mm/core-mm)-          | CORE_MM | VQA                                               | [**MMT-Bench**](https://mmt-bench.github.io)                 | MMT-Bench_[VAL/VAL_MI/ALL/ALL_MI]                | MCQ  |
| [**MLLMGuard**](https://github.com/Carol-gutianle/MLLMGuard) - | MLLMGuard_DS | VQA | [**AesBench**](https://github.com/yipoh/AesBench) | AesBench_[VAL/TEST] | MCQ |
| [**VCR-wiki**](https://huggingface.co/vcr-org/) + | VCR\_[EN/ZH]\_[EASY/HARD]_[ALL/500/100] | VQA | [**MMLongBench-Doc**](https://mayubo2333.github.io/MMLongBench-Doc/)+ | MMLongBench_DOC | VQA |
| [**BLINK**](https://zeyofu.github.io/blink/) + | BLINK | MCQ | [**MathVision**](https://mathvision-cuhk.github.io)+ | MathVision<br>MathVision_MINI | VQA |
| [**MT-VQA**](https://github.com/bytedance/MTVQA)+ | MTVQA_TEST | VQA | [**MMDU**](https://liuziyu77.github.io/MMDU/)+ | MMDU | VQA (multi-turn) |
| [**Q-Bench1**](https://github.com/Q-Future/Q-Bench)+ | Q-Bench1_[VAL/TEST] | MCQ | [**A-Bench**](https://github.com/Q-Future/A-Bench)+ | A-Bench_[VAL/TEST] | MCQ |
| [**TaskMeAnything ImageQA Random**](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random)+ | TaskMeAnything_v1_imageqa_random | MCQ  | | | |

**\*** ゼロショット設定で合理的な結果を出せないVLMの一部の評価結果のみを提供しています

**\+** 評価結果はまだ利用できません

**\-** VLMEvalKitでは推論のみがサポートされています

VLMEvalKitは、キーを設定すると**判定LLM**を使用して出力から回答を抽出し、それ以外の場合は**正確なマッチング**モード（出力文字列で「はい」、「いいえ」、「A」、「B」、「C」...を検索）を使用します。**正確なマッチングは、はい/いいえのタスクと多肢選択問題にのみ適用できます。**

**Supported Video Understanding Dataset**

| Dataset                                              | Dataset Names (for run.py) | Task | Dataset                                       | Dataset Names (for run.py) | Task |
| ---------------------------------------------------- | -------------------------- | ---- | --------------------------------------------- | -------------------------- | ---- |
| [**MMBench-Video**](https://mmbench-video.github.io) | MMBench-Video              | VQA  | [**Video-MME**](https://video-mme.github.io/) | Video-MME                  | MCQ  |

**Supported API Models**

| [**GPT-4v (20231106, 20240409)**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**GPT-4o**](https://openai.com/index/hello-gpt-4o/) 🎞️🚅      | [**Gemini-1.0-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Gemini-1.5-Pro**](https://platform.openai.com/docs/guides/vision) 🎞️🚅 | [**Step-1V**](https://www.stepfun.com/#step1v) 🎞️🚅 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- |
| [**Reka-[Edge / Flash / Core]**](https://www.reka.ai)🚅       | [**Qwen-VL-[Plus / Max]**](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 🎞️🚅 | [**Claude-3v-[Haiku / Sonnet / Opus]**](https://www.anthropic.com/news/claude-3-family) 🎞️🚅 | [**GLM-4v**](https://open.bigmodel.cn/dev/howuse/glm4v) 🚅    | [**CongRong**](https://mllm.cloudwalk.com/web) 🎞️🚅 |
| [**Claude3.5-Sonnet**](https://www.anthropic.com/news/claude-3-5-sonnet) 🎞️🚅 | [**GPT-4o-Mini**](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 🎞️🚅 | [**Yi-Vision**](https://platform.lingyiwanwu.com)🎞️🚅          | [**Hunyuan-Vision**](https://cloud.tencent.com/document/product/1729)🎞️🚅 |   [**BlueLM-V**](https://developers.vivo.com/) 🎞️🚅  |

**Supported PyTorch / HF Models**

| [**IDEFICS-[9B/80B/v2-8B]-Instruct**](https://huggingface.co/HuggingFaceM4/idefics-9b-instruct)🎞️🚅 | [**InstructBLIP-[7B/13B]**](https://github.com/salesforce/LAVIS/blob/main/projects/instructblip/README.md) | [**LLaVA-[v1-7B/v1.5-7B/v1.5-13B]**](https://github.com/haotian-liu/LLaVA) | [**MiniGPT-4-[v1-7B/v1-13B/v2-7B]**](https://github.com/Vision-CAIR/MiniGPT-4) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [**mPLUG-Owl2**](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2)🎞️ | [**OpenFlamingo-v2**](https://github.com/mlfoundations/open_flamingo)🎞️ | [**PandaGPT-13B**](https://github.com/yxuansu/PandaGPT)      | [**Qwen-VL**](https://huggingface.co/Qwen/Qwen-VL)🎞️🚅, [**Qwen-VL-Chat**](https://huggingface.co/Qwen/Qwen-VL-Chat)🎞️**🚅** |
| [**VisualGLM-6B**](https://huggingface.co/THUDM/visualglm-6b)🚅 | [**InternLM-XComposer-[1/2]**](https://huggingface.co/internlm/internlm-xcomposer-7b)🚅 | [**ShareGPT4V-[7B/13B]**](https://sharegpt4v.github.io)🚅     | [**TransCore-M**](https://github.com/PCIResearch/TransCore-M) |
| [**LLaVA (XTuner)**](https://huggingface.co/xtuner/llava-internlm-7b)🚅 | [**CogVLM-[Chat/Llama3]**](https://huggingface.co/THUDM/cogvlm-chat-hf)🚅 | [**ShareCaptioner**](https://huggingface.co/spaces/Lin-Chen/Share-Captioner)🚅 | [**CogVLM-Grounding-Generalist**](https://huggingface.co/THUDM/cogvlm-grounding-generalist-hf)🚅 |
| [**Monkey**](https://github.com/Yuliang-Liu/Monkey)🚅, [**Monkey-Chat**](https://github.com/Yuliang-Liu/Monkey)🚅 | [**EMU2-Chat**](https://github.com/baaivision/Emu)🚅🎞️         | [**Yi-VL-[6B/34B]**](https://huggingface.co/01-ai/Yi-VL-6B)  | [**MMAlaya**](https://huggingface.co/DataCanvas/MMAlaya)🚅    |
| [**InternLM-XComposer-2.5**](https://github.com/InternLM/InternLM-XComposer)🚅🎞️ | [**MiniCPM-[V1/V2/V2.5/V2.6]**](https://github.com/OpenBMB/MiniCPM-V)🚅🎞️ | [**OmniLMM-12B**](https://huggingface.co/openbmb/OmniLMM-12B) | [**InternVL-Chat-[V1-1/V1-2/V1-5/V2]**](https://github.com/OpenGVLab/InternVL)🚅🎞️, <br>[**Mini-InternVL-Chat-[2B/4B]-V1-5**](https://github.com/OpenGVLab/InternVL)🚅🎞️ |
| [**DeepSeek-VL**](https://github.com/deepseek-ai/DeepSeek-VL/tree/main)🎞️ | [**LLaVA-NeXT**](https://llava-vl.github.io/blog/2024-01-30-llava-next/)🚅🎞️ | [**Bunny-Llama3**](https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V)🚅 | [**XVERSE-V-13B**](https://github.com/xverse-ai/XVERSE-V-13B/blob/main/vxverse/models/vxverse.py) |
| [**PaliGemma-3B**](https://huggingface.co/google/paligemma-3b-pt-448) 🚅 | [**360VL-70B**](https://huggingface.co/qihoo360/360VL-70B) 🚅 | [**Phi-3-Vision**](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)🚅 | [**WeMM**](https://github.com/scenarios/WeMM)🚅               |
| [**GLM-4v-9B**](https://huggingface.co/THUDM/glm-4v-9b) 🚅    | [**Cambrian-[8B/13B/34B]**](https://cambrian-mllm.github.io/) | [**LLaVA-Next-[Qwen-32B]**](https://huggingface.co/lmms-lab/llava-next-qwen-32b) 🎞️ | [**Chameleon-[7B/30B]**](https://huggingface.co/facebook/chameleon-7b)🚅🎞️ |
| [**Video-LLaVA-7B-[HF]**](https://github.com/PKU-YuanGroup/Video-LLaVA) 🎬 | [**VILA1.5-[8B/13B/40B]**](https://github.com/NVlabs/VILA/)🎞️ | [**Ovis1.5-Llama3-8B**](https://github.com/AIDC-AI/Ovis) 🚅🎞  | [**Mantis-8B-[siglip-llama3/clip-llama3/Idefics2/Fuyu]**](https://huggingface.co/TIGER-Lab/Mantis-8B-Idefics2) 🎞️ |

🎞️: 複数の画像を入力としてサポートします。

🚅: 追加の設定/操作なしで使用できるモデルです。

🎬: 入力としてビデオをサポート。

**Transformersバージョンの推奨事項:**

特定のtransformerバージョンで一部のVLMが実行できない可能性があることに注意してください。各VLMを評価するために、以下の設定を推奨します：

- **`transformers==4.33.0`を使用してください**: `Qwenシリーズ`, `Monkeyシリーズ`, `InternLM-XComposerシリーズ`, `mPLUG-Owl2`, `OpenFlamingo v2`, `IDEFICSシリーズ`, `VisualGLM`, `MMAlaya`, `ShareCaptioner`, `MiniGPT-4シリーズ`, `InstructBLIPシリーズ`, `PandaGPT`, `VXVERSE`, `GLM-4v-9B`.
- **`transformers==4.37.0`を使用してください**: `LLaVAシリーズ`, `ShareGPT4Vシリーズ`, `TransCore-M`, `LLaVA (XTuner)`, `CogVLMシリーズ`, `EMU2シリーズ`, `Yi-VLシリーズ`, `MiniCPM-[V1/V2]`, `OmniLMM-12B`, `DeepSeek-VLシリーズ`, `InternVLシリーズ`, `Cambrianシリーズ`, `VILA-VLシリーズ`.
- **`transformers==4.40.0`を使用してください**: `IDEFICS2`, `Bunny-Llama3`, `MiniCPM-Llama3-V2.5`, `360VL-70B`, `Phi-3-Vision`, `WeMM`.
- **`transformers==latest`を使用してください**: `LLaVA-Nextシリーズ`, `PaliGemma-3B`, `Chameleon-VLシリーズ`, `Video-LLaVA-7B-HF`, `Ovis1.5シリーズ`, `Mantisシリーズ`, `MiniCPM-V2.6`.

```python
# デモ
from vlmeval.config import supported_VLM
model = supported_VLM['idefics_9b_instruct']()
# 単一画像のフォワード
ret = model.generate(['assets/apple.jpg', 'この画像には何がありますか？'])
print(ret)  # この画像には葉がついた赤いリンゴがあります。
# 複数画像のフォワード
ret = model.generate(['assets/apple.jpg', 'assets/apple.jpg', '提供された画像にはリンゴが何個ありますか？'])
print(ret)  # 提供された画像にはリンゴが2個あります。
```

## 🏗️ クイックスタート

クイックスタートガイドについては、[クイックスタート](/docs/en/Quickstart.md)を参照してください。

## 🛠️ 開発ガイド

カスタムベンチマーク、VLMsを開発するか、単に**VLMEvalKit**に他のコードを貢献する場合は、[開発ガイド](/docs/en/Development.md)を参照してください。

コミュニティからの共有を奨励し、それに応じたクレジットを共有するために、次回のレポート更新では以下のことを実施します：

- 全ての貢献に対して感謝の意を示します
- 新しいモデル、評価セット、または主要な機能への3つ以上の主要な貢献を持つ貢献者は、テクニカルレポートの著者リストに加わることができます。適格な貢献者は、issueを作成するか、または[VLM評価キット ディスコードチャンネル](https://discord.com/invite/evDT4GZmxN)で kennyutc にDMを送ることができます。私たちはそれに応じてフォローアップします。

## 🎯 VLMEvalKitの目標

**このコードベースは以下を目的として設計されています：**

1. 研究者や開発者が既存のLVLMsを評価し、評価結果を**簡単に再現できるようにする**ための**使いやすい**、**オープンソースの評価ツールキット**を提供します。
2. VLMの開発者が自分のモデルを簡単に評価できるようにします。複数のサポートされているベンチマークでVLMを評価するには、単一の`generate_inner()`関数を**実装するだけで**、他のすべてのワークロード（データのダウンロード、データの前処理、予測の推論、メトリックの計算）はコードベースによって処理されます。

**このコードベースは以下を目的として設計されていません：**

1. すべての**第三者ベンチマーク**の元の論文で報告された正確な精度数値を再現すること。その理由は2つあります：
   1. VLMEvalKitは、すべてのVLMに対して**生成ベースの評価**を使用します（オプションで**LLMベースの回答抽出**を使用）。一方、一部のベンチマークは異なるアプローチを使用する場合があります（SEEDBenchはPPLベースの評価を使用します）。これらのベンチマークについては、対応する結果で両方のスコアを比較します。開発者には、コードベースで他の評価パラダイムをサポートすることをお勧めします。
   2. デフォルトでは、すべてのVLMに対して同じプロンプトテンプレートを使用してベンチマークを評価します。一方、**一部のVLMには特定のプロンプトテンプレートがある**場合があります（現時点ではコードベースでカバーされていない場合があります）。VLMの開発者には、現在カバーされていない場合でも、VLMEvalKitで独自のプロンプトテンプレートを実装することをお勧めします。これにより、再現性が向上します。

## 🖊️ 引用

この作業が役立つ場合は、このリポジトリに**スター🌟**を付けてください。サポートありがとうございます！

[![Stargazers repo roster for @open-compass/VLMEvalKit](https://reporoster.com/stars/open-compass/VLMEvalKit)](https://github.com/open-compass/VLMEvalKit/stargazers)

研究でVLMEvalKitを使用する場合、または公開されたオープンソースの評価結果を参照する場合は、以下のBibTeXエントリと、使用した特定のVLM/ベンチマークに対応するBibTexエントリを使用してください。

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

<p align="right"><a href="#top">🔝Top に戻る</a></p>

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
