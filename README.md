# VLMEvalKit (lite)

> This is a fork of [VLMEvalKit](https://github.com/OpenCompass/VLMEvalKit) with additional features, maintained by the OpenSource community.

![LOGO](https://opencompass.openxlab.space/utils/MMLB.jpg)

<b>A Toolkit for Evaluating Large Vision-Language Models. </b>

**VLMEvalKit** (the python package name is **vlmeval**) is an **open-source evaluation toolkit** of **large vision-language models (LVLMs)**. It enables **one-command evaluation** of LVLMs on various benchmarks, without the heavy workload of data preparation under multiple repositories. In VLMEvalKit, we adopt **generation-based evaluation** for all LVLMs, and provide the evaluation results obtained with both **exact matching** and **LLM-based answer extraction**.

## Installation

 We recommend installing VLMEvalKit in a virtual environment, such as using conda / venv to create a virtual environment (recommended to be based on python 3.10+). After creating the environment, please follow the installation steps below:
1. Since different open-source models often use different versions of torch and transformers, to avoid conflicts, it is recommended to install the corresponding versions of torch and transformers before installing VLMEvalKit. If there are no special requirements for torch and transformers versions, you can directly install any newer version (recommended torch 2.4+)
2. Install dependencies `pip install -r requirements.txt`
3. Install VLMEvalKit `pip install -e .`

## Documentation

- 中文文档：[/docs/zh-CN/README.md](/docs/zh-CN/README.md)
- English Docs：[/docs/en/README.md](/docs/en/README.md)
- Quickstart（中文 / English）：[/docs/zh-CN/Quickstart.md](/docs/zh-CN/Quickstart.md) / [/docs/en/Quickstart.md](/docs/en/Quickstart.md)
- Workflow（中文 / English）：[/docs/zh-CN/Workflow.md](/docs/zh-CN/Workflow.md) / [/docs/en/Workflow.md](/docs/en/Workflow.md)
- Model Interfaces（中文 / English）：[/docs/zh-CN/Model.md](/docs/zh-CN/Model.md) / [/docs/en/Model.md](/docs/en/Model.md)
- Dataset Conventions（中文 / English）：[/docs/zh-CN/Dataset.md](/docs/zh-CN/Dataset.md) / [/docs/en/Dataset.md](/docs/en/Dataset.md)
- CLI Tools（中文 / English）：[/docs/zh-CN/Tools.md](/docs/zh-CN/Tools.md) / [/docs/en/Tools.md](/docs/en/Tools.md)
- Config System（中文 / English）：[/docs/zh-CN/ConfigSystem.md](/docs/zh-CN/ConfigSystem.md) / [/docs/en/ConfigSystem.md](/docs/en/ConfigSystem.md)
- Environment（中文 / English）：[/docs/zh-CN/Environment.md](/docs/zh-CN/Environment.md) / [/docs/en/Environment.md](/docs/en/Environment.md)
- Trouble Shooting（中文 / English）：[/docs/zh-CN/Troubleshooting.md](/docs/zh-CN/Troubleshooting.md) / [/docs/en/Troubleshooting.md](/docs/en/Troubleshooting.md)

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
