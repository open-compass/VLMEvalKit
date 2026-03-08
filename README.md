# VLMEvalKit-Lite

## 文档 Documentation

- 中文文档入口：[/docs/zh-CN/README.md](/docs/zh-CN/README.md)
- English docs entry: [/docs/en/README.md](/docs/en/README.md)
- 快速开始（中文 / English）：[/docs/zh-CN/Quickstart.md](/docs/zh-CN/Quickstart.md) / [/docs/en/Quickstart.md](/docs/en/Quickstart.md)
- 工作流与产物（中文 / English）：[/docs/zh-CN/Workflow.md](/docs/zh-CN/Workflow.md) / [/docs/en/Workflow.md](/docs/en/Workflow.md)
- 模型接口（中文 / English）：[/docs/zh-CN/Model.md](/docs/zh-CN/Model.md) / [/docs/en/Model.md](/docs/en/Model.md)
- 数据集规范（中文 / English）：[/docs/zh-CN/Dataset.md](/docs/zh-CN/Dataset.md) / [/docs/en/Dataset.md](/docs/en/Dataset.md)
- CLI 工具（中文 / English）：[/docs/zh-CN/Tools.md](/docs/zh-CN/Tools.md) / [/docs/en/Tools.md](/docs/en/Tools.md)
- 配置系统（中文 / English）：[/docs/zh-CN/ConfigSystem.md](/docs/zh-CN/ConfigSystem.md) / [/docs/en/ConfigSystem.md](/docs/en/ConfigSystem.md)
- 环境建议（中文 / English）：[/docs/zh-CN/Environment.md](/docs/zh-CN/Environment.md) / [/docs/en/Environment.md](/docs/en/Environment.md)
- 常见问题（中文 / English）：[/docs/zh-CN/Troubleshooting.md](/docs/zh-CN/Troubleshooting.md) / [/docs/en/Troubleshooting.md](/docs/en/Troubleshooting.md)

## Installation

1. 推荐在虚拟环境中安装 VLMEvalKit，例如使用 conda / venv 创建虚拟环境 (推荐基于 python 3.10+)，创建好环境后，请遵循以下安装步骤：
  - 由于不同开源模型常使用不同的 torch 与 transformers 版本，为了避免冲突，建议在安装 VLMEvalKit 前先安装好对应版本的 torch 与 transformers。如对 torch，transformers 版本无特殊要求，可直接安装任一较新版本 (推荐 torch 2.4+)
  - 安装依赖 `pip install -r requirements.txt`，其中包含一些内场依赖
  - 安装 VLMEvalKit `pip install -e .`
