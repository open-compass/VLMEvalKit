# 交互式多轮评测：设计与集成

本文介绍如何将多轮 benchmark 接入 VLMEvalKit，以及运行器、数据集和策略各自负责的部分。现有实现包括图像数据集 MMDU、VisChainBench 和视频数据集 MT-Video-Bench。

## 设计

多轮（MT）描述的是**一个样本如何推理**，与图像、视频等模态和数据集的任务类型是独立的。数据集通过继承 `InteractiveMTMixin` 获得 `interactive_mt = True`；运行器依据该标志选择多轮流程，不依赖 `TYPE == 'MT'`。`TYPE` 仍应填写数据集本身的任务类型，例如 `VQA`。

一行数据对应一个独立的 **case**，即所有模型调用之间可能存在依赖的最大单元。不同 case 可以并行，同一 case 的调用顺序由策略决定。对于同一原始题目下互相独立的条件，应在加载数据时拆为不同的行或数据集名称，并在评分时按需汇总。

```text
run.py 构造 dataset / model / 可选 policy LLM
    └─ 按模态选择已有推理路径；按 interactive_mt 选择 case 执行方式
       └─ 构造首轮消息 → chat_interactive_mt → dataset.run_case
          └─ build_policy → ModelCaller.chat → policy.step → ... → finalize
             └─ 保存每个 case 的 prediction → dataset.evaluate
```

职责划分如下：

- **数据集**负责首轮消息、为每个 case 创建策略，以及解析多轮 prediction。
- **`TurnPolicy`** 保存单个 case 的状态，根据回答决定下一轮，最后生成该 case 的 prediction。不要在 dataset 或共享模型对象中保存 case 的可变状态。
- **默认 case 循环**管理模型调用、对话历史、终止和失败处理。它通过 `ModelCaller` 调用被测模型的 `chat`；策略需要的辅助 LLM 与被测模型及评分 judge 是独立的。
- **运行器**按 case 调度；`--api-mode` 的并发针对 case，case 内有依赖的轮次按策略顺序执行。

## 数据集接口

1. 将 `InteractiveMTMixin` 放在现有模态基类之前，例如 `class MyDataset(InteractiveMTMixin, ImageBaseDataset)`；视频使用 `VideoBaseDataset`。数据准备沿用对应模态基类的约定。
2. 覆盖 `build_prompt`，返回**第一次模型调用的完整上下文**，而不是所有未来轮次。返回值是带角色的消息列表，最后一条必须是 `user`；每条消息的 `content` 包含有序的文本和媒体项：

   ```python
   [
       {'role': 'system', 'content': [{'type': 'text', 'value': 'Follow the instructions.'}]},
       {'role': 'user', 'content': [
           {'type': 'image', 'value': '/path/to/image.jpg'},
           {'type': 'text', 'value': 'What is shown?'},
       ]},
   ]
   ```

   视频数据集的 `build_prompt` 还要接受 `video_llm`。首轮没有合法的角色消息或没有以 user 结尾会被默认循环拒绝。
3. 实现 `build_policy(self, case, messages, *, policy_llm=None)`，为每个 case 返回新的策略。若没有辅助 LLM，也可以省略 `policy_llm` 参数：默认循环仅在其非空时传入该关键字。`messages` 是已构造的首轮上下文。
4. 实现 `evaluate`，读取每行的 `prediction`，按官方协议解析、评分和汇总。明确设计序列化格式；保存到 TSV 后，列表或字典可能需要按所用格式反序列化。不要假设所有 benchmark 返回相同的 prediction 结构。

## 编写策略

固定的后续 user 轮次可直接使用 `StaticTurnPolicy`。下面是接口示意，`questions` 的读取与图片路径需按实际数据集实现；加载、注册和评分代码未列出。

```python
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.interactive_mt import InteractiveMTMixin
from vlmeval.interactive_mt import StaticTurnPolicy


class ExampleMT(InteractiveMTMixin, ImageBaseDataset):
    TYPE = 'VQA'

    def build_prompt(self, case):
        questions = self.questions_of(case)  # 非空的题目字符串列表
        image_path = self.image_path_of(case)  # 数据集负责准备图片
        return [{'role': 'user', 'content': [
            {'type': 'image', 'value': image_path},
            {'type': 'text', 'value': questions[0]},
        ]}]

    def build_policy(self, case, messages):
        followups = [
            {'role': 'user', 'content': [{'type': 'text', 'value': question}]}
            for question in self.questions_of(case)[1:]
        ]
        return StaticTurnPolicy(messages, followups)
```

`StaticTurnPolicy` 的 `finalize()` 返回回答列表，包括失败时记录的失败消息。可参考 `vlmeval/dataset/image_mt.py` 中完整的 MMDU 实现。

如果下一题依赖回答，实现 `TurnPolicy`：初始化时调用 `super().__init__()` 并设置正整数 `max_turns`；`step(assistant_response, history)` 记录回答并返回 `Step`；`finalize()` 返回 evaluator 所需的结果。`history` 已包含本轮 assistant 回答。可用的 `Step` 字段：

| 字段 | 行为 |
| --- | --- |
| `messages` | 追加到历史的 user 消息列表；继续时必须提供消息或 `history`。 |
| `history` | 非 `None` 时替换下一次调用的完整历史，可用于参考答案历史或上下文重置；下一次调用的最终消息应为 user。 |
| `chat_kwargs` | 仅传给下一次模型调用的参数；须确认对应模型的 `chat` 接口接受它们。 |
| `num_samples` | 大于 1 时从同一前缀多次调用模型，并将结果交给策略的 `on_samples`；默认实现不支持多候选。 |
| `done` | 为真时结束 case。 |

模型异常和已识别的失败消息会调用 `on_failure(..., reason='model')`；`step` / `on_samples` 的异常使用 `reason='policy'`，随后调用 `finalize()`。失败信息应保留在可识别的 prediction 中，以便断点续跑和评分区分失败与正常答案。需要重写失败输出时可覆盖 `on_failure`。消息格式、非法的下一步和首轮参数错误也可能直接抛出异常。超过 `max_turns` 后循环结束并调用 `finalize()`。

复杂依赖图或需要自定义调用顺序时，可以覆盖 `run_case(case, messages, caller, policy_llm=None)`；使用 `caller.chat` / `caller.chat_n` 访问被测模型，以复用统一失败检测。参考 `vischainbench.py` 的动态策略和 `mt_video_bench.py` 的历史替换。

## 辅助 LLM、运行与验证

仅当推理过程中的策略确实需要辅助 LLM 时，才覆盖 `build_policy_llm(policy_kwargs)`。运行器每个数据集构造一次，并将返回对象原样共享给各 case 和 API worker；它必须支持并发，不能持有 case 状态。`--policy`、`--policy-args`（JSON 对象）、`--policy-base-url`、`--policy-key`、`--policy-retry` 和 `--policy-timeout` 专门配置该对象；`--policy-base-url` 会被转换为 `api_base`，并追加 `/chat/completions`。未指定 `--policy-retry` 时沿用 `--retry`，未指定模型时调用数据集的 `get_default_policy_model`。`--policy-args` 中的角色约定由数据集实现，不由框架自动创建多个客户端。评分阶段仍使用独立的 judge 配置。不要把密钥写入代码或文档。

多轮模型必须支持 `chat` 和所需媒体、角色及历史。基础模型的 `chat` 可能在异常时逐步丢弃历史重试，接入时应验证目标模型与 benchmark 的历史要求相符。`--n-runs > 1` 当前不支持交互式多轮数据集。

接入本地模型时，可在模型类实现 `chat_inner(messages, dataset=None)`，将完整角色历史和每轮媒体转换为该模型的输入并返回文本；API 模型按其 `BaseAPI` 约定实现或适配 `chat_inner`。具体可参考 `vlmeval/vlm/qwen3_vl/model.py`，并检查所用 API wrapper 是否保留 benchmark 所需的 system、历史及多模态内容。

接入时至少验证：

- 单个 case 的首轮消息、媒体顺序、各轮历史和最终 prediction；动态分支、提前结束、失败与恢复；
- 多个 case 的状态隔离，以及多轮 prediction 的序列化和评分；
- 与官方实现使用相同输入时的消息、轮次及指标，尤其是参考答案历史、自预测历史和评分所需格式。

相关回归测试可参考 `tests/test_interactive_mt.py`、`tests/test_checkpoint_retry.py` 和 `tests/test_policy_config.py`。
