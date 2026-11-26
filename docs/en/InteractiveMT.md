# Interactive multi-turn evaluation: design and integration

This guide describes the interactive multi-turn (MT) runner and how to integrate a benchmark. For existing implementations, see MMDU, VisChainBench, and MT-Video-Bench.

## Design

Multi-turn describes **how a sample is evaluated**, independently of its modality and task type. `InteractiveMTMixin` sets `interactive_mt = True`; runners use that flag, not `TYPE == 'MT'`, to select the interactive path. Keep `TYPE` as the task type, such as `VQA`.

One dataset row is one independent **case**: the largest unit whose model calls may depend on one another. Cases may run concurrently; a case's policy controls its call order. Expand independent conditions for the same original item into separate rows or dataset names and regroup them during evaluation.

```text
run.py creates dataset / model / optional policy LLM
    └─ dispatch by modality and interactive_mt
       └─ first prompt → chat_interactive_mt → dataset.run_case
          └─ build_policy → ModelCaller.chat → policy.step → ... → finalize
             └─ save prediction for each case → dataset.evaluate
```

The dataset owns the first prompt, per-case policy construction, and parsing of multi-turn predictions. A `TurnPolicy` owns mutable state for one case and decides what happens after each response. The default case loop manages model calls, history, stopping, and failure handling. In `--api-mode`, concurrency is across cases; dependent calls within a case follow the policy's order.

## Dataset contract

1. Put `InteractiveMTMixin` before the modality base class, for example `class MyDataset(InteractiveMTMixin, ImageBaseDataset)`; use `VideoBaseDataset` for video. Follow the modality base class's data preparation conventions.
2. Override `build_prompt` to return the **complete context for the first model call**, not all future turns. It must be a list of role messages ending in a user message. Each content list holds ordered text and media items:

   ```python
   [
       {'role': 'system', 'content': [{'type': 'text', 'value': 'Follow the instructions.'}]},
       {'role': 'user', 'content': [
           {'type': 'image', 'value': '/path/to/image.jpg'},
           {'type': 'text', 'value': 'What is shown?'},
       ]},
   ]
   ```

   A video dataset's `build_prompt` also accepts `video_llm`. The default loop rejects a first prompt without a final user message.
3. Implement `build_policy(self, case, messages, *, policy_llm=None)` and return a fresh policy per case. If no policy LLM is used, the `policy_llm` parameter may be omitted: the default loop passes that keyword only when its value is non-`None`.
4. Implement `evaluate` to read each row's `prediction` and apply the benchmark's parsing and metrics. Define the serialization explicitly: lists or dictionaries stored in TSV may need deserialization, and prediction formats differ across benchmarks.

## Policies

For fixed follow-up user turns, use `StaticTurnPolicy`. This interface sketch omits loading, registration, scoring, and the dataset-specific question and image helpers:

```python
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.interactive_mt import InteractiveMTMixin
from vlmeval.interactive_mt import StaticTurnPolicy


class ExampleMT(InteractiveMTMixin, ImageBaseDataset):
    TYPE = 'VQA'

    def build_prompt(self, case):
        questions = self.questions_of(case)  # nonempty list of strings
        image_path = self.image_path_of(case)  # prepared by the dataset
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

`StaticTurnPolicy.finalize()` returns a list of responses, including a failure message in the failed turn. See `vlmeval/dataset/image_mt.py` for the complete MMDU implementation.

For answer-dependent behavior, subclass `TurnPolicy`: call `super().__init__()`, set a positive `max_turns`, implement `step(assistant_response, history)` and return a `Step`, and implement `finalize()` in the format the evaluator expects. `history` already includes the current assistant response.

| `Step` field | Behavior |
| --- | --- |
| `messages` | User messages appended to history; to continue, provide messages or `history`. |
| `history` | If non-`None`, replaces the complete history for the next call; the resulting call should end in a user message. |
| `chat_kwargs` | Arguments for the next model call only; verify that the model's `chat` accepts them. |
| `num_samples` | For values greater than one, sample repeatedly from the same prefix and pass candidates to `on_samples`; its default implementation does not support this. |
| `done` | Stop the case when true. |

Model exceptions and recognized failure messages call `on_failure(..., reason='model')`; exceptions from `step` or `on_samples` use `reason='policy'`, after which `finalize()` runs. Preserve recognizable failure information in the prediction for retry and scoring. A policy may override `on_failure`. Invalid messages, initial input, or continuation steps can also raise directly. On reaching `max_turns`, the loop stops and calls `finalize()`.

For dependency graphs or custom call order, override `run_case(case, messages, caller, policy_llm=None)` and use `caller.chat` / `caller.chat_n` for the evaluated model so calls retain common failure handling. See `vischainbench.py` for dynamic turns and `mt_video_bench.py` for history replacement.

## Policy LLM and validation

Override `build_policy_llm(policy_kwargs)` only if the inference policy needs an auxiliary LLM. The runner creates its return value once per dataset and shares it across cases and API workers, so it must be safe for concurrent use and must not contain case state. `--policy`, `--policy-args` (a JSON object), `--policy-base-url`, `--policy-key`, `--policy-retry`, and `--policy-timeout` configure it independently of the evaluated model and judge. The runner turns `--policy-base-url` into `api_base` by appending `/chat/completions`; policy retry defaults to `--retry`, and an unspecified model comes from `get_default_policy_model`. Role-specific conventions in `--policy-args` belong to the dataset; the framework does not automatically create role clients. Keep credentials out of code and documentation.

The model must support `chat` and the required roles, media, and history. The base model `chat` may retry after dropping earlier history on exceptions, so verify that behavior against the benchmark protocol. `--n-runs > 1` is currently unsupported for interactive MT datasets.

For a local model, implement `chat_inner(messages, dataset=None)` to convert the complete role history and each turn's media into model inputs and return text. API models implement or adapt `chat_inner` according to their `BaseAPI` contract. See `vlmeval/vlm/qwen3_vl/model.py` and check that the selected API wrapper retains the system message, history, and multimodal content required by the benchmark.

At minimum, verify the first prompt and media ordering, every turn's history and final prediction, dynamic branches, early stopping, and failures. Check case state isolation and multi-turn prediction serialization and scoring. Compare runtime messages and metrics with the official implementation, especially for reference versus predicted history. See `tests/test_interactive_mt.py`, `tests/test_checkpoint_retry.py`, and `tests/test_policy_config.py`.
