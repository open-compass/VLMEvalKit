# Model Interfaces and Best Practices (API / Open-Source / Generation)

This document describes the minimal model interfaces required by VLMEvalKit, the unified input message format, how inference calls models, and best practices for adding new models (including `transformers` version considerations and common pitfalls).

## 1. Where models live in the framework

VLMEvalKit uses `vlmeval.config.build_model(model_name)` as the unified model construction entry (see `vlmeval/config/__init__.py`). Models are roughly grouped into:

- Open-source/local understanding models: `vlmeval/vlm/*`, typically subclass `vlmeval.vlm.base.BaseModel`
- API understanding models: `vlmeval/api/*`, typically subclass `vlmeval.api.base.BaseAPI` (`is_api=True`)
- Generation models (T2I / editing / etc.): `vlmeval/ulm/*` (plus some API generation models), typically subclass `vlmeval.ulm.base.BaseGenModel`, set `SUPPORT_GEN=True`, and use `EXPERTISE` to declare capability domains

Inference differentiates paths by model attributes:

- API models: `getattr(model, "is_api", False) == True` → concurrent API inference path
- Open-source models: otherwise → local loop inference path

## 2. Unified input: interleaved multimodal message format

VLMEvalKit abstracts model inputs as an “interleaved message list”, where each item is a dict:

```python
[
  {"type": "image", "value": "/abs/path/to/img.png"},
  {"type": "text", "value": "Question ..."},
]
```

Allowed `type`:

- Understanding models usually support: `text`, `image` (video models additionally support `video`)
- Generation models usually support: `text`, `image`

Allowed `value`:

- `text`: plain string
- `image`/`video`: usually a local file path; the framework tries to normalize URLs/relative paths into local paths when possible

### 2.1 Convenience input forms

For usability, `generate()` often also accepts other forms and normalizes them into `list[dict]`, such as:

```python
["assets/apple.jpg", "What is in this image?"]
```

or explicit dicts:

```python
[{"type": "image", "value": "assets/apple.jpg"},
 {"type": "text", "value": "What is in this image?"}]
```

API models commonly accept role-based chat structures as well (they are expanded and normalized):

```python
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": ["assets/apple.jpg", "Describe the image."]},
]
```

## 3. Open-source understanding models (BaseModel): required interfaces

The minimal requirement for open-source/local VLMs is implementing `generate_inner()`. The base class handles input normalization and common utilities.

### 3.1 Required: generate_inner

Implement:

```python
def generate_inner(self, message: list[dict], dataset: str | None = None) -> str:
    ...
```

Here `message` is already normalized to `list[dict(type,value)]`. Convert it to your backend’s input and return a string answer.

### 3.2 Recommended: INTERLEAVE

- `INTERLEAVE=True`: model supports interleaved image/text inputs (multi-image, mixed text)
- `INTERLEAVE=False`: if interleave is not supported, call `message_to_promptimg()` in `generate_inner()` to degrade to “first image + concatenated text”

### 3.3 Optional: use_custom_prompt / build_prompt

Inference prefers model-defined prompts when enabled:

- `use_custom_prompt(dataset_name) -> bool`: return True to enable model-defined prompts
- `build_prompt(line, dataset=dataset_name) -> message`: build message for a dataset row

`build_prompt()` is called only when `use_custom_prompt()` returns True; otherwise inference calls `dataset.build_prompt()`.

### 3.4 Optional: set_dump_image

If you use a custom prompt and need to materialize base64 images to disk inside `build_prompt()`, implement:

```python
def set_dump_image(self, dump_image_func):
    ...
```

Inference injects `dataset.dump_image` into the model, and the model can call `self.dump_image(...)` inside `build_prompt()`.

## 4. API models (BaseAPI): required interfaces

API models typically subclass `vlmeval.api.base.BaseAPI`. Key differences vs. local models:

- `is_api=True`: used by inference to select API concurrency path
- `generate()` handles retries/logging/input normalization, and calls your `generate_inner()`

### 4.1 Required: generate_inner (API version)

Implement:

```python
def generate_inner(self, inputs: list[dict], **kwargs) -> tuple[int, Any, Any]:
    # returns (ret_code, answer, log)
```

Where:

- `ret_code`: 0 means success (framework uses it to decide whether to retry)
- `answer`: typically a string; generation APIs may return `PIL.Image` or mixed outputs
- `log`: statistics such as tokens/latency/cost; if `keep_stats=True`, `generate()` returns `{"response": answer, "stats": log}`

### 4.2 kwargs and best practices

In API wrappers, it is recommended to:

- Read keys from environment variables (and support `.env`)
- Provide common knobs like `retry/timeout/verbose`
- Record useful stats in `log` (but never store sensitive secrets)

## 5. Generation models (BaseGenModel / SUPPORT_GEN)

Generation models are used for T2I / TI2I / TI2TI tasks. The entrypoint is `run_gen.py`, and inference is implemented in `vlmeval/inference_gen.py`.

### 5.1 Key fields: SUPPORT_GEN and EXPERTISE

Generation models usually need:

- `SUPPORT_GEN = True`
- `EXPERTISE`: list of capability domains, for example:
  - `T2I`: text-to-image
  - `TI2I`: image editing
  - `TI2TI`: image editing with text output (mixed tasks)

`vlmutil check` uses these fields to decide which minimal checks to run.

### 5.2 Output types

Generation `generate()` may return:

- `PIL.Image.Image`
- `str` (e.g., a description)
- `list[str|Image]` (mixed outputs)

Each dataset defines how these outputs are parsed in `evaluate()`.

## 6. How inference calls models (understanding path)

Understanding inference typically splits into:

- API models: build prompt then call `model.generate(message=..., dataset=...)` concurrently
- Open-source models: build prompts per sample, call `model.generate(...)` serially or by rank slicing

When `use_custom_prompt()` is implemented, inference calls `model.build_prompt()` first; otherwise it calls `dataset.build_prompt()`.

## 7. transformers version guidance

VLMEvalKit covers many open-source models; requirements differ across model families. Recommended practice:

- Pin a reproducible environment per model family (python/torch/cuda/transformers)
- Freeze dependencies with containers or conda locks in CI or team environments
- When adding a new model, implement and validate on the `transformers` version recommended by the model’s official repo

## 8. Checklist for adding a new model

1. Decide the model category: understanding VLM or generation ULM (or API)
2. Implement the minimal interface:
   - VLM: `generate_inner()`
   - API: `generate_inner()` returning `(ret_code, answer, log)`
   - ULM: `generate_inner()` and set `SUPPORT_GEN=True`
3. Define message support:
   - multi-image, interleave, video
   - degradation strategy if interleave is unsupported
4. If custom prompts are needed:
   - implement `use_custom_prompt()` + `build_prompt()`
5. Register into config:
   - add the class to `vlmeval/vlm/__init__.py` or `vlmeval/api/__init__.py` / `vlmeval/ulm/__init__.py`
   - add it into `supported_VLM / supported_ULM / supported_APIs` under `vlmeval/config/*`
6. Run minimal validation with `vlmutil check {MODEL_NAME}`
