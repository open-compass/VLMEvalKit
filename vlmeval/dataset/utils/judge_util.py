import os
import warnings

from vlmeval.smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def warn_if_judge_model_changed(model, legacy_models, benchmark):
    """Warn when evaluation uses a judge outside the benchmark's legacy set."""
    if isinstance(legacy_models, str):
        legacy_models = (legacy_models, )
    else:
        legacy_models = tuple(legacy_models)

    if model in legacy_models:
        return

    legacy_model_names = ', '.join(repr(name) for name in legacy_models)
    warnings.warn(
        f'Judge model {model!r} differs from the legacy judge model(s) used for '
        f'{benchmark}: {legacy_model_names}. Using a different or newer judge model '
        'may change evaluation scores.',
        UserWarning,
        stacklevel=2,
    )


def build_judge(**kwargs):
    from vlmeval.api import HFChatModel, OpenAIWrapper, SiliconFlowAPI
    model = kwargs.pop('model', None)
    kwargs.pop('nproc', None)
    load_env()
    LOCAL_LLM = os.environ.get('LOCAL_LLM', None)
    if LOCAL_LLM is None:
        model_map = {
            'gpt-4-turbo': 'gpt-4-1106-preview',
            'gpt-4-0613': 'gpt-4-0613',
            'gpt-4-0125': 'gpt-4-0125-preview',
            'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
            'chatgpt-1106': 'gpt-3.5-turbo-1106',
            'chatgpt-0125': 'gpt-3.5-turbo-0125',
            'gpt-4o': 'gpt-4o-2024-05-13',
            'gpt-4o-0806': 'gpt-4o-2024-08-06',
            'gpt-4o-1120': 'gpt-4o-2024-11-20',
            'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
            'qwen-7b': 'Qwen/Qwen2.5-7B-Instruct',
            'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
            'deepseek': 'deepseek-ai/DeepSeek-V3',
            'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        }
        model_version = model_map[model] if model in model_map else model
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    elif model == 'llama31-8b':
        model = HFChatModel(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You can see the specific error if the API call fails.
"""
