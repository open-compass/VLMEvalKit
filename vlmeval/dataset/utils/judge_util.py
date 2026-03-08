import os
from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)

DEFAULT_JUDGE = 'gpt-4o-mini-2024-07-18'

JudgeAbbr_JudgeName = {
    'gpt-4-turbo': 'gpt-4-1106-preview',
    'gpt-4-0125': 'gpt-4-0125-preview',
    'chatgpt-1106': 'gpt-3.5-turbo-1106',
    'chatgpt-0125': 'gpt-3.5-turbo-0125',
    'gpt-4o': 'gpt-4o-2024-05-13',
    'gpt-4o-0806': 'gpt-4o-2024-08-06',
    'gpt-4o-1120': 'gpt-4o-2024-11-20',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4.1': 'gpt-4.1-2025-04-14',
    'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
    'deepseek': 'deepseek-ai/DeepSeek-V3',
    'o3-mini': 'o3-mini-2025-01-31',
    'gpt-5': 'gpt-5-2025-08-07',
    'gpt-5.1': 'gpt-5.1-2025-11-13',
    'gpt-5.2': 'gpt-5.2-2025-12-11',
    'gpt-oss-120b': 'gpt-oss-120b',
    'qwen-max': 'qwen-max'
}

JudgeName_OpenRouterName = {
    'gpt-4-1106-preview': 'openai/gpt-4-1106-preview',
    'gpt-3.5-turbo-1106': 'openai/gpt-3.5-turbo-1106',
    'gpt-3.5-turbo-0125': 'openai/gpt-3.5-turbo-0125',
    'gpt-4o-2024-05-13': 'openai/gpt-4o-2024-05-13',
    'gpt-4o-2024-08-06': 'openai/gpt-4o-2024-08-06',
    'gpt-4o-2024-11-20': 'openai/gpt-4o-2024-11-20',
    'gpt-4o-mini-2024-07-18': 'openai/gpt-4o-mini-2024-07-18',
    'gpt-4.1-2025-04-14': 'openai/gpt-4.1',
    'gpt-4.1-mini-2025-04-14': 'openai/gpt-4.1-mini',
    'gpt-4.1-nano-2025-04-14': 'openai/gpt-4.1-nano',
    'o3-mini-2025-01-31': 'openai/o3-mini',
    'Qwen/Qwen2.5-7B-Instruct': 'qwen/qwen-2.5-7b-instruct',
    'Qwen/Qwen2.5-72B-Instruct': 'qwen/qwen-2.5-72b-instruct',
    'deepseek-ai/DeepSeek-V3': 'deepseek/deepseek-chat-v3-0324',
    'gpt-5-2025-08-07': 'openai/gpt-5'
}


def build_judge_default(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI
    model = kwargs.pop('model', None)
    if model == 'chatgpt-0125':
        model = 'gpt-4o-mini'
    kwargs.pop('nproc', None)
    load_env()
    model_version = JudgeAbbr_JudgeName[model] if model in JudgeAbbr_JudgeName else model

    if model in ['qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model


def build_judge_openrouter(**kwargs):
    from vlmeval.api.openrouter import OpenRouter
    model = kwargs.pop('model', None)
    if model == 'chatgpt-0125':
        model = 'gpt-4o-mini'
    kwargs.pop('nproc', None)
    load_env()

    model_version = JudgeAbbr_JudgeName[model] if model in JudgeAbbr_JudgeName else model
    if model_version in JudgeName_OpenRouterName:
        openrouter_name = JudgeName_OpenRouterName[model_version]
        model = OpenRouter(openrouter_name, **kwargs)
    else:
        model = None
    return model


def build_judge_given_router(router, **kwargs):
    if router == 'default':
        return build_judge_default(**kwargs)
    elif router == 'openrouter':
        return build_judge_openrouter(**kwargs)
    else:
        raise ValueError(f"Unknown judge router: {router}")


def build_judge_w_fallback(router, **kwargs):
    fallback_list = ['openrouter', 'default']
    assert router in fallback_list, (router, fallback_list)
    fallback_list.remove(router)
    fallback_list = [router] + fallback_list
    for router in fallback_list:
        model = build_judge_given_router(router=router, **kwargs)
        if model is not None and hasattr(model, 'working') and model.working():
            return model
    return None


def build_judge(**kwargs):
    from vlmeval.smp import get_logger
    judge_router = os.environ.get('JUDGE_ROUTER', 'default')
    logger = get_logger('Judge')
    logger.info(f'Building Judge with JUDGE_ROUTER: {judge_router}')
    # model = kwargs.get('model', None)
    return build_judge_w_fallback(router=judge_router, **kwargs)


DEBUG_MESSAGE = """
To debug the OpenAI API, you can try the following scripts in python:
```python
from vlmeval.api import OpenAIWrapper
model = OpenAIWrapper('gpt-4o', verbose=True)
msgs = [dict(type='text', value='Hello!')]
code, answer, resp = model.generate_inner(msgs)
print(code, answer, resp)
```
You cam see the specific error if the API call fails.
"""
