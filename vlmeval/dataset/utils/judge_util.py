import os
from ...smp import load_env

INTERNAL = os.environ.get('INTERNAL', 0)


def build_judge(**kwargs):
    from ...api import OpenAIWrapper, SiliconFlowAPI, HFChatModel, XHSVLMAPIWrapper
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
            'xhs-deepseek': 'DeepSeek-V3',
            "gptoss-120b" : "gptoss-120b",
            'llama31-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        }
        model_version = model_map[model] if model in model_map else model
    else:
        model_version = LOCAL_LLM

    if model in ['qwen-7b', 'qwen-72b', 'deepseek']:
        model = SiliconFlowAPI(model_version, **kwargs)
    elif model == 'llama31-8b':
        model = HFChatModel(model_version, **kwargs)
    elif model == 'gpt-4o-mini':
        api_base = os.environ.get('XHS_OPENAI_GPT4OMINI_API_BASE', None)
        key = os.environ.get('XHS_OPENAI_GPT4OMINI_KEY', None)
        assert api_base is not None and key is not None, (
            "Please set `XHS_OPENAI_GPT4OMINI_API_BASE` and `XHS_OPENAI_GPT4OMINI_KEY` in '$VLMEVALKIT/.env'")
        model = OpenAIWrapper(
            model_version, 
            api_base=api_base, 
            key=key, 
            use_azure=True,
            wait=30,
            **kwargs)
    elif model == 'gpt-4o':
        api_base = os.environ.get('XHS_OPENAI_GPT4O_API_BASE', None)
        key = os.environ.get('XHS_OPENAI_GPT4O_KEY', None)
        assert api_base is not None and key is not None, (
            "Please set `XHS_OPENAI_GPT4O_API_BASE` and `XHS_OPENAI_GPT4O_KEY` in '$VLMEVALKIT/.env'")
        model = OpenAIWrapper(
            'gpt-4o', 
            api_base=api_base, 
            key=key, 
            use_azure=True,
            **kwargs)
    elif model == 'xhs-deepseek':
        api_base = os.environ.get('XHS_DEEPSEEK_API_BASE', None)
        key = os.environ.get('XHS_DEEPSEEK_KEY', None)
        assert api_base is not None and key is not None, (
            "Please set `XHS_DEEPSEEK_API_BASE` and `XHS_DEEPSEEK_KEY` in '$VLMEVALKIT/.env'")
        model = XHSVLMAPIWrapper(
            'deepseek-v3', 
            api_base=api_base, 
            key=key,
            **kwargs)
    elif model == 'gptoss-120b':
        print("Using self-hosted GPT-OSS-120B from local platform......")
        api_base = os.environ.get('XHS_GPT_OSS_120B_API_BASE', None)
        key = os.environ.get('XHS_GPT_OSS_120B_KEY', None)
        assert api_base is not None, (
            "Please set `XHS_GPT_OSS_120B_API_BASE` in '$VLMEVALKIT/.env'")
        if "," in api_base:
            api_base = api_base.split(",")
            for api in api_base:
                assert api.startswith('http'), (
                    "Please set `XHS_GPT_OSS_120B_API_BASE` in '$VLMEVALKIT/.env'")
        model = XHSVLMAPIWrapper(
            'openai/gpt-oss-120b', 
            api_base=api_base, 
            key=key,
            reasoning_effort = "high",
            timeout = 300,
            **kwargs)
    elif model == 'chatgpt-0125':
        if os.environ.get('MCQ_MATCH_MODEL_TYPE', "self-gptoss") == "chatgpt35":
            api_base = os.environ.get('XHS_OPENAI_GPT35_API_BASE', None)
            key = os.environ.get('XHS_OPENAI_GPT35_KEY', None)
            assert api_base is not None and key is not None, (
                "Please set `XHS_OPENAI_GPT35_API_BASE` and `XHS_OPENAI_GPT35_KEY` in '$VLMEVALKIT/.env'")
            model = OpenAIWrapper(
                model_version, 
                api_base=api_base, 
                key=key, 
                use_azure=True,
                wait=50,
                **kwargs)
        elif os.environ.get('MCQ_MATCH_MODEL_TYPE', "self-gptoss") == "gptoss":
            print("Using GPT-OSS-120B from ALLLINAI platform......")
            api_base = os.environ.get('ALLLINAI_API_BASE', None)
            key = os.environ.get('ALLINAI_KEY', None)
            assert api_base is not None and key is not None, (
                "Please set `ALLLINAI_API_BASE` and `ALLINAI_KEY` in '$VLMEVALKIT/.env'")
            model = XHSVLMAPIWrapper(
                'gpt-oss-120b', 
                api_base=api_base, 
                key=key,
                **kwargs)
        elif os.environ.get('MCQ_MATCH_MODEL_TYPE', "self-gptoss") == "qwen3-32b":
            print("Using Qwen3-32B from ALLLINAI platform......")
            api_base = os.environ.get('ALLLINAI_API_BASE', None)
            key = os.environ.get('ALLINAI_KEY', None)
            assert api_base is not None and key is not None, (
                "Please set `ALLLINAI_API_BASE` and `ALLINAI_KEY` in '$VLMEVALKIT/.env'")
            model = XHSVLMAPIWrapper(
                'qwen3-32b', 
                api_base=api_base, 
                key=key,
                **kwargs)
        elif os.environ.get('MCQ_MATCH_MODEL_TYPE', "self-gptoss") == "qwen3-30b-a3b":
            api_base = os.environ.get('ALLLINAI_API_BASE', None)
            key = os.environ.get('ALLINAI_KEY', None)
            assert api_base is not None and key is not None, (
                "Please set `ALLLINAI_API_BASE` and `ALLINAI_KEY` in '$VLMEVALKIT/.env'")
            model = XHSVLMAPIWrapper(
                'qwen3-30b-a3b', 
                api_base=api_base, 
                key=key,
                **kwargs)
        elif os.environ.get('MCQ_MATCH_MODEL_TYPE', "self-gptoss") == "self-gptoss":
            print("Using self-hosted GPT-OSS-120B from local platform......")
            api_base = os.environ.get('XHS_GPT_OSS_120B_API_BASE', None)
            key = os.environ.get('XHS_GPT_OSS_120B_KEY', None)
            assert api_base is not None, (
                "Please set `XHS_GPT_OSS_120B_API_BASE` in '$VLMEVALKIT/.env'")
            if "," in api_base:
                api_base = api_base.split(",")
                for api in api_base:
                    assert api.startswith('http'), (
                        "Please set `XHS_GPT_OSS_120B_API_BASE` in '$VLMEVALKIT/.env'")
            model = XHSVLMAPIWrapper(
                'openai/gpt-oss-120b', 
                api_base=api_base, 
                key=key,
                reasoning_effort = "high",
                **kwargs)
        else:
            raise ValueError(f"Invalid MCQ_MATCH_MODEL_TYPE: {os.environ.get('MCQ_MATCH_MODEL_TYPE', 'chatgpt35')}")
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
You cam see the specific error if the API call fails.
"""
