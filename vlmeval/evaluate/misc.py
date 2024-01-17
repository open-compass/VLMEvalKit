import os
from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal

INTERNAL = os.environ.get('INTERNAL', 0)

def build_judge(version, **kwargs):
    model_map = {
        'gpt-4-turbo': 'gpt-4-1106-preview', 
        'gpt-4-0613': 'gpt-4-0613',
        'gpt-4-0314': 'gpt-4-0314',
        'chatgpt-1106': 'gpt-3.5-turbo-1106',
        'chatgpt-0613': 'gpt-3.5-turbo-0613'
    }
    model_version = model_map[version]
    if INTERNAL:
        model = OpenAIWrapperInternal(model_version, **kwargs)
    else:
        model = OpenAIWrapper(model_version, **kwargs)
    return model