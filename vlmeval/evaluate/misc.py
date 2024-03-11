import os
from vlmeval.api import OpenAIWrapper, OpenAIWrapperInternal
from vlmeval.llm import model_map as hf_model_map
from vlmeval.llm import HFChatModel

INTERNAL = os.environ.get('INTERNAL', 0)

def build_judge(version, **kwargs):
    model_map = {
        'gpt-4-turbo': 'gpt-4-1106-preview', 
        'gpt-4-0613': 'gpt-4-0613',
        'gpt-4-0314': 'gpt-4-0314',
        'gpt-4-0125': 'gpt-4-0125-preview', 
        'chatgpt-1106': 'gpt-3.5-turbo-1106',
        'chatgpt-0613': 'gpt-3.5-turbo-0613',
        'chatgpt-0125': 'gpt-3.5-turbo-0125'
    }
    if version in model_map:
        model_version = model_map[version]
        if INTERNAL:
            model = OpenAIWrapperInternal(model_version, **kwargs)
        else:
            model = OpenAIWrapper(model_version, **kwargs)
    elif version in hf_model_map:
        model_name = hf_model_map[version]
        model = HFChatModel(model_name, temperature=0)
    else:
        model = None
    return model