import torch
import re
from vlmeval.api import OpenAIWrapper


# remap the gpt model name
gpt_version_map = {
    'gpt-4-0409': 'gpt-4-turbo-2024-04-09',
    'gpt-4-0125': 'gpt-4-0125-preview',
    'gpt-4-turbo': 'gpt-4-1106-preview',
    'gpt-4-0613': 'gpt-4-0613',
    'chatgpt-1106': 'gpt-3.5-turbo-1106',
    'chatgpt-0613': 'gpt-3.5-turbo-0613',
    'chatgpt-0125': 'gpt-3.5-turbo-0125',
    'gpt-4o': 'gpt-4o-2024-05-13'
}

# # map the model name to the api type
# reasoning_mapping = {
#     'llama3-70b-chat':'vllm',
#     'Mixtral-8x22B-chat':'vllm',
#     'deepseek-chat':'deepseek',
# }
#
# # stop_tokens for deploying vllm
# stop_tokens = {
#     'llama3-70b-chat': ["<|eot_id|>"],
# }

mapping = {}
mapping.update(gpt_version_map)

#mapping.update(reasoning_mapping)



class Prism():

    def __init__(self, supported_VLM, **kwargs):
        self.supported_VLM = supported_VLM
        self.config = kwargs

        self.model_name_fronted = self.config['model']['fronted']['model']
        self.model_name_backend = self.config['model']['backend']['model']

        self.model_fronted = supported_VLM[self.model_name_fronted]() if isinstance(self.model_name_fronted, str) else None
        self.model_backend = Reasoning(model=self.model_name_backend)


    def set_dump_image(self, dump_image):
        pass

    def generate(self, message, dataset=None):

        # struct prompt
        question = message[1]['value']

        #generate description
        is_api = getattr(self.model_fronted, 'is_api', False)
        if is_api:
            response = self.fronted_api()
        else:
            response = self.model_fronted.generate(message=message, dataset=dataset)

        print(response)



        breakpoint()
        pass

    def fronted_api(self):

        pass

    def use_custom_prompt(self, dataset):

        pass

    def build_prompt(self, line, dataset):

        pass


class Reasoning:
    def __init__(self, model):
        self.model = LLMWrapper(model)

    def generate(self, des, question):
        prompt = build_infer_prompt_external(question, des)
        return self.model.generate(prompt)


def build_infer_prompt_external(question, des):
    if not question.endswith('\n'):
        question += '\n'
    if not question.lower().startswith('question:') and not question.lower().startswith('hint:'):
        question = 'Question: ' + question
    if not des.endswith('\n'):
        des += '\n'
    description = 'Description: ' + des
    role = 'You are an excellent text-based reasoning expert. You are required to answer the question based on the detailed description of the image.\n\n'

    prompt = role + description + question
    return prompt


class LLMWrapper:

    def __init__(self, model_name, max_tokens=512, verbose=True, retry=5):

        # api bases, openai default
        # self.deepseek_api_base = 'https://api.deepseek.com/v1/chat/completions'

        # server settings of vllm
        # self.PORT = 8080
        # self.vllm_api_base = f'http://localhost:{self.PORT}/v1/chat/completions'


        if model_name.endswith('-2048'):
            model_name = model_name.replace('-2048', '')
            max_tokens = 2048

        if model_name in gpt_version_map:
            gpt_version = gpt_version_map[model_name]
            model = OpenAIWrapper(gpt_version, max_tokens=max_tokens, verbose=verbose, retry=retry)
        # elif reasoning_mapping[model_name] == 'vllm':
        #     model = OpenAIWrapper(model_name, api_base=self.vllm_api_base, max_tokens=max_tokens,
        #                           system_prompt='You are a helpful assistant.', verbose=verbose, retry=retry,
        #                           stop=stop_tokens[model_name])
        # elif reasoning_mapping[model_name] == 'deepseek':
        #     deepseek_key = os.environ['DEEPSEEK_API_KEY']
        #     model = OpenAIWrapper(model_name, api_base=self.deepseek_api_base, key=deepseek_key, max_tokens=max_tokens,
        #                           system_prompt='You are a helpful assistant.', verbose=verbose, retry=retry)
        else:
            print('Unknown API model for inference')

        self.model = model

    def generate(self, prompt, **kwargs):
        response = self.model.generate(prompt, **kwargs)
        return response

    @staticmethod
    def api_models():
        gpt_models = list(gpt_version_map.keys())
        api_models = gpt_models.copy()
        #api_models.extend(list(reasoning_mapping.keys()))
        return api_models