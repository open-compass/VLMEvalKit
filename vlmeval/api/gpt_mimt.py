from ..smp import *
import os
import sys
from .base import BaseAPI

APIBASES = {
    'OFFICIAL': 'https://api.openai.com/v1/chat/completions',
}


def GPT_context_window(model):
    length_map = {
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-turbo-preview': 128000,
        'gpt-4-1106-preview': 128000,
        'gpt-4-0125-preview': 128000,
        'gpt-4-vision-preview': 128000,
        'gpt-4-turbo': 128000,
        'gpt-4-turbo-2024-04-09': 128000,
        'gpt-3.5-turbo': 16385,
        'gpt-3.5-turbo-0125': 16385,
        'gpt-3.5-turbo-1106': 16385,
        'gpt-3.5-turbo-instruct': 4096,
    }
    if model in length_map:
        return length_map[model]
    else:
        return 128000

from openai import OpenAI
import base64
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'  # 重置颜色
def mllm_openai(query, images, model, conversation_history):
    api_key=os.environ.get('OPENAI_API_KEY', None)
    client = OpenAI(api_key=api_key)

    base64_images = []
    for image_path in images:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            base64_images.append(base64_image)

    # 创建一个modified_conversation_history的副本,用于传递给API,只取role和content字段
    modified_conversation_history = [
        {"role": message["role"], "content": message["content"]}
        for message in conversation_history
    ]

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages.extend(modified_conversation_history)

    if len(images)!=0:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images],
            ],
        })
    
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                *[{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}} for base64_image in base64_images],
            ],
        })
    else:
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})
        conversation_history.append({"role": "user", "content": [{"type": "text", "text": query}]})
        
    response = client.chat.completions.create(
        model= model,
        messages=messages,
        max_tokens=4096,
    )

    assistant_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response, conversation_history


class GPT4V_mimt:
    is_api: bool = True
    def __init__(self,
                 model: str = 'gpt-3.5-turbo-0613',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 api_base: str = None,
                 max_tokens: int = 1024,
                 img_size: int = 512,
                 img_detail: str = 'low',
                 use_azure: bool = False,
                 ):

        self.wait = wait
        self.retry = retry
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, message, dataset=None):
        # return ['a1','a2','a3']
        img_paths = []
        questions = []
        for item in message:
            if item['type'] == 'image':
                img_paths.append(item['value'])
            elif item['type'] == 'text':
                questions.append(item['value'])
        questions = eval(questions[0])

        # return questions

        responses = []
        try:
            pics_number = 0
            history = []
            for index, q in enumerate(questions):
                if "<ImageHere>" in q:
                    tag_number = q.count('<ImageHere>')
                    if tag_number ==1:
                        pics_number += 1
                        images = [img_paths[pics_number-1]]
                    else:
                        pics_number_end = pics_number+tag_number
                        images = img_paths[pics_number: pics_number_end]
                        pics_number += tag_number
                    print(pics_number)
                else:
                    images = []
                
                response, history = mllm_openai(query=q, images=images, model =self.model, conversation_history=history)
                responses.append(response)
        except Exception as e:
            print({e})
        return responses