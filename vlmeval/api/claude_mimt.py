from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from time import sleep
import base64
import mimetypes
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from openai import OpenAI
import base64

def mllm_openai(query, images, model, conversation_history):
    # api_base = "http://23.224.95.56:3000/v1"
    api_base = "http://new-api.xxlab.tech/v1"
    api_key = os.environ.get('ALLES', '')
    client = OpenAI(api_key=api_key, base_url=api_base)

    base64_images = []
    for image_path in images:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
            base64_images.append(base64_image)

    # conversation_history.append({"role": "user", "content": query, "images_count": len(images)})

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
        model=model,
        messages=messages,
        max_tokens=4096,
    )

    assistant_response = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_response})
    return assistant_response, conversation_history


class Claude3V_mimt:
    is_api: bool = True
    def __init__(self,
                 model: str = 'claude-3-opus-20240229',
                 key: str = None,
                 retry: int = 10,
                 wait: int = 3,
                 system_prompt: str = None,
                 verbose: bool = True,
                 temperature: float = 0,
                 max_tokens: int = 1024,
                 **kwargs):

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('ALLES', '')

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
                    logging.info(pics_number)
                else:
                    images = []
                
                response, history = mllm_openai(query=q, images=images, model =self.model, conversation_history=history)
                responses.append(response)
        except Exception as e:
            print({e})
        return responses
