import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

def img2base64(file_name):
    with open(file_name, 'rb') as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string

def chat_MiniCPM(query, image_list, model, tokenizer, history):
    images = [Image.open(img).convert('RGB') for img in image_list]

    history.append({'role': 'user', 'content': [query, *images]})
    
    response = model.chat(
        image=None,
        msgs=history,
        context=None,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7
    )
    history.append({'role': 'assistant', 'content': [response]})

    return response

class MiniCPM_Llama3_V_mimt:
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='openbmb/MiniCPM-Llama3-V-2_5', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        print(f'load from {self.model_path}')
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval().cuda()
        self.kwargs = kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        self.num_beams = 1 if self.model_path == 'openbmb/MiniCPM-V' else 3
        self.options_system_prompt = ('Carefully read the following question and select the letter corresponding '
                                      'to the correct answer. Highlight the applicable choices without giving '
                                      'explanations.')
        self.wo_options_system_prompt = 'Carefully read the following question Answer the question directly.'
        self.detail_system_prompt = 'Answer this question in detail.'
        self.vqa_prompt = 'Answer the question using a single word or phrase.'
    
    def generate(self, message, dataset=None):

        img_paths = []
        questions = []
        for item in message:
            if item['type'] == 'image':
                img_paths.append(item['value'])
            elif item['type'] == 'text':
                questions.append(item['value'])
        questions = eval(questions[0])

        responses = []
        pics_number = 0
        history = []
        for index, q in enumerate(questions):
            if "<ImageHere>" in q:
                tag_number = q.count('<ImageHere>')
                images = img_paths[pics_number:pics_number+tag_number]
                pics_number += tag_number
            else:
                images = []
            logging.info(pics_number)

            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        response = chat_MiniCPM(q, images, self.model, self.tokenizer, history=history)
                responses.append(response)
            except Exception as e:
                logging.info({e})
                responses.append(None)
        return responses

