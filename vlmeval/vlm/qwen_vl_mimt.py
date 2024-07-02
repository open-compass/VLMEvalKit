import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import copy as cp
from .base import BaseModel
from ..smp import isimg, listinstr
from ..dataset import DATASET_TYPE
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class QwenVLChat_mimt:

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='Qwen/Qwen-VL-Chat', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda', trust_remote_code=True).eval()
        torch.cuda.empty_cache()
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

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
                images = img_paths[pics_number : pics_number+tag_number]
                pics_number += tag_number
                for i, image in enumerate(images):
                    q = q.replace('<ImageHere>', '<img>'+image+'</img>', 1) 
                logging.info(pics_number)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    response, history = self.model.chat(self.tokenizer, query=q, history=history)
            responses.append(response)

        return responses
