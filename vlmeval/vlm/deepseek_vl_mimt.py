import sys
import torch
from transformers import AutoModelForCausalLM
import warnings
from .base import BaseModel
import deepseek_vl
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_response_concat(model, question, image_path_list, history, max_new_tokens=2048):
    messages = history.copy()
    # print(messages)
    conversation = [
        {
            "role": "User",
            "content": question,
            "images": image_path_list,
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]
    messages.extend(conversation)
    # print(messages)
    
    # load images and prepare for inputs
    pil_images = load_pil_images(messages)
    prepare_inputs = model.vl_chat_processor(
        conversations=messages,
        images=pil_images,
        force_batchify=True
    ).to(model.device)

    # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=model.tokenizer.eos_token_id,
        bos_token_id=model.tokenizer.bos_token_id,
        eos_token_id=model.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True
    )

    response = model.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    new_conversation = [
        {
            "role": "User",
            "content": question,
            "images": image_path_list,
        },
        {
            "role": "Assistant",
            "content": response
        }
    ]
    history.extend(new_conversation)
    # print(history)
    return response, history

class DeepSeekVL_mimt:

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import deepseek_vl
        except ImportError:
            warnings.warn(
                'Please first install deepseek_vl from source codes in: https://github.com/deepseek-ai/DeepSeek-VL')
            sys.exit(-1)

    def __init__(self, model_path='deepseek-ai/deepseek-vl-1.3b-chat', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl.models import VLChatProcessor

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).cuda().eval()
        self.model.vl_chat_processor = self.vl_chat_processor
        self.model.tokenizer = self.tokenizer

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
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

            q = q.replace("<ImageHere>","<image_placeholder>")
            # with torch.cuda.amp.autocast():
            with torch.no_grad():
                response, history = get_response_concat(model=self.model,question=q, image_path_list=images, history=history)

            responses.append(response)
        return responses

