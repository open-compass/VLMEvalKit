import torch
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def keep_last_n_images_and_others(input_str, n):
    parts = input_str.split("<image>")
    
    if len(parts) > n:
        kept_images = parts[-n:]
        remaining_parts = parts[:-n]
        result_str = "<image>".join(remaining_parts).replace("<image>", "") + "<image>" + "<image>".join(kept_images)
    else:
        result_str = input_str
    
    return result_str

def process_string(input_str, insertion_list, n):
    lines = input_str.split("\n")
    
    image_count = 0
    filtered_lines = []
    for line in reversed(lines):
        if "<image>" in line:
            if image_count < n:
                filtered_lines.insert(0, line)
                image_count += 1
        else:
            filtered_lines.insert(0, line)
    
    processed_str = "\n".join(filtered_lines)

    assistant_split = processed_str.split("Assistant:")
    for i in range(1, len(assistant_split)):
        if i-1 < len(insertion_list):
            assistant_split[i] = f"Assistant:{insertion_list[i-1]}{assistant_split[i]}"
        else:
            assistant_split[i] = f"Assistant:{assistant_split[i]}"

    final_str = "".join(assistant_split)
    return final_str

def get_response_concat(model, question, image_path_list, history, max_new_tokens=2048):
    # print(history)
    content = [{"type": "image"}] * len(image_path_list)
    content.append({"type": "text", "text": question},)

    messages = history.copy()
    # print(messages)
    new_input = [
        {
            "role": "user",
            "content": content,
        }
    ]
    messages.extend(new_input)
    # print(messages)
    history_answers = []
    for item in messages:
        if item["role"]=="Assistant":
            history_answers.append(item["content"])
    prompt = model.processor.apply_chat_template(messages, add_generation_prompt=True)
    # print(prompt)
    prompt = process_string(prompt, history_answers, len(image_path_list))
    prompt = keep_last_n_images_and_others(prompt, len(image_path_list))
    # print(prompt)
    inputs = model.processor(text=prompt, images=[image_path_list], return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    try:
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = model.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    except Exception as e:
        print(e)
        response = "Failed"

    new_messages = [
        {
            "role": "user",
            "content": content,
        },
        {
            "role": "Assistant",
            "content": response,            
        }
    ]
    history.extend(new_messages)
    # print(history)
    return response, history

class IDEFICS2_mimt:
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='HuggingFaceM4/idefics2-8b', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation='flash_attention_2',
            device_map='cuda',
        )
        self.model.processor = self.processor
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f'Following kwargs received: {self.kwargs}, will use as generation config. '
        )
        torch.cuda.empty_cache()

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
                pics_number += tag_number
                images = img_paths[:pics_number]
            else:
                images = img_paths[:pics_number]
            logging.info(pics_number)

            q = q.replace("<ImageHere>","")
            with torch.no_grad():
                response, history = get_response_concat(model=self.model,question=q, image_path_list=images, history=history)

            responses.append(response)
        return responses