import sys
import re

import torch
from PIL import Image
from abc import abstractproperty
from collections import defaultdict
import os.path as osp
import numpy as np
from .minigpt4_mmbench import MiniGPT4_mmbench

class MiniGPT4_mllm:
    def __init__(self,
                 mode = 'v2',
                 root = '/mnt/petrelfs/share_data/duanhaodong/',
                 end_sym = "###"):
        model_path = '/mnt/petrelfs/qiaoyuxuan/share_data/minigpt4/blip2_pretrained_flant5xxl.pth'
        if mode == 'v1_7b':
            load_from = osp.join(root,'pretrained_minigpt4_7B.pth')
            llama_path = osp.join(root,'vicuna-7b-v0')
        elif mode == 'v1_13b':
            load_from = osp.join(root,'pretrained_minigpt4.pth')
            llama_path = osp.join(root,'vicuna-13b-v0')
        elif mode == 'v2':
            load_from = osp.join(root,'MiniGPT-4','minigptv2_checkpoint.pth')
            llama_path = osp.join(root,'Llama-2-7b-chat-hf')
            end_sym = "</s>"
            
        model = MiniGPT4_mmbench(
            freeze_vit=True,
            freeze_qformer=True,
            max_txt_len=160,
            end_sym=end_sym,
            low_resource=False,
            q_former_model= model_path,
            llama_model=llama_path,
            sys_prompt=  # noqa: E251
            '###Human: What is the capital of China? There are several options:\nA. Beijing\nB. Shanghai\nC. Guangzhou\nD. Shenzhen\n###Assistant: A\n'
        )

        if load_from is not None:
            state_dict = torch.load(load_from, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            msg = model.load_state_dict(state_dict, strict=False)

        self.img_size = 224
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        model.to(self.device)
        self.model = model
    
    def post_process(self, output_text):
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        pattern = re.compile(r'([A-Z]\.)')
        res = pattern.findall(output_text)
        if len(res) > 0:
            output_text = res[0][:-1]
        return output_text


    def generate(self, image_path, prompt):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((self.img_size,self.img_size))
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()/255.0
        image = image.unsqueeze(0).to(self.device)
        img_prompt = '###Human: <Img><ImageHere></Img> '
        prompt = img_prompt + ' ' + prompt
        img_embeds, _ = self.model.encode_img(image)

        prompt_segs = prompt.split('<ImageHere>')
        prompt_seg_tokens = [
            self.model.llama_tokenizer(seg,
                                 return_tensors='pt',
                                 add_special_tokens=i == 0).
            to(self.model.llama_model.model.embed_tokens.weight.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [
            self.model.llama_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)

        outputs = self.model.llama_model.generate(
            inputs_embeds=prompt_embs,
            max_new_tokens=20,
            num_beams=5,
            do_sample=False,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=-1.0,
            temperature=1.0,
            stopping_criteria=self.model.stopping_criteria,
            num_return_sequences=1)

        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token,
                                                  add_special_tokens=False)
        output_text = self.post_process(output_text)
        return output_text

