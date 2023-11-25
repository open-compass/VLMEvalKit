import torch
from PIL import Image
from abc import abstractproperty
import os
import os.path as osp
from vlmeval.smp import *

class LLaVA:
    def __init__(self, name, do_sample=True, temperature=0.2, max_new_tokens=512, top_p=None, num_beams=1):
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        self.name_map = {
            'llava_v1.5_7b': [
                '/mnt/petrelfs/share_data/duanhaodong/llava-v1.5-7b',
                '/cpfs01/shared/llmeval/dhd/llava-v1.5-7b'
            ],
            'llava_v1.5_13b': [
                '/mnt/petrelfs/share_data/duanhaodong/llava-v1.5-13b',
                '/cpfs01/shared/llmeval/dhd/llava-v1.5-13b'
            ],
            'llava_v1_7b': [
                '/mnt/petrelfs/share_data/duanhaodong/LLaVA-7B-v1'
            ],
        }
        model_path = None
        assert name in self.name_map or osp.exists(name)
        if name in self.name_map:
            for pth in self.name_map[name]:
                if osp.exists(pth):
                    model_path = pth
        else:
            model_path = name
        assert model_path is not None
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, 
            model_base=None, 
            model_name=get_model_name_from_path(model_path), 
            device='cpu', 
            device_map='cpu'
        )
        self.model = self.model.cuda()
        if 'v1' in model_path.lower():
            self.conv_mode =  'llava_v1'
        else:
            self.conv_mode = 'vicuna_v1'
        self.do_sample = do_sample
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.num_beams = num_beams

    def build_mmbench_prompt(self, img_dir, line):
        os.makedirs(img_dir, exist_ok=True)
        idx = line['index']
        img = line['image']
        tgt_path = osp.join(img_dir, f'{idx}.jpg')
        decode_base64_to_image_file(img, tgt_path)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question + hint + '\n' + question

        option_candidate = ['A', 'B', 'C', 'D', 'E']
        options = {
            cand: line[cand]
            for cand in option_candidate
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if not cn_string(prompt):
            prompt = prompt + "\n" + "Answer with the option's letter from the given choices directly."
        else:
            prompt = prompt + "\n" + "请直接回答选项字母。"
        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        image = Image.open(image_path).convert('RGB')
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images([image], self.image_processor, args).to('cuda', dtype=torch.float16)
        if self.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, do_sample=self.do_sample, temperature=self.temperature, 
                top_p=self.top_p, num_beams=self.num_beams, max_new_tokens=self.max_new_tokens, 
                stopping_criteria=[stopping_criteria])
        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        return output