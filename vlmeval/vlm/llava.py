import torch
from PIL import Image
from abc import abstractproperty
import os, sys
import os.path as osp
from ..smp import *
from ..utils import DATASET_TYPE, CustomPrompt

class LLaVA(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self, 
                 name,
                 model_path_map = {
                     'llava_v1.5_7b': 'liuhaotian/llava_v1.5_7b',
                     'llava_v1.5_13b': 'liuhaotian/llava_v1.5_13b',
                     'llava_v1_7b': 'Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. '
                 },
                 **kwargs): 
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn("Please install llava before using LLaVA")
            sys.exit(-1)
            
        self.model_path_map = model_path_map
        assert name in self.model_path_map or osp.exists(name) or splitlen(name) == 2
        if name in self.model_path_map:
            model_path = self.model_path_map[name]
        else:
            model_path = name

        assert osp.exists(model_path) or splitlen(model_path) == 2
        
        model_name = 'llava-v1.5-7b' if model_path == 'Lin-Chen/ShareGPT4V-7B' else get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path, 
            model_base=None, 
            model_name=model_name, 
            device='cpu', 
            device_map='cpu'
        )
        self.model = self.model.cuda()
        self.conv_mode =  'llava_v1'

        kwargs_default = dict(do_sample=True, temperature=0.2, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice':
            return True
        return False
    
    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += "\n请直接回答选项字母。" if cn_string(prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += "\n请直接回答问题。" if cn_string(prompt) else "\nAnswer the question directly."

        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
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
            output_ids = self.model.generate(input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip().split("</s>")[0]
        return output
