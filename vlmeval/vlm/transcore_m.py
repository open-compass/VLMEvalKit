import os
import sys
import torch
from abc import abstractproperty
from ..smp import *
from ..utils import DATASET_TYPE

class TransCoreM:

    INSTALL_REQ = True

    def __init__(self,
                 root=None,
                 **kwargs):

        self.root = root
        sys.path.append(root)
        from transcorem.model.builder import load_pretrained_model

        model_path = 'PCIResearch/TransCore-M'
        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=None,
            device='cpu',
            device_map='cpu'
        )
        self.model = self.model.cuda()
        print("==============conv_mode: default")
        self.conv_mode = "default"

        kwargs_default = dict(do_sample=False, temperature=0.0, max_new_tokens=1024, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def get_options(self,row, options):
        parsed_options = []
        for option in options:
            option_value = row[option]
            if self.is_none(option_value):
                break
            parsed_options.append(option_value)
        return parsed_options


    def is_none(self,value):
        if value is None:
            return True
        if type(value) is float and math.isnan(value):
            return True
        if type(value) is str and value.lower() == 'nan':
            return True
        if type(value) is str and value.lower() == 'none':
            return True
        return False

    def build_prompt(self, line, dataset=None):

        from ..utils import img_root_map
        assert dataset is None or isinstance(dataset, str)
        img_root = osp.join('images', img_root_map[dataset])
        os.makedirs(img_root, exist_ok=True)

        if isinstance(line['image'], list):
            tgt_path = []
            for img, im_name in zip(line['image'], line['image_path']):
                path = osp.join(self.img_root, im_name)
                if not osp.exists(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not osp.exists(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
        
        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = ['A', 'B', 'C', 'D']
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
        else:
            prompt = line['question']
        return {'image': tgt_path, 'text': prompt}

    def generate(self, image_path, prompt, dataset=None):
        from transcorem.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from transcorem.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from transcorem.conversation import conv_templates, SeparatorStyle

        temperature=0.0
        top_p=None
        num_beams=1


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
        prompt_conv = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_conv, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs
