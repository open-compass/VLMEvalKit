import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy


# This function is used to split Eagle-X5-34B
def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers_map = {
        'Eagle-X5-34B-Chat': 60,
        'Eagle-X5-34B-Plus': 60
    }
    if model_name not in num_layers_map:
        return 'cuda'
    num_layers = num_layers_map[model_name] + 8
    # Since the first GPU will be used for ViT, treat it as 0.5 GPU.
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[-1] = num_layers - sum(num_layers_per_gpu[:-1])
    num_layers_per_gpu[0] -= 4
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['model.vision_tower'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = rank
    device_map['model.rotary_emb'] = rank
    device_map['model.mm_projector'] = rank
    device_map['lm_head'] = rank
    device_map[f'model.layers.{num_layers - 1}'] = rank

    logging.warning("Remove L157-L158 in https://github.com/NVlabs/EAGLE/blob/fef95f103b5e9899acbbe2c237e5b99147ab7e8e/eagle/model/builder.py to make it work properly.")  # noqa: E501
    return device_map


class Eagle(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='NVEagle/Eagle-X5-7B',
                 **kwargs):
        try:
            from eagle.model.builder import load_pretrained_model
            from eagle.utils import disable_torch_init
            from eagle.mm_utils import get_model_name_from_path
        except Exception as e:
            logging.critical('''Please install eagle before using Eagle,
            you can install it from "https://github.com/NVlabs/EAGLE.git"''')
            raise e

        warnings.warn('Please install the latest version of eagle from github before you evaluate the Eagle model.')
        assert osp.exists(model_path) or splitlen(model_path) == 2
        model_name = get_model_name_from_path(model_path)
        rank, world_size = get_rank_and_world_size()

        device_map = split_model(model_path.split('/')[-1])

        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(model_path, None, model_name, False, False, device_map=device_map)
        )
        self.model.eval()
        self.conv_mode = 'vicuna_v1'

        default_kwargs = dict(
            do_sample=True,
            temperature=0.2,
            top_p=0.5,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True
        )

        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        try:
            from eagle import conversation as conversation_lib
            from eagle.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                                         DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
            from eagle.conversation import conv_templates, SeparatorStyle
            from eagle.mm_utils import tokenizer_image_token, process_images, KeywordsStoppingCriteria
        except Exception as e:
            logging.critical('''Please install eagle before using Eagle,
            you can install it from "https://github.com/NVlabs/EAGLE.git"''')
            raise e

        kwargs = self.kwargs

        images = []
        prompt = ''

        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
            elif s['type'] == 'text':
                prompt += s['value']

        DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN * len(images)
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        images = [Image.open(s).convert('RGB') for s in images]

        image_tensor = process_images(images, self.image_processor, self.model.config)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids.unsqueeze(0),
                images=image_tensor,
                image_sizes=[img.size for img in images],
                **kwargs
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMMU'], dataset):
            return False
        if DATASET_TYPE(dataset) == 'MCQ' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        if dataset == 'MMVet':
            prompt = question + '\nAnswer the question directly. '
        elif DATASET_TYPE(dataset) == 'MCQ':
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'

            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            prompt = f'Hint: {hint}\n' if hint is not None else ''
            prompt += f'{question}\n'
            prompt += (
                f'{options_prompt}\nAnswer with the option’s letter from the given choices directly. '
                if len(options) else 'Answer the question directly. '
            )
        else:
            raise NotImplementedError

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message
