import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModel

from .utils import (build_multi_choice_prompt,
                    build_yesorno_cot_prompt,
                    build_mcq_cot_prompt,
                    build_qa_cot_prompt,
                    reorganize_prompt,
                    load_image)

from ..base import BaseModel
from ...smp import *

upper_path = Path(__file__).parent

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()


class RBdashMMChat3(BaseModel):
    def __init__(self,
                 model_path,
                 load_in_8bit=False,
                 # Best-of-N parameters
                 best_of_n=1,
                 **kwargs):
        assert best_of_n >= 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.system_prompt = None
        self.cot_prompt = None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto").eval()
        self.device = 'cuda'

        self.best_of_n = best_of_n
        kwargs_default = dict(do_sample=False, max_new_tokens=16384, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

    def build_prompt(self, line, dataset=None):
        tgt_path = self.dump_image(line, dataset)
        if dataset is not None and listinstr(['MMBench_V11'], dataset):  # cot0, v1: self.system_prompt = None
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['MMStar'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)  # cot0, v1: self.system_prompt = None
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            prompt = build_multi_choice_prompt(line, dataset)  # cot1, r1: self.system_prompt = R1_SYSTEM_PROMPT
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):  # cot1, r1: self.system_prompt = R1_SYSTEM_PROMPT
            prompt = line['question']
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['OCRBench'], dataset):  # cot0, v1: self.system_prompt = None
            prompt = line['question'] + '\nAnswer the question using a single word or phrase. The image is guaranteed to contain the correct answer. Please provide the most likely answer — do not answer "No." Let us think step by step.'
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):  # cot0, v1: self.system_prompt = None
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['HallusionBench'], dataset):  # cot自定义, v1: self.system_prompt = None
            self.cot_prompt = None
            prompt = build_yesorno_cot_prompt(line, line['question'], self.cot_prompt)
        elif dataset is not None and listinstr(['MMVet'], dataset):  # cot0 v1: self.system_prompt = None
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is not None and listinstr(['MMBench_V11'], dataset):
            self.max_num = 6
        elif dataset is not None and listinstr(['MMStar'], dataset):
            self.max_num = 8
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            self.max_num = 24
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            self.max_num = 6
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['MMVet'], dataset):
            self.max_num = 24
        else:
            self.max_num = 6
    
    @torch.no_grad()
    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        kwargs_default = self.kwargs.copy()
        kwargs_default['do_sample'] = kwargs_default.get('do_sample', False)
        kwargs_default['temperature'] = 0.6
        kwargs_default['top_p'] = 0.95

        if listinstr(['MMMU_DEV_VAL', 'MathVista_MINI'], dataset):
            self.system_prompt = R1_SYSTEM_PROMPT
        else:
            self.system_prompt = None

        if self.system_prompt is not None:
                self.model.system_message = self.system_prompt
                
        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt,
            generation_config=kwargs_default,
            verbose=True
        )

        return response


class RBdashMMChat3_5(BaseModel):
    def __init__(self,
                 model_path,
                 load_in_8bit=False,
                 # Best-of-N parameters
                 best_of_n=1,
                 **kwargs):
        assert best_of_n >= 1
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.37.2', 'ge')

        self.system_prompt = None
        self.cot_prompt = None

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=load_in_8bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto").eval()
        self.device = 'cuda'

        self.best_of_n = best_of_n
        kwargs_default = dict(do_sample=False, max_new_tokens=16384, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        return True

    def build_prompt(self, line, dataset=None):
        tgt_path = self.dump_image(line, dataset)
        if dataset is not None and listinstr(['MMBench_V11'], dataset):  # cot1, v1: self.system_prompt = R1_SYSTEM_PROMPT
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MMStar'], dataset):  # cot1, v1: self.system_prompt = R1_SYSTEM_PROMPT
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):  # cot1, r1: self.system_prompt = R1_SYSTEM_PROMPT
            prompt = build_multi_choice_prompt(line, dataset)
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_mcq_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):  # cot1, r1: self.system_prompt = R1_SYSTEM_PROMPT
            prompt = line['question']
            self.cot_prompt = 'Please answer the question and put the final answer within \\boxed{}.'
            prompt = build_qa_cot_prompt(line, prompt, self.cot_prompt)
        elif dataset is not None and listinstr(['OCRBench'], dataset):  # cot0, v1: self.system_prompt = None
            prompt = line['question'] + '\nAnswer the question using a single word or phrase. The image is guaranteed to contain the correct answer. Please provide the most likely answer — do not answer "No." Let us think step by step.'
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):  # cot0, v1: self.system_prompt = None
            prompt = build_multi_choice_prompt(line, dataset)
        elif dataset is not None and listinstr(['HallusionBench'], dataset):  # cot自定义, v1: self.system_prompt = None
            self.cot_prompt = None
            prompt = build_yesorno_cot_prompt(line, line['question'], self.cot_prompt)
        elif dataset is not None and listinstr(['MMVet'], dataset):  # cot0 v1: self.system_prompt = None
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is not None and listinstr(['MMBench_V11'], dataset):
            self.max_num = 6
        elif dataset is not None and listinstr(['MMStar'], dataset):
            self.max_num = 8
        elif dataset is not None and listinstr(['MMMU_DEV_VAL'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['MathVista_MINI'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['OCRBench'], dataset):
            self.max_num = 24
        elif dataset is not None and listinstr(['AI2D_TEST'], dataset):
            self.max_num = 6
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            self.max_num = 6
        elif dataset is not None and listinstr(['MMVet'], dataset):
            self.max_num = 12
        else:
            self.max_num = 6
    
    @torch.no_grad()
    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list, pixel_values_list = [], []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and listinstr(['MMMU'], dataset)
                curr_pixel_values = load_image(
                    file_name, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
            pixel_values = load_image(
                image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        kwargs_default = self.kwargs.copy()
        kwargs_default['do_sample'] = kwargs_default.get('do_sample', False)
        kwargs_default['temperature'] = 0.6
        kwargs_default['top_p'] = 0.95

        if listinstr(['MMMU_DEV_VAL', 'MathVista_MINI', "MMBench_V11", "MMStar"], dataset):
            self.system_prompt = R1_SYSTEM_PROMPT
        else:
            self.system_prompt = None

        if self.system_prompt is not None:
                self.model.system_message = self.system_prompt
                
        response = self.model.chat(
            self.tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_list,
            question=prompt,
            generation_config=kwargs_default,
            verbose=True
        )

        return response