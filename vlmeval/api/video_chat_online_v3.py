import pandas as pd
import requests
import json
import os
import base64
from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.dataset import img_root_map
from vlmeval.api.base_client import VLLMClient

API_ENDPOINT = ''
APP_CODE = ''

class VideoChatOnlineV3Wrapper(BaseAPI):
    is_api: bool = True
    INTERLEAVE = False

    def __init__(self,
                 model: str = 'jtchat',
                 retry: int = 5,
                 wait: int = 5,
                 api_base: str = '',
                 app_code: str = '',
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 proxy: str = None,
                 **kwargs):
        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base
        self.app_code = app_code

        
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)

        img_root = os.path.join(ROOT, 'images', img_root_map(dataset) if dataset in img_root_map(dataset) else dataset)
        os.makedirs(img_root, exist_ok=True)
        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        return tgt_path

    def use_custom_prompt(self, dataset):
        assert dataset is not None
         if listinstr(['MMMU_DEV_VAL','MMMU_TEST'], dataset):
            return False
        else:
            return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        # options = {
        #     cand: line[cand]
        #     for cand in string.ascii_uppercase
        #     if cand in line and not pd.isna(line[cand])
        # }
        options = {
            cand.upper(): line[cand]
            for cand in string.ascii_letters  # string.ascii_letters 包含所有大小写字母
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)

        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and listinstr(['MME'], dataset):
            question = line['question']
            prompt = question + ' Answer the question using a single word or phrase.'
        elif dataset is not None and listinstr(['HallusionBench'], dataset):
            question = line['question']
            prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['MathVista', 'MathVision','LogicVista','MultimodalCreation','QA_CN',"VQU","Perception_ZJ","OCRBench_v2"], dataset):
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def message_to_promptimg(self, message, dataset=None):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        import warnings
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            if dataset == 'BLINK':
                image = concat_images_vlmeval(
                    [x['value'] for x in message if x['type'] == 'image'],
                    target_size=512)
            else:
                image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get('dataset', None)
        prompt, image_path = self.message_to_promptimg(message=inputs, dataset=dataset)

        client = VLLMClient(base_url=API_ENDPOINT)
        print("\n=== 示例: 多图输入流式输出 ===")
        print(API_ENDPOINT)
        image_paths = [image_path]
        import os
        for img_path in image_paths:
            if not os.path.exists(img_path):
                print(f"警告: 图片文件不存在: {img_path}")
                # 使用占位符
                image_paths = []
        
        if image_paths:
            # prompt = "请描述这些图片的内容"
            print(f"提示: {prompt}")
            print(f"图片数量: {len(image_paths)}")
            print("响应:", end=" ", flush=True)
            
            full_response = ""

            try:
                for chunk in client.stream_completion(
                    prompt=prompt,
                    image_paths=image_paths,
                    model="jtchat"
                ):
                    print(chunk, end="", flush=True)
                    full_response += chunk
            except Exception as e:
                print(f"错误: {e}")
        else:
            print("无有效图片，跳过示例2")
        
        print("\n" + "="*50)
        print("完整输出：",full_response)
        return 0,full_response,'Succeeded! '

class VideoChatOnlineV3API(VideoChatOnlineV3Wrapper):

    def generate(self, message, dataset=None):
        return super(VideoChatOnlineV3API, self).generate(message, dataset=dataset)
