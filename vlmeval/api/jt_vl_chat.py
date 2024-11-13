import pandas as pd
import requests
import json
import os
import base64
from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.dataset import img_root_map


API_ENDPOINT = 'https://jiutian.10086.cn/kunlun/ingress/api/h3t-eeceff/92390745235a40a484d850be19e1f8b4/ai-5d7ae47ec93f4280953273c4001aafee/service-7544ea5ee3e841ad9d01e7af44acef7c/v1/chat/completions'  # noqa: E501
APP_CODE = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzdWIiOiI5ZGQwNmQ2ZjU4YTU0ZGY0OGEzNjRhMjQyNGMwODEyNSIsImlzcyI6ImFwaS1hdXRoLWtleSIsImV4cCI6NDg4MjkwNDA3OX0.k5t_T-955xWMndzBbx4WQQNAgm5DpMos9mHm7vkFipQ3yebCFMfyufpSxORSfEVpBaDS3Nly0dd8ygQYGnDgIQcC72vQ1xtkjCP49LNcqlceoET4rGc1zwRi76XLPSGFES4GcwvEmr7Ilth7XtqZNxcDF_Z7HyHyf1-zF0JIQETYSoxenqLU-gNteNfqRUnlyCgaKh03DscAbYvtoMUxEaFa2ZqyRSwekdHI_SPKCq9aC9G19yDPHTjeiwl1ubtyC5uMy5pERn_ClRsZS3Wyb-GmD5QQsFofrWvCiU_fVJuUiez39pYZvEP8awH0R9B7SkpQ4XOzj3fdytTPYy3g6g'  # noqa: E501


class JTVLChatWrapper(BaseAPI):
    is_api: bool = True
    INTERLEAVE = False

    def __init__(self,
                 model: str = 'jt-vl-chat',
                 retry: int = 5,
                 wait: int = 5,
                 api_base: str = API_ENDPOINT,
                 key: str = APP_CODE,
                 verbose: bool = True,
                 system_prompt: str = None,
                 temperature: float = 0.7,
                 max_tokens: int = 256,
                 proxy: str = None,
                 **kwargs):
        self.model = model

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_base = api_base

        if key is None:
            key = os.environ.get('JTVLChat_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (also called app_code, obtain it here: https://github.com/jiutiancv/JT-VL-Chat)'
        )

        self.key = key
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

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
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
            if listinstr(['MathVista', 'MathVision'], dataset):
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

    def get_send_data(self,prompt, image_path, temperature, max_tokens):
        image = ''
        with open(image_path, 'rb') as f:
            image = str(base64.b64encode(f.read()), 'utf-8')
        send_data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "image_base64": image,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        return send_data

    def get_send_data_no_image(self,prompt, temperature, max_tokens):
        send_data = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        return send_data

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get('dataset', None)
        prompt, image_path = self.message_to_promptimg(message=inputs, dataset=dataset)
        # print("prompt:",prompt)
        if image_path:
            send_data = self.get_send_data(
                prompt=prompt,
                image_path=image_path,
                temperature=self.temperature,
                max_tokens=self.max_tokens)
        else:
            send_data = self.get_send_data_no_image(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens)

        json_data = json.dumps(send_data)

        header_dict = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key}

        r = requests.post(self.api_base, headers=header_dict, data=json_data, timeout=3000)
        try:
            assert r.status_code == 200
            r_json = r.json()
            output = r_json['choices'][0]['message']['content']
            if self.verbose:
                self.logger.info(f'inputs: {inputs}\nanswer: {output}')

            return 0,output,'Succeeded! '

        except:
            error_msg = f'Error! code {r.status_code} content: {r.content}'
            error_con = r.content.decode('utf-8')
            if self.verbose:
                self.logger.error(error_msg)
                self.logger.error(error_con)
                self.logger.error(f'The input messages are {inputs}.')
            return -1,error_msg,''


class JTVLChatAPI(JTVLChatWrapper):

    def generate(self, message, dataset=None):
        return super(JTVLChatAPI, self).generate(message, dataset=dataset)
