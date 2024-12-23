# from http import HTTPStatus
import os
import requests
from ..dataset import DATASET_TYPE
from vlmeval.api.base import BaseAPI
from vlmeval.smp import *


class InternVL2_PromptUtil:

    pattern = r'Image(\d+)'
    replacement = r'Image-\1'
    reverse_pattern = r'Image-(\d+)'
    reverse_replacement = r'Image\1'

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if listinstr(['MMBench-Video', 'Video-MME', 'MVBench', 'Video'], dataset):
            # For Video benchmarks we don't have custom prompt at here
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

    def build_video_prompt(self, prompt, dataset=None, max_frames=64):
        for start in range(0, max_frames, 8):
            images_to_remove = ''.join([f'<Image-{i}>' for i in range(start + 1, start + 9)])
            prompt = prompt.replace(images_to_remove, '')
        for i in range(max_frames):
            prompt = prompt.replace(f'Image-{i + 1}', f'Frame-{i + 1}')
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
        elif listinstr(['Video-MME'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += "\nAnswer with the option's letter from the given choices directly."
        elif listinstr(['MVBench'], dataset):
            prompt = prompt.replace('Best option:(', '')
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse'], dataset):
                prompt = question
            elif listinstr(['LLaVABench'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])
        return message

    def get_max_num(self, dataset):
        assert dataset is not None
        res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'MME-RealWorld', 'VCR_EN', 'VCR_ZH']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if listinstr(res_1_datasets, dataset):
            return 1
        elif listinstr(res_12_datasets, dataset):
            return 12
        elif listinstr(res_18_datasets, dataset):
            return 18
        elif listinstr(res_24_datasets, dataset):
            return 24
        else:
            return 6


class CogVLM2_PromptUtil:

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) in 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            question = line['question']
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
            if hint is not None:
                question = hint + '\n' + question

            option_candidate = string.ascii_uppercase
            options = {
                cand: line[cand]
                for cand in option_candidate
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = question

            if not cn_string(prompt):
                prompt = prompt + '\n' + "Answer with the option's letter from the given choices directly."
            else:
                prompt = prompt + '\n' + '请直接回答选项字母。'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=p) for p in tgt_path])
        return message


class LMDeployWrapper(BaseAPI):

    is_api: bool = True

    custom_prompt: str = None
    prompt_map = {
        'cogvlm2': CogVLM2_PromptUtil(),
        'internvl2': InternVL2_PromptUtil()
    }

    def __init__(self,
                 retry: int = 5,
                 wait: int = 5,
                 key: str = 'sk-123456',
                 verbose: bool = True,
                 temperature: float = 0.0,
                 timeout: int = 60,
                 api_base: str = None,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 **kwargs):
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        self.model = resp.json()['data'][0]['id']
        self.logger.info(f'lmdeploy evaluate model: {self.model}')
        self.set_prompt_pattern(self.model)
        if hasattr(self, 'custom_prompt'):
            self.logger.info(f'using custom prompt {self.custom_prompt}')

    def set_dump_image(self, dump_image_func):
        if self.custom_prompt in self.prompt_map:
            self.prompt_map[self.custom_prompt].dump_image_func = dump_image_func
        self.dump_image_func = dump_image_func

    def use_custom_prompt(self, dataset):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].use_custom_prompt(dataset)
        return False

    def build_prompt(self, line, dataset=None):
        if self.custom_prompt in self.prompt_map:
            return self.prompt_map[self.custom_prompt].build_prompt(line, dataset)
        raise NotImplementedError

    def set_prompt_pattern(self, model_name):
        if 'Phi-3.5-Vision'.lower() in model_name.lower():
            self.max_tokens = 1000
            self.temperature = 0.0
        if 'cogvlm2-llama3-chat-19B'.lower() in model_name.lower():
            self.max_tokens = 2048
            self.temperature = 0.0
            self.custom_prompt = 'cogvlm2'
        if 'InternVL2-'.lower() in model_name.lower():
            self.max_tokens = 1024
            self.temperature = 0.0
            self.custom_prompt = 'internvl2'

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    extra_args = msg.copy()
                    extra_args.pop('type')
                    extra_args.pop('value')
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)

        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature,
            **kwargs)
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response


class LMDeployAPI(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(LMDeployAPI, self).generate(message, dataset=dataset)
