# from http import HTTPStatus
import os
import requests
from ..dataset import DATASET_TYPE, DATASET_MODALITY
from vlmeval.api.base import BaseAPI
from vlmeval.smp import *


class InternVL2_PromptUtil:

    def __init__(self, use_mpo_prompt=False):
        self.use_mpo_prompt = use_mpo_prompt

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        assert DATASET_MODALITY(dataset) != 'VIDEO', 'not supported'
        if dataset in [
            'atomic_dataset', 'electro_dataset', 'mechanics_dataset',
            'optics_dataset', 'quantum_dataset', 'statistics_dataset'
        ]:
            return False
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN', 'WeMath_COT', 'MMAlignBench'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if DATASET_MODALITY(dataset) == 'VIDEO':
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_prompt(self, line, dataset=None):
        use_cot = (os.getenv('USE_COT') == '1')
        use_mpo_prompt = self.use_mpo_prompt and (use_cot or dataset in ['MMStar', 'HallusionBench', 'OCRBench'])

        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        from ..vlm.internvl.utils import (build_multi_choice_prompt,
                                          build_mcq_cot_prompt,
                                          build_qa_cot_prompt,
                                          build_mpo_prompt,
                                          reorganize_prompt)

        tgt_path = self.dump_image(line, dataset)
        max_num = self.get_max_num(dataset)
        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench', 'AMBER'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
            else:
                prompt = question
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = build_multi_choice_prompt(line, dataset)
            if os.getenv('USE_COT') == '1':
                prompt = build_mcq_cot_prompt(line, prompt)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['LLaVABench', 'WildVision'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['OCRVQA', 'TextVQA', 'ChartQA', 'DocVQA', 'InfoVQA', 'OCRBench',
                            'DUDE', 'SLIDEVQA', 'GQA', 'MMLongBench_DOC'], dataset):
                prompt = question + '\nAnswer the question using a single word or phrase.'
            elif listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse',
                            'MMDU', 'CRPE', 'MIA-Bench', 'MM-Math', 'DynaMath',
                            'QSpatial', 'WeMath', 'LogicVista'], dataset):
                prompt = question
                if os.getenv('USE_COT') == '1':
                    prompt = build_qa_cot_prompt(line, prompt)
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            # VQA_ex_prompt: OlympiadBench, VizWiz
            prompt = line['question']
            if os.getenv('USE_COT') == '1':
                prompt = build_qa_cot_prompt(line, prompt)

        message = [dict(type='text', value=prompt)]
        image_num = len(tgt_path)
        max_num = max(1, min(max_num, 64 // image_num))
        # TODO：support upscale_flag
        message.extend([dict(type='image', value=s, max_dynamic_patch=max_num) for s in tgt_path])

        if use_mpo_prompt:
            message = build_mpo_prompt(message, line, dataset)

        # reorganize_prompt
        prompt = reorganize_prompt(message, image_num, dataset=dataset)
        prompt.replace('<image>', '<IMAGE_TOKEN>')
        message[0] = dict(type='text', value=prompt)
        return message

    def get_max_num(self, dataset):
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video', 'WorldSense']  # noqa: F841
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if DATASET_MODALITY(dataset) == 'VIDEO':
            self.max_num = 1
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
        'internvl2': InternVL2_PromptUtil(),
        'internvl2-mpo-cot': InternVL2_PromptUtil(use_mpo_prompt=True),
    }

    def __init__(self,
                 model: str = None,
                 retry: int = 5,
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
        self.timeout = timeout

        key = os.environ.get('LMDEPLOY_API_KEY', key)
        api_base = os.environ.get('LMDEPLOY_API_BASE', api_base)
        assert key is not None, 'Please set the environment variable LMDEPLOY_API_KEY.'
        assert api_base is not None, 'Please set the environment variable LMDEPLOY_API_BASE.'
        self.key = key
        self.api_base = api_base
        super().__init__(retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        model_url = ''.join([api_base.split('v1')[0], 'v1/models'])
        resp = requests.get(model_url)
        model_id_list = [str(data['id']) for data in resp.json()['data']]
        self.model = model if model in model_id_list else model_id_list[0]
        self.logger.info(f'lmdeploy evaluate model: {self.model}')
        self.set_prompt_pattern(self.model)
        if hasattr(self, 'custom_prompt'):
            self.logger.info(f'using custom prompt {self.custom_prompt}')
        self.temperature = temperature
        self.logger.info(f'Init temperature: {self.temperature}')

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
        if 'internvl2' in model_name.lower() or 'internvl3' in model_name.lower():
            self.max_tokens = 1024
            self.temperature = 0.0
            if 'mpo' in model_name.lower():
                self.max_tokens = 4096
                self.logger.info('Use custom prompt internvl2-mpo-cot')
                self.custom_prompt = 'internvl2-mpo-cot'
            else:
                self.logger.info('Use custom prompt internvl2')
                self.custom_prompt = 'internvl2'
        if 'internvl2-8b-mpo-cot'.lower() in model_name.lower():
            self.use_mpo_prompt = True
            self.max_tokens = 1024
            self.temperature = 0.0
            self.logger.info('Use custom prompt internvl2-mpo-cot')
            self.custom_prompt = 'internvl2-mpo-cot'
        if 'qvq'.lower() in model_name.lower():
            self.max_tokens = 4096
            self.temperature = 0.0
            self.logger.info('QVQ model detected, do not use custom prompt')

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
        self.logger.info(f'Generate temperature: {temperature}')
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        dataset = kwargs.pop('dataset', None)
        if dataset is not None and listinstr(['BMMR'], dataset):
            # BMMR dataset has a very long prompt, so we need to increase max_tokens
            max_tokens = 8196
            self.logger.info('BMMR dataset detected, set max_tokens to 8196')

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

            # for internvl2-8b-mpo-cot
            if getattr(self, 'use_mpo_prompt', False):
                from ..vlm.internvl.utils import mpo_post_processing
                answer = mpo_post_processing(answer, kwargs.get('dataset'))
        except:
            pass
        return ret_code, answer, response


class LMDeployAPI(LMDeployWrapper):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def generate(self, message, dataset=None):
        return super(LMDeployAPI, self).generate(message, dataset=dataset)
