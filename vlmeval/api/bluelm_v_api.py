from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
import os
import json


def multimodal(images, text, url, key, temperature=0, max_tokens=1024, history=[]):
    if images:
        pics = []
        for image in images:
            with open(image, 'rb') as f:
                pic = base64.b64encode(f.read()).decode('utf-8')
            pics.append(pic)
        data = {'images': pics, 'text': text, 'key': key, 'temperature': temperature, 'max_new_tokens': max_tokens}
    else:
        data = {'text': text, 'key': key, 'temperature': temperature, 'max_new_tokens': max_tokens}
    response = requests.post(url, json=data, headers={'Content-Type': 'application/json'})
    response = json.loads(response.text)
    return response


class BlueLMWrapper(BaseAPI):
    is_api: bool = True

    def __init__(self,
                 model: str = 'BlueLM-V-v3.0',
                 retry: int = 5,
                 wait: int = 5,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 key: str = None,
                 url: str = 'http://api-ai.vivo.com.cn/multimodal',
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer BlueLM-V API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.url = url
        self.key = key

        if self.key is None:
            self.key = os.environ.get('BLUELM_V_API_KEY', None)
        assert self.key is not None, (
            'Please set the API Key (obtain it here: '
            'contact by email : shuai.ren@vivo.com'
        )

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def message_to_promptimg(self, message, dataset=None):

        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        elif num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image']
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<image>' for x in message])
            if dataset == 'BLINK':
                image = concat_images_vlmeval(
                    [x['value'] for x in message if x['type'] == 'image'],
                    target_size=512)
            else:
                image = [x['value'] for x in message if x['type'] == 'image']

        if dataset in ['MMBench_DEV_EN_V11', 'MMBench_DEV_CN_V11', 'MMBench_TEST_EN_V11', 'MMBench_TEST_CN_V11',
                       'AI2D_TEST', 'AI2D_TEST_TO_MASK', 'MMMU_DEV_VAL']:
            prompt = prompt.replace('Please select the correct answer from the options above.',
                                    'Answer with the option’s letter from the given choices directly.')
        elif dataset in ['ChartQA_TEST']:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                    'Answer the question using a single number or phrase.')
        elif dataset in ['DocVQA_VAL', 'DocVQA_TEST', ]:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                    'Give the short answer directly.')
        elif dataset in ['TextVQA_VAL']:
            prompt = prompt.replace('Answer the question using a single word or phrase.',
                                    'When the provided information is insufficient, respond with ’Unanswerable’.'
                                    'Answer the question using a single word or phrase.')
        elif dataset in ['MTVQA_TEST']:
            prompt = prompt.replace('\nAnswer the question using a word or phrase in the language of the question.', '')
        elif dataset in ['MathVista_MINI']:
            if 'Choices:' in prompt:
                prompt = prompt.replace('Choices:', 'Options:').replace('Hint:', 'Context:')
                for i in range(1, 7):  # replace A ~ F
                    prompt = prompt.replace(f'({chr(64 + i)})', f'{chr(64 + i)}.')
                prompt += '\nAnswer with the option’s letter from the given choices directly.'
            else:
                prompt += '\nAnswer the question using a single word or phrase.'

        return prompt, image

    def generate_inner(self, inputs, **kwargs) -> str:

        assert isinstance(inputs, str) or isinstance(inputs, list)
        pure_text = np.all([x['type'] == 'text' for x in inputs])
        assert not pure_text

        prompt, image_path = self.message_to_promptimg(inputs, kwargs['dataset'])

        try:
            response = multimodal(image_path, prompt, self.url, self.key, self.temperature, self.max_tokens)
            answer = response['result']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(f'{type(err)}: {err}')
                self.logger.error(f'The input messages are {inputs}.')
            return -1, '', ''


class BlueLM_V_API(BlueLMWrapper):

    def generate(self, message, dataset=None):
        return super(BlueLM_V_API, self).generate(message, dataset=dataset)
