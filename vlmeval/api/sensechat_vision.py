from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import img_root_map
from vlmeval.dataset import DATASET_TYPE


class SenseChatVisionWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'SenseChat-5-Vision',
                 retry: int = 5,
                 wait: int = 5,
                 ak: str = None,
                 sk: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.ak = os.environ.get('SENSECHAT_AK', None) if ak is None else ak
        self.sk = os.environ.get('SENSECHAT_SK', None) if sk is None else sk
        assert self.ak is not None and self.sk is not None
        self.max_new_tokens = max_tokens
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
        img_root = osp.join(ROOT, 'images', img_root_map[dataset] if dataset in img_root_map else dataset)
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

    def image_to_base64(self, image_path):
        import base64
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
            return encoded_string.decode('utf-8')

    def encode_jwt_token(self, ak, sk):
        import jwt
        headers = {'alg': 'HS256', 'typ': 'JWT'}
        payload = {
            'iss': ak,
            'exp': int(time.time())
            + 1800,  # 填写您期望的有效时间，此处示例代表当前时间+30分钟
            'nbf': int(time.time()) - 5,  # 填写您期望的生效时间，此处示例代表当前时间-5秒
        }
        token = jwt.encode(payload, sk, headers=headers)
        return token

    def use_custom_prompt(self, dataset):
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
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ' and 'MMMU' not in dataset:
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            if 'MathVista' in dataset:
                prompt = line['question']
            elif listinstr(['LLaVABench'], dataset):
                question = line['question']
                prompt = question + '\nAnswer this question in detail.'
            elif listinstr(['MMVet'], dataset):
                prompt = line['question']
            else:
                question = line['question']
                prompt = question + '\nAnswer the question using a single word or phrase.'
        elif dataset is not None and 'MMMU' in dataset:
            question = line['question']
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            for key, item in options.items():
                question += f'\n{key}. {item}'
            prompt = {
                'multiple-choice': 'You are an expert in {}. Please solve the university-level {} examination question, which includes interleaved images and text. Your output should be divided into two parts: First, reason about the correct answer. Then write the answer in the following format where X is exactly one of the choices given by the problem: "ANSWER: X". If you are uncertain of the correct answer, guess the most likely one.',  # noqa: E501
                'open': 'You are an expert in {}. Please solve the university-level {} examination question, which includes interleaved images and text. Your output should be divided into two parts: First, reason about the correct answer. Then write the answer in the following format where X is only the answer and nothing else: "ANSWER: X"'  # noqa: E501
            }
            subject = '_'.join(line['id'].split('_')[1:-1])
            prompt = prompt[line['question_type']].format(subject, subject) + '\n' + question
        else:
            prompt = line['question']

        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])

        return message

    def message_to_promptimg(self, message, dataset=None):
        if dataset is None or listinstr(['MMMU', 'BLINK'], dataset):
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [[x['value'] for x in message if x['type'] == 'image'][0]]
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image']
        return prompt, image

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs
        dataset = kwargs.get('dataset', None)

        if dataset is not None and listinstr(['ChartQA_TEST'], dataset):
            self.max_num = 12
        elif dataset is not None and listinstr(['DocVQA_VAL', 'DocVQA_TEST'], dataset):
            self.max_num = 18
        elif dataset is not None and listinstr(['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench'], dataset):
            self.max_num = 24
        else:
            self.max_num = 6

        if dataset is None:
            pass
        elif listinstr(['AI2D_TEST'], dataset):
            self.max_new_tokens = 10
        elif 'MMMU' in dataset:
            self.max_new_tokens = 1024
        elif 'MMBench' in dataset:
            self.max_new_tokens = 100

        prompt, image = self.message_to_promptimg(message=inputs, dataset=dataset)

        url = 'https://api.sensenova.cn/v1/llm/chat-completions'
        api_secret_key = self.encode_jwt_token(self.ak, self.sk)

        content = [{
            'image_base64': self.image_to_base64(item),
            'image_file_id': '',
            'image_url': '',
            'text': '',
            'text': '',
            'type': 'image_base64'
        } for item in image]

        content.append({
            'image_base64': '',
            'image_file_id': '',
            'image_url': '',
            'text': prompt,
            'type': 'text'
        })

        message = [{'content': content, 'role': 'user'}]

        data = {
            'messages': message,
            'max_new_tokens': self.max_new_tokens,
            'model': self.model,
            'stream': False,
        }
        headers = {
            'Content-type': 'application/json',
            'Authorization': 'Bearer ' + api_secret_key
        }

        response = requests.post(
            url,
            headers=headers,
            json=data,
        )
        request_id = response.headers['x-request-id']

        time.sleep(1)
        try:
            assert response.status_code == 200
            response = response.json()['data']['choices'][0]['message'].strip()
            if dataset is not None and 'MMMU' in dataset:
                response = response.split('ANSWER: ')[-1].strip()
            if self.verbose:
                self.logger.info(f'inputs: {inputs}\nanswer: {response}')
            return 0, response, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error('---------------------------ERROR---------------------------')
                self.logger.error(response.json())
                self.logger.error(err)
                self.logger.error('---------------------------request_id---------------------------' + request_id)
                self.logger.error(
                    'api error' + response.json()['error']['message']
                    + str([input['value'] if input['type'] == 'image' else None for input in inputs])
                )
                self.logger.error(f'The input messages are {inputs}.')
            return -1, response.json()['error']['message'], ''


class SenseChatVisionAPI(SenseChatVisionWrapper):

    def generate(self, message, dataset=None):
        return super(SenseChatVisionAPI, self).generate(message, dataset=dataset)
