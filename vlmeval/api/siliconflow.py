from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import img_root_map
from vlmeval.dataset import DATASET_TYPE

API_BASE = 'https://api.siliconflow.cn/v1/chat/completions'


class SiliconFlowAPI(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'deepseek-ai/DeepSeek-V2.5',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 api_base: str = API_BASE,
                 verbose: bool = True,
                 system_prompt: str = None,
                 timeout: int = 60,
                 **kwargs):

        self.model = model
        self.api_base = api_base

        default_kwargs = {
            'stream': False,
            'temperature': 0,
            'frequency_penalty': 0,
            'n': 1,
            'max_tokens': 1024,
        }
        for k, v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = default_kwargs[k]
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('SiliconFlow_API_KEY', '')
        headers = {
            "Authorization": 'Bearer {}',
            "Content-Type": "application/json"
        }
        headers['Authorization'] = headers['Authorization'].format(self.key)
        self.headers = headers
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        message = {'role': 'user', 'content': []}

        def encode_image_to_base64_PNG(image_dir):
            image = Image.open(image_dir)
            from io import BytesIO
            byte_stream = BytesIO()
            image.save(byte_stream, format="PNG")
            byte_data = byte_stream.getvalue()
            base64_encoded_data = base64.b64encode(byte_data)
            base64_string = base64_encoded_data.decode("utf-8")

            return base64_string
        image_b64 = None
        for msg in msgs_raw:
            if msg['type'] == 'image' and not image_b64:
                image_b64 = encode_image_to_base64_PNG(msg['value'])
                message['content'].append({
                    'image_url': {'url': image_b64},
                    'type': 'image_url'
                })
            elif msg['type'] == 'text':
                message['content'].append({
                    'text': msg['value'],
                    'type': 'text'
                })

        messages.append(message)
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        default_kwargs = self.default_kwargs
        default_kwargs.update(kwargs)

        payload = dict(
            model=self.model,
            messages=self.build_msgs(msgs_raw=inputs),
            **default_kwargs)

        response = requests.post(self.api_base, headers=self.headers, data=json.dumps(payload))
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()
        except:
            pass
        return ret_code, answer, response


class TeleMMAPI(SiliconFlowAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'TeleAI/TeleMM',
                 key: str = None,
                 **kwargs):
        super().__init__(model=model, key=key, **kwargs)

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
        # img_root = osp.join(ROOT, 'images', img_root_map[dataset] if dataset in img_root_map else dataset)
        img_root = osp.join(ROOT, 'images', img_root_map(dataset))
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
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if 'mmmu' in dataset.lower():
            return True
        return False

    def build_mmmu(self, line):
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        return prompt

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        if 'mmmu' in dataset.lower():
            prompt = self.build_mmmu(line)

        ret = [dict(type='text', value=prompt)]
        ret.extend([dict(type='image', value=s) for s in tgt_path])
        return ret
