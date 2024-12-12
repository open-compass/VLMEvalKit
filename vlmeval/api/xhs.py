from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

from ..vlm.qwen2_vl import Qwen2VLPromptMixin

DEFAULT_URL="http://1781574661016173.ap-southeast-1.pai-eas.aliyuncs.com/api/predict/deploy_services/v1/chat/completions"

class XDGPromptMixin(Qwen2VLPromptMixin):

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE
        if dataset in {'OCRBench'}:
            return self._build_ocrbench_prompt(line, dataset)
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
            return self._build_mmmu_prompt(line, dataset)
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""
        import string, re
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        prompt = re.sub(r"<image \d+>", "", prompt)
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            for i, p in enumerate(tgt_path):
                msgs.extend([ 
                    dict(type='text', value=f"Picture {i+1}:"),
                    dict(type='image', value=p),
                    dict(type='text', value="\n"),
                ])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_ocrbench_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs

class XDGAPI(BaseAPI, XDGPromptMixin):
    """Class for 小红书公司 VLM-XDG(小地瓜) API"""
    is_api: bool = True

    def __init__(self,
                 model: str,
                 base_url: str = DEFAULT_URL, 
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 use_custom_prompt=True,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('XDG_API_KEY', None)
        assert key is not None, ('Please set the API Key in environ like: `export XDG_API_KEY="YOUR API KAY"`')
        self.api_key = key
        self.base_url = base_url
        self._use_custom_prompt = use_custom_prompt
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        message = {'role': 'user', 'content': []}

        for msg in msgs_raw:
            if msg['type'] == 'image':
                image_b64 = encode_image_file_to_base64(msg['value'])
                message['content'].append({
                    'image_url': {'url': 'data:image/jpeg;base64,' + image_b64},
                    'type': 'image_url'
                })
            elif msg['type'] == 'text':
                message['content'].append({
                    'text': msg['value'],
                    'type': 'text'
                })

        messages.append(message)
        return messages

    def _make_payload(self, messages, gen_config):
        return {
            "messages": messages,
            'model' : f'{self.model}',
            **gen_config
        }

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)

        if 'type' in inputs[0]:
            pure_text = np.all([x['type'] == 'text' for x in inputs])
        else:
            pure_text = True
            for inp in inputs:
                if not np.all([x['type'] == 'text' for x in inp['content']]):
                    pure_text = False
                    break

        assert not pure_text
        messages = self.build_msgs(inputs)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)
        gen_config.update(kwargs)
        
        payload = self._make_payload(messages, gen_config)
        payload.pop("dataset")

        headers = {
            "User-Agent": "Test Client",
            "Authorization": f'{self.api_key}',
        }
        response = requests.post(self.base_url, json=payload, headers=headers)

        # 检查响应状态码
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code} {response.text}"
        
        # 检查响应内容
        response_data = response.json()
        assert "choices" in response_data, "Response does not contain 'choices'"
        assert "usage" in response_data, "Response does not contain 'usage'"
        ret_code = 0 if response.status_code == 200 else response.status_code 

        return ret_code, response_data["choices"][0]["message"]["content"], str(response_data)
