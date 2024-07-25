from vlmeval.smp import *
from vlmeval.api.base import BaseAPI


class QwenVLWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'qwen-vl-plus',
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 temperature: float = 0.0,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        assert model in ['qwen-vl-plus', 'qwen-vl-max']
        self.model = model
        import dashscope
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature
        if key is None:
            key = os.environ.get('DASHSCOPE_API_KEY', None)
        assert key is not None, (
            'Please set the API Key (obtain it here: '
            'https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start)'
        )
        dashscope.api_key = key
        if proxy is not None:
            proxy_set(proxy)
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    # inputs can be a lvl-2 nested list: [content1, content2, content3, ...]
    # content can be a string or a list of image & text
    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(text=msg['value']))
                elif msg['type'] == 'image':
                    content_list.append(dict(image='file://' + msg['value']))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(text=text)]
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
        from dashscope import MultiModalConversation
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
        messages = self.prepare_inputs(inputs)
        gen_config = dict(max_output_tokens=self.max_tokens, temperature=self.temperature)
        gen_config.update(kwargs)
        try:
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if self.verbose:
                print(response)
            answer = response.output.choices[0]['message']['content'][0]['text']
            return 0, answer, 'Succeeded! '
        except Exception as err:
            if self.verbose:
                self.logger.error(err)
                self.logger.error(f'The input messages are {inputs}.')

            return -1, '', ''


class QwenVLAPI(QwenVLWrapper):

    def generate(self, message, dataset=None):
        return super(QwenVLAPI, self).generate(message)
