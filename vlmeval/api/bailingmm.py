import base64
from vlmeval.smp import *
from vlmeval.api.base import BaseAPI
from vlmeval.dataset import DATASET_TYPE
from vlmeval.smp.vlm import encode_image_file_to_base64


class BailingMMWrapper(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str,
                 retry: int = 5,
                 wait: int = 5,
                 key: str = None,
                 verbose: bool = True,
                 system_prompt: str = None,
                 max_tokens: int = 1024,
                 proxy: str = None,
                 **kwargs):

        self.model = model
        self.fail_msg = 'Failed to obtain answer via BailingMM API.'
        # if key is None:
        #     key = os.environ.get('GLMV_API_KEY', None)
        # assert key is not None, (
        #     'Please set the API Key'
        # )
        self.headers = {"Content-Type": "application/json"}
        # self.headers['key'] = key
        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

    def image_to_base64(self, image_path):
        with open(image_path, 'rb') as image_file:
            encoded_string = str(base64.b64encode(image_file.read()), 'utf-8')
            return encoded_string

    def prepare_inputs(self, inputs):
        msgs = cp.deepcopy(inputs)
        content = []
        text = ''
        content = []
        for i, msg in enumerate(msgs):
            if msg['type'] == 'text':
                pass
            else:
                try:
                    image_data = self.image_to_base64(msg['value'])
                except Exception as e:
                    if self.verbose:
                        self.logger.error(e)
                    image_data = ''
                msg['value'] = image_data
            content.append(msg)
        return json.dumps(content, ensure_ascii=False)

    def generate_inner(self, inputs, **kwargs) -> str:
        assert isinstance(inputs, str) or isinstance(inputs, list)
        inputs = [inputs] if isinstance(inputs, str) else inputs

        messages = self.prepare_inputs(inputs)

        service_url = 'http://cv.gateway.alipay.com/ua/invoke'
        # service_url = 'http://gateway-cv-pre.alipay.com/ua/invoke'

        payload = {
            "serviceCode": "OC_TEST",
            "uri": "V2",
            "appId": "test",
            "appName": "test",
            "attributes": {
                "_ROUTE_": "MAYA",
                "_TIMEOUT_": "60000",
            },
            "params": {
                "objectFeatures": {
                    "struct_input": messages
                }
            },
        }

        response = requests.post(service_url, headers=self.headers, data=json.dumps(payload), verify=False)
        try:
            assert response.status_code == 200
            output = json.loads(response.text)
            answer = output['resultMap']['attributes']['pred']
            if self.verbose:
                self.logger.info(f'inputs: {inputs}\nanswer: {answer}')
            return 0, answer, 'Succeeded! '
        except Exception as e:
            if self.verbose:
                self.logger.error(e)
                self.logger.error(f'The input messages are {inputs}.')
            return -1, self.fail_msg, ''


class BailingMMAPI(BailingMMWrapper):

    def generate(self, message, dataset=None):
        return super(BailingMMAPI, self).generate(message, dataset=dataset)
