from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

url = "https://b-openapi.basemind.com/openapi/v1/chat/completions"
headers = {
    'X-Request-Orgcode': 'companyA',
    'Authorization': 'Bearer {}',
    'Content-Type': 'application/json'
}

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

class StepAPI(BaseAPI):

    is_api: bool = True

    def __init__(self, 
                 model: str = 'stepapi-rankboard',
                 retry: int = 10,
                 wait: int = 3,
                 key: str = None,
                 temperature: float = 0, 
                 max_tokens: int = 300,
                 verbose: bool = True,
                 system_prompt: str = None,
                 **kwargs):
        self.model = model
        self.fail_msg = 'Fail to obtain answer via API.'
        self.headers = headers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get('STEPAI_API_KEY', '')
        headers['Authorization'] = headers['Authorization'].format(self.key)

        super().__init__(retry=retry, wait=wait, verbose=verbose, system_prompt=system_prompt, **kwargs)
        
    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        message = {"role": "user", "content": []}
        
        for msg in msgs_raw:
            if isimg(msg):
                image_b64 = convert_image_to_base64(msg)
                message['content'].append({
                    "image_b64": {'b64_json': image_b64},
                    "type": "image_b64"
                })
            else:
                message['content'].append({
                    'text': msg,
                    "type": 'text'
                })

        messages.append(message)
        return messages
        
    def generate_inner(self, inputs, **kwargs) -> str:
        print(inputs, '\n')
        payload = dict(
            model=self.model, 
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages= self.build_msgs(msgs_raw=inputs), #需要构建message
            **kwargs)
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print('response is here!!:',response.text,'\n')
        ret_code = response.status_code
        # print('ret_code is:',ret_code)
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        # print('initial answer is',answer)
        try:
            resp_struct = json.loads(response.text)
            # print('resp_struct is',resp_struct)
            answer = resp_struct['choices'][0]['message']['content'].strip()
            # print('answer!!!!!!=========',answer,'\n')
        except:
            pass
        # print('finial answer is',answer)
        return ret_code, answer, response
            

class Step1V(StepAPI):

    def generate(self, image_path, prompt, dataset=None):
        return super(StepAPI, self).generate([image_path, prompt])
    
    def interleave_generate(self, ti_list, dataset=None):
        return super(StepAPI, self).generate(ti_list)