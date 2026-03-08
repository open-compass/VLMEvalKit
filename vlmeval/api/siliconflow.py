import math
from vlmeval.smp import *
from vlmeval.api.base import BaseAPI

API_BASE = "https://api.siliconflow.cn/v1/chat/completions"


def resize_image(image: Image.Image, max_height: int, max_width: int) -> Image.Image:
    width, height = image.size
    if min(width, height) < 50:
        scale = 50 / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)))
    current_pixels = width * height

    if current_pixels <= max_height * max_width:
        return image

    scale = math.sqrt(max_height * max_width / current_pixels)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def encode_image(path: str, max_height: int = 1024, max_width: int = 1024) -> str:
    image = Image.open(path).convert("RGB")
    image = resize_image(image, max_height, max_width)
    width, height = image.size
    if min(height, width) < 50:
        scale = 50 / min(width, height)
        image = image.resize((int(width * scale), int(height * scale)))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


class SiliconFlowAPI(BaseAPI):

    def __init__(
        self,
        model: str = "deepseek-ai/DeepSeek-V2.5",
        retry: int = 5,
        key: str = None,
        api_base: str = API_BASE,
        verbose: bool = True,
        system_prompt: str = None,
        timeout: int = 60,
        reasoning: bool = False,  # If set, will return results in the format of {'content': '...', 'reasoning': '...'}
        **kwargs,
    ):

        self.model = model
        self.api_base = api_base
        self.reasoning = reasoning
        self.timeout = timeout

        default_kwargs = {
            "stream": False,
            "temperature": 0,
            "n": 1,
            "max_tokens": 1280,
        }
        for k, v in default_kwargs.items():
            if k not in kwargs:
                kwargs[k] = default_kwargs[k]
        if key is not None:
            self.key = key
        else:
            self.key = os.environ.get("SiliconFlow_API_KEY", "")
        headers = {"Authorization": "Bearer {}", "Content-Type": "application/json"}
        headers["Authorization"] = headers["Authorization"].format(self.key)
        self.headers = headers
        super().__init__(
            retry=retry,
            system_prompt=system_prompt,
            verbose=verbose,
            **kwargs,
        )

    @staticmethod
    def build_msgs(msgs_raw):
        messages = []
        message = {"role": "user", "content": []}
        image_b64 = None
        for msg in msgs_raw:
            if msg["type"] == "image" and not image_b64:
                image_b64 = encode_image(msg["value"])
                message["content"].append({
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                    "type": "image_url"
                })
            elif msg["type"] == "text":
                message["content"].append({"text": msg["value"], "type": "text"})

        messages.append(message)
        return messages

    def generate_inner(self, inputs, **kwargs) -> str:
        default_kwargs = self.default_kwargs
        default_kwargs.update(kwargs)

        payload = dict(
            model=self.model,
            messages=self.build_msgs(msgs_raw=inputs),
            **default_kwargs,
        )

        response = requests.post(
            self.api_base, headers=self.headers, data=json.dumps(payload), timeout=self.timeout * 1.1
        )
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code

        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            msg = resp_struct["choices"][0]["message"]
            if self.reasoning and 'reasoning_content' in msg:
                answer = {'content': msg['content'], 'reasoning': msg['reasoning_content']}
            else:
                answer = resp_struct["choices"][0]["message"]["content"].strip()
        except:
            pass
        return ret_code, answer, response
