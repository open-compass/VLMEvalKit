import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
import warnings
from huggingface_hub import snapshot_download


class Pixtral(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='mistralai/Pixtral-12B-2409', **kwargs):

        self.model_path = model_path
        try:
            from mistral_inference.transformer import Transformer
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        if os.path.exists(model_path):
            cache_path = model_path
        else:
            if get_cache_path(model_path) is None:
                snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(self.model_path)

        self.tokenizer = MistralTokenizer.from_file(f'{cache_path}/tekken.json')
        model = Transformer.from_folder(cache_path, device='cpu')
        model.cuda()
        self.model = model
        self.max_tokens = 512

    def generate_inner(self, message, dataset=None):
        try:
            from mistral_inference.generate import generate
            from mistral_common.protocol.instruct.messages import UserMessage, TextChunk, ImageURLChunk
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        msg_new = []
        for msg in message:
            tp, val = msg['type'], msg['value']
            if tp == 'text':
                msg_new.append(TextChunk(text=val))
            elif tp == 'image':
                b64 = encode_image_file_to_base64(val)
                image_url = f'data:image/jpeg;base64,{b64}'
                msg_new.append(ImageURLChunk(image_url=image_url))

        completion_request = ChatCompletionRequest(messages=[UserMessage(content=msg_new)])
        encoded = self.tokenizer.encode_chat_completion(completion_request)
        images = encoded.images
        tokens = encoded.tokens

        out_tokens, _ = generate(
            [tokens],
            self.model,
            images=[images],
            max_tokens=self.max_tokens,
            temperature=0,
            eos_id=self.tokenizer.instruct_tokenizer.tokenizer.eos_id)

        result = self.tokenizer.decode(out_tokens[0])
        return result
