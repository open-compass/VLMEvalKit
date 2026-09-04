import logging
import os

from huggingface_hub import snapshot_download

from vlmeval.smp import encode_image_file_to_base64, get_cache_path
from .base import BaseModel


class Pixtral(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='mistralai/Pixtral-12B-2409', **kwargs):

        self.model_path = model_path
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_inference.transformer import Transformer
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        if os.path.exists(model_path):
            cache_path = model_path
        else:
            if get_cache_path(model_path, repo_type='models') is None:
                snapshot_download(repo_id=model_path)
            cache_path = get_cache_path(self.model_path, repo_type='models')

        self.tokenizer = MistralTokenizer.from_file(f'{cache_path}/tekken.json')
        model = Transformer.from_folder(cache_path, device='cpu')
        model.cuda()
        self.model = model
        self.max_tokens = 2048

    @staticmethod
    def _mistral_content(message, allow_images=True):
        from mistral_common.protocol.instruct.messages import ImageURLChunk, TextChunk

        content = []
        for item in message:
            if item['type'] == 'text':
                content.append(TextChunk(text=item['value']))
            elif item['type'] == 'image':
                if not allow_images:
                    raise ValueError('Pixtral only supports images in user turns.')
                b64 = encode_image_file_to_base64(item['value'])
                image_url = f'data:image/jpeg;base64,{b64}'
                content.append(ImageURLChunk(image_url=image_url))
        return content

    def _run_chat(self, messages):
        try:
            from mistral_common.protocol.instruct.request import ChatCompletionRequest
            from mistral_inference.generate import generate
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        completion_request = ChatCompletionRequest(messages=messages)
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

    def generate_inner(self, message, dataset=None):
        try:
            from mistral_common.protocol.instruct.messages import UserMessage
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        content = self._mistral_content(message)
        return self._run_chat([UserMessage(content=content)])

    def chat_inner(self, message, dataset=None):
        try:
            from mistral_common.protocol.instruct.messages import (AssistantMessage, SystemMessage,
                                                                   UserMessage)
        except ImportError as err:
            logging.critical('Please install `mistral-inference` and `mistral_common`')
            raise err

        messages = []
        for turn in message:
            if turn['role'] == 'user':
                content = self._mistral_content(turn['content'])
                messages.append(UserMessage(content=content))
            elif turn['role'] == 'assistant':
                content = self._mistral_content(turn['content'], allow_images=False)
                messages.append(AssistantMessage(content=content))
            elif turn['role'] == 'system':
                content = self._mistral_content(turn['content'], allow_images=False)
                messages.append(SystemMessage(content=content))
            else:
                raise ValueError(f'Unsupported Pixtral chat role: {turn["role"]}')
        return self._run_chat(messages)
