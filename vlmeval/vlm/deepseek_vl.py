import sys
import torch
from transformers import AutoModelForCausalLM
import warnings
from .base import BaseModel


class DeepSeekVL(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def check_install(self):
        try:
            import deepseek_vl
        except ImportError:
            warnings.warn(
                'Please first install deepseek_vl from source codes in: https://github.com/deepseek-ai/DeepSeek-VL')
            sys.exit(-1)

    def __init__(self, model_path='deepseek-ai/deepseek-vl-1.3b-chat', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl.models import VLChatProcessor

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).cuda().eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        content, images = '', []
        for s in message:
            if s['type'] == 'image':
                images.append(s['value'])
                content += '<image_placeholder>'
            elif s['type'] == 'text':
                content += s['value']
        conversation = [
            dict(role='User', content=content, images=images),
            dict(role='Assistant', content='')
        ]
        return conversation

    def generate_inner(self, message, dataset=None):
        conversation = self.prepare_inputs(message)
        from deepseek_vl.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
        prepare_inputs = prepare_inputs.to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs)
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer
