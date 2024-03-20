import sys
import torch
from transformers import AutoModelForCausalLM
import warnings
from vlmeval.smp import isimg, pip_install


class DeepSeekVL:

    INSTALL_REQ = True

    def check_install(self):
        installed = pip_install('deepseek_vl')
        if not installed:
            warnings.warn(
                'Please first install deepseek_vl from source codes in: https://github.com/deepseek-ai/DeepSeek-VL')
            sys.exit(-1)

    def __init__(self, model_path='deepseek-ai/deepseek-vl-7b-chat', **kwargs):
        self.check_install()
        assert model_path is not None
        self.model_path = model_path
        from deepseek_vl.models import VLChatProcessor

        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device='cpu')
        self.model = model.to(torch.bfloat16).cuda().eval()

        torch.cuda.empty_cache()
        default_kwargs = dict(max_new_tokens=512, do_sample=False, use_cache=True)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, msgs):
        content, images = '', []
        for s in msgs:
            if isimg(s):
                images.append(s)
                content += '<image_placeholder>'
            else:
                content += s
        conversation = [
            dict(role='User', content=content, images=images),
            dict(role='Assistant', content='')
        ]
        return conversation

    def interleave_generate(self, ti_list, dataset=None):
        conversation = self.prepare_inputs(ti_list)
        from deepseek_vl.utils.io import load_pil_images
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)
        prepare_inputs = prepare_inputs.to(self.model.device)
        input_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        outputs = self.model.language_model.generate(
            input_embeds=input_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.kwargs)
        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return answer

    def generate(self, image_path, prompt, dataset=None):
        return self.interleave_generate([image_path, prompt], dataset=dataset)
