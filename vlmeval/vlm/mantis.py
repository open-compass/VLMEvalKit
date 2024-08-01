import PIL
import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
import warnings

try:
    from mantis.models.mllava import LlavaForConditionalGeneration, MLlavaProcessor
    from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor
    from mantis.models.conversation import conv_mllava_v1 as default_conv, conv_templates

except Exception as e:
    warnings.warn(
        "Mantis is not installed. Please install Mantis to use this model.Please use 'pip install git+https://github.com/TIGER-AI-Lab/Mantis.git' to install")

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except Exception as e:
    warnings.warn("Upgrade transformers to use Mantis's idefics model.\nError: %s" % e)
except:
    warnings.warn('Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git')

try:
    import flash_attn

    best_fit_attn_implementation = 'flash_attention_2'
except ImportError:
    best_fit_attn_implementation = 'eager'
class Mantis(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='TIGER-Lab/Mantis-8B-siglip-llama3', **kwargs):
        assert model_path is not None


        self.model_path = model_path
        attn_implementation = best_fit_attn_implementation
        self._is_idefics = 'idefics' in model_path.lower()
        if not self._is_idefics:
            if 'fuyu' in model_path.lower():
                self.processor = MFuyuProcessor.from_pretrained(self.model_path)
                model = MFuyuForCausalLM.from_pretrained(self.model_path, device_map='cuda',
                                                               attn_implementation=attn_implementation,
                                                               torch_dtype=torch.float16)
            else:
                self.processor = MLlavaProcessor.from_pretrained(self.model_path)
                model = LlavaForConditionalGeneration.from_pretrained(self.model_path, device_map='cuda',
                                                                            attn_implementation=attn_implementation,
                                                                            torch_dtype=torch.float16)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            model = AutoModelForVision2Seq.from_pretrained(self.model_path, device_map='cuda', torch_dtype=torch.float16)
        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

        self.tokenizer = self.processor.tokenizer

    def output_process(self, answer):
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()
        elif '<|end_header_id|>\n\n' in answer:
            answer = answer.split('<|end_header_id|>\n\n')[2].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        elif '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        elif '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        return answer


    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                images.append(Image.open(msg['value']).convert('RGB'))
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')
        if self._is_idefics:
            # Follow the idefics implementation:
            content = []
            if self.DEFAULT_IMAGE_TOKEN not in content:
                for _ in images:
                    content.append({'type': 'image'})
            content.append({'type': 'text', 'text': content})
            prompt = [{'role': 'user', 'content': content}]
            prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        else:
            if 'llama-3' in self.model.language_model.name_or_path.lower():
                conv = conv_templates['llama_3']
                terminators = [
                    self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.convert_tokens_to_ids('<|eot_id|>')
                ]
            else:
                conv = default_conv
                terminators = None
            if 'eos_token_id' not in self.kwargs:
                self.kwargs['eos_token_id'] = terminators

            conv = conv.copy()
            conv.append_message(conv.roles[0], content)
            conv.append_message(conv.roles[1], '')
            assert conv.messages[-1][0] == conv.roles[1] and conv.messages[-1][1] == '', 'Format check'
            prompt = conv.get_prompt()

        inputs = self.processor(prompt, images, return_tensors='pt',truncation=True)
        if 'image_patches' in inputs.keys():
            inputs['image_patches'] = inputs['image_patches'][0]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        output = self.model.generate(**inputs, **self.kwargs)
        output = output[0]
        generated_ids = output[inputs['input_ids'].shape[-1]:]
        answer = self.processor.decode(generated_ids , skip_special_token=True)
        answer = self.output_process(answer)
        return answer
