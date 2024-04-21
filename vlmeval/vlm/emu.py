import os
import torch
from PIL import Image
import os.path as osp
from .base import BaseModel
from ..smp import *


class Emu(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_path='BAAI/Emu2-Chat',
                 **kwargs):

        self.model_path = model_path
        assert osp.exists(model_path) or splitlen(model_path) == 2

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

        local_rank = os.environ.get('LOCAL_RANK', 0)

        device_num = torch.cuda.device_count()
        assert local_rank * 2 <= device_num, 'The number of devices does not match the world size'
        assert device_num >= 2, 'You need at least 2 GPUs to use EMU'

        device_1 = local_rank
        device_2 = local_rank + device_num // 2

        torch.cuda.set_device(device_1)
        torch.cuda.set_device(device_2)

        tokenizer = AutoTokenizer.from_pretrained(model_path)  # "BAAI/Emu2-Chat"
        self.tokenizer = tokenizer
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,  # "BAAI/Emu2-Chat"
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True)

        device_map = infer_auto_device_map(
            model,
            max_memory={
                device_1: '38GiB',
                device_2: '38GiB'
            },
            no_split_module_classes=['Block', 'LlamaDecoderLayer'])

        # input and output logits should be on same device
        device_map['model.decoder.lm.lm_head'] = device_1

        model = dispatch_model(
            model,
            device_map=device_map).eval()

        self.model = model
        kwargs_default = dict(max_new_tokens=512, length_penalty=-1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        query, images = '', []
        for item in message:
            if item['type'] == 'image':
                images.append(Image.open(item['value']).convert('RGB'))
                query += '[<IMG_PLH>]'
            elif item['type'] == 'text':
                query += item['value']

        inputs = self.model.build_input_ids(
            text=[query],
            tokenizer=self.tokenizer,
            image=images
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                image=inputs['image'].to(torch.bfloat16),
                **self.kwargs)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text[0]
