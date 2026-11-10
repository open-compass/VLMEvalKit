import logging
import math

import torch
from PIL import Image

from vlmeval.dataset import DATASET_MODALITY
from vlmeval.smp import version_cmp
from .base import BaseModel


class MiniCPM_OCR(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(
        self,
        model_path='openbmb/MiniCPM-OCR-1.0',
        downsample_mode='4x',
        max_slice_nums=9,
        use_image_id=True,
        default_prompt='Text Recognition:',
        **kwargs,
    ):
        try:
            import transformers
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as exc:
            logging.critical('Please install transformers>=5.21.0 to use MiniCPM-OCR.')
            raise exc

        err_msg = (f'MiniCPM-OCR requires transformers>=5.21.0, but found {transformers.__version__}. '
                   'Please upgrade transformers.')
        # Use 5.21.0.dev0 so that pre-releases of 5.21.0 (e.g. 5.21.0rc1) are not rejected.
        assert version_cmp(transformers.__version__, '5.21.0.dev0', 'ge'), err_msg

        assert model_path is not None
        self.model_path = model_path
        self.downsample_mode = downsample_mode
        self.max_slice_nums = max_slice_nums
        self.use_image_id = use_image_id
        self.default_prompt = default_prompt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, dtype=self.dtype)
        self.model.to(self.device).eval()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        self.kwargs = kwargs

    def build_content(self, message):
        content = []
        has_text = False
        for item in message:
            if item['type'] == 'text':
                text = item['value']
                if not isinstance(text, str):
                    try:
                        is_nan = math.isnan(text)
                    except TypeError:
                        is_nan = False
                    text = '' if text is None or is_nan else str(text)
                content.append(dict(type='text', text=text))
                has_text = has_text or bool(text.strip())
            elif item['type'] == 'image':
                image = Image.open(item['value']).convert('RGB')
                content.append(dict(type='image', image=image))
            else:
                raise ValueError(f'Unsupported message type for MiniCPM-OCR: {item["type"]}')
        if not has_text:
            content.append(dict(type='text', text=self.default_prompt))
        return content

    def get_processor_kwargs(self, dataset=None):
        # Frame-based video datasets should use a single slice per frame, like MiniCPM-V-4.6.
        is_video = dataset is not None and DATASET_MODALITY(dataset) == 'VIDEO'
        return dict(
            downsample_mode=self.downsample_mode,
            max_slice_nums=1 if is_video else self.max_slice_nums,
            use_image_id=False if is_video else self.use_image_id,
        )

    def generate_inner(self, message, dataset=None):
        messages = [dict(role='user', content=self.build_content(message))]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors='pt',
            enable_thinking=False,
            processor_kwargs=self.get_processor_kwargs(dataset),
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=8192,
            do_sample=False,
            num_beams=1,
        )
        gen_kwargs.update(self.kwargs)

        generated_ids = self.model.generate(
            **inputs,
            downsample_mode=self.downsample_mode,
            **gen_kwargs,
        )
        trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], generated_ids)]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0].strip() if output_text else ''
