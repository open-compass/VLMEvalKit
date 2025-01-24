import os
import sys
import torch
from PIL import Image
import os.path as osp
from .base import BaseModel
from ..smp import *
from huggingface_hub import snapshot_download


def get_local_root(repo_id):
    if osp.exists(repo_id) and osp.isdir(repo_id):
        return repo_id

    cache_path = get_cache_path(repo_id, repo_type='models')
    if cache_path is None:
        cache_path = snapshot_download(repo_id=repo_id)
    assert osp.exists(cache_path) and osp.isdir(cache_path)
    return cache_path


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


class Emu3_chat(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self, model_path='BAAI/Emu3-Chat', tokenizer_path='BAAI/Emu3-VisionTokenizer', **kwargs):
        assert model_path is not None
        assert tokenizer_path is not None
        try:
            from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
            local_root = get_local_root(model_path)
            sys.path.append(local_root)
            from processing_emu3 import Emu3Processor
        except Exception as err:
            raise err

        # load model wights
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True)
        model.eval()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        self.image_processor = AutoImageProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(
            tokenizer_path, device_map='cuda', trust_remote_code=True).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)
        self.kwargs = kwargs
        self.cuda = cuda

    def generate_inner(self, message, dataset=None):
        query, images = '', []
        for item in message:
            if item['type'] == 'image':
                images.append(Image.open(item['value']).convert('RGB'))
            elif item['type'] == 'text':
                query += item['value']

        inputs = self.processor(
            text=[query],
            image=images,
            mode='U',
            return_tensors="pt",
            padding="longest",
        )
        from transformers.generation.configuration_utils import GenerationConfig
        # prepare hyper parameters
        GENERATION_CONFIG = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
        )
        # generate
        outputs = self.model.generate(
            inputs.input_ids.to(self.cuda),
            GENERATION_CONFIG,
            attention_mask=inputs.attention_mask.to(self.cuda),
        )

        outputs = outputs[:, inputs.input_ids.shape[-1]:]
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return response


class Emu3_gen(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = False

    def __init__(self,
                 model_path='BAAI/Emu3-Gen',
                 tokenizer_path='BAAI/Emu3-VisionTokenizer',
                 output_path='',
                 **kwargs):

        assert model_path is not None
        assert tokenizer_path is not None
        try:
            from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
            local_root = get_local_root(model_path)
            sys.path.append(local_root)
            from processing_emu3 import Emu3Processor
        except Exception as err:
            raise err

        # load model wights
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True)
        model.eval()
        self.model = model

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        self.image_processor = AutoImageProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.image_tokenizer = AutoModel.from_pretrained(
            tokenizer_path,
            device_map='cuda',
            trust_remote_code=True).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)
        self.kwargs = kwargs
        self.cuda = cuda
        self.output_path = output_path

    def generate_inner(self, message, dataset=None):
        query = ''
        for item in message:
            if item['type'] == 'text':
                query += item['value']
            else:
                raise ValueError('Please input the text in generation stage.')

        # prepare input
        POSITIVE_PROMPT = " masterpiece, film grained, best quality."
        NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."  # noqa: E501

        classifier_free_guidance = 3.0
        prompt = "a portrait of young girl."
        prompt += POSITIVE_PROMPT

        kwargs = dict(
            mode='G',
            ratio="1:1",
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest")  # noqa: E501

        pos_inputs = self.processor(text=prompt, **kwargs)
        neg_inputs = self.processor(text=NEGATIVE_PROMPT, **kwargs)
        from transformers.generation.configuration_utils import GenerationConfig
        # prepare hyper parameters
        GENERATION_CONFIG = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)
        from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor  # noqa: E501
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                classifier_free_guidance,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
            ),
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
        ])

        # generate
        outputs = self.model.generate(
            pos_inputs.input_ids.to("cuda:0"),
            GENERATION_CONFIG,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to("cuda:0"),
        )

        mm_list = self.processor.decode(outputs[0])
        for idx, im in enumerate(mm_list):
            if not isinstance(im, Image.Image):
                continue
            im.save(os.path.join(self.output_path, f"result_{idx}.png"))
