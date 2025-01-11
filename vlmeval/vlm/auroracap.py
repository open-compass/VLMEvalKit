import torch
import warnings
import logging
import copy as cp
from PIL import Image
import pandas as pd
import string
import re
try:
    import av
    from av.codec.context import CodecContext
except:
    logging.critical('Please install by `pip install av`"')
import numpy as np
import os.path as osp
from huggingface_hub import snapshot_download
from .base import BaseModel
from ..smp import isimg, listinstr, cn_string
from ..dataset import DATASET_TYPE, DATASET_MODALITY
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPImageProcessor,
)
try:
    from .auroracap_xtuner.model.aurora import AuroraEncoder, AuroraModel
    from .auroracap_xtuner.utils import PROMPT_TEMPLATE
except Exception as e:
    logging.critical('Please install AuroraCap to use this model by `git clone https://github.com/rese1f/aurora.git` and link `src/xtuner/xtuner` to `vlmeval/vlm/auroracap_xtuner`"')
    logging.critical('Please install packages by `pip install mmengine` and `pip install -U "xtuner[deepspeed]"`')
    raise e

try:
    from llava.mm_utils import tokenizer_image_token, process_anyres_image
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
except ImportError:
    logging.error("LLaVA is not installed. Please install LLaVA to use this model.")


class AuroraCap(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False


    def __init__(
        self,
        model_path: str='wchai/AuroraCap-7B-VID-xtuner',
        resolution: int=378,
        token_merge_ratio: float=0.01,
        max_frames_num: int=32,
        slowfast: bool=False,
        conv_template="vicuna_v1",
        video_decode_backend: str='pyav',
        **kwargs
    ):
        assert model_path is not None

        self.VIDEO_LLM = True
        self.model_path = model_path
        pretrained_pth = snapshot_download(repo_id=model_path) if not osp.isdir(model_path) else model_path
        pretrained_llm = pretrained_pth
        pretrained_vit = osp.join(pretrained_pth, "visual_encoder")

        self.model = AuroraModel(
            slowfast=slowfast,
            llm=AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=pretrained_llm,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='cuda',
            ),
            visual_encoder=AuroraEncoder.from_pretrained(
                pretrained_model_name_or_path=pretrained_vit,
                torch_dtype=torch.float16,
                device_map='cuda',
            ),
        ).eval()

        projector_path = osp.join(pretrained_pth, "projector")
        self.model.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='cuda')

        # compute token merge ratio settings
        self.patch_size = self.model.visual_encoder.config.patch_size
        self.num_layers = self.model.visual_encoder.config.num_hidden_layers
        self.token_merge_ratio = token_merge_ratio
        self.conv_template = conv_template
        self.video_decode_backend = video_decode_backend
        self.max_frames_num = int(max_frames_num)

        tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=pretrained_llm,
                    trust_remote_code=True,
                    padding_side="right",
        )
        processor = CLIPImageProcessor.from_pretrained(
            pretrained_model_name_or_path="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",  # use standard CLIP processor
            trust_remote_code=True,
            size=resolution,
            crop_size=resolution,
        )

        self.processor = processor
        self.tokenizer = tokenizer

        self.kwargs = kwargs

        torch.cuda.empty_cache()


    def adjust_kwargs(self, dataset):
        kwargs = cp.deepcopy(self.kwargs)
        kwargs["temperature"] = 0.0
        kwargs["do_sample"] = False

        if listinstr(['MMMU', 'MMStar', 'Math'], dataset):
            # These datasets may lead the model to work as a CoT-alike behaviour.
            # Allow to output longer.
            kwargs['max_new_tokens'] = 512
            return kwargs
        if DATASET_TYPE(dataset) in ['MCQ', 'Y/N']:
            kwargs['max_new_tokens'] = 64
        elif DATASET_TYPE(dataset) == 'Caption' and 'COCO' in dataset:
            kwargs['max_new_tokens'] = 64
        elif DATASET_TYPE(dataset) == 'VQA':
            if listinstr(['OCRVQA', 'ChartQA', 'DocVQA'], dataset):
                kwargs['max_new_tokens'] = 128
            elif listinstr(['TextVQA'], dataset):
                kwargs['max_new_tokens'] = 32

        return kwargs
    

    # This one is faster
    def record_video_length_stream(self, container, indices):
        frames = []
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return frames


    # This one works for all types of video
    def record_video_length_packet(self, container):
        frames = []
        # https://github.com/PyAV-Org/PyAV/issues/1269
        # https://www.cnblogs.com/beyond-tester/p/17641872.html
        # context = CodecContext.create("libvpx-vp9", "r")
        for packet in container.demux(video=0):
            for frame in packet.decode():
                frames.append(frame)
        return frames


    def read_video_pyav(self, video_path, num_frm=8):
        if isinstance(video_path, dict):
            video_path = video_path['video_path']

        if "webm" not in video_path and "mkv" not in video_path:
            # For mp4, we try loading with stream first
            try:
                container = av.open(video_path)
                total_frames = container.streams.video[0].frames
                sampled_frm = min(total_frames, num_frm)
                indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
                frames = self.record_video_length_stream(container, indices)
            except:
                container = av.open(video_path)
                frames = self.record_video_length_packet(container)
                total_frames = len(frames)
                sampled_frm = min(total_frames, num_frm)
                indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
                frames = [frames[i] for i in indices]
        else:
            container = av.open(video_path)
            frames = self.record_video_length_packet(container)
            total_frames = len(frames)
            sampled_frm = min(total_frames, num_frm)
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            frames = [frames[i] for i in indices]
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    def process_images(self, images, image_processor, model_cfg):
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                def expand2square(pil_img, background_color=(122, 116, 104)):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                new_images.append(image)
        elif image_aspect_ratio == "anyres":
            for image in images:
                image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
                new_images.append(image)
        else:
            return image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images
    
    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def generate_inner(self, message, dataset=None):
        if dataset is not None:
            kwargs = self.adjust_kwargs(dataset)
        else:
            kwargs = self.kwargs

        if DATASET_MODALITY(dataset) == 'VIDEO':
            context, visuals = self.message_to_promptvideo(message)
        elif DATASET_MODALITY(dataset) == 'IMAGE':
            context, visuals = self.message_to_promptimg(message, dataset)

        if DATASET_MODALITY(dataset) == "VIDEO" and visuals is not None:
            if self.video_decode_backend == "decord":
                video = self.load_video(visuals, self.max_frames_num)
            elif self.video_decode_backend == "pyav":
                video = self.read_video_pyav(visuals, num_frm=self.max_frames_num)
        elif DATASET_MODALITY(dataset) == "IMAGE" and visuals is not None:
            if isinstance(visuals, str):
                visuals = [visuals]
            images = [Image.open(frame_path).convert('RGB') for frame_path in visuals]
            video = np.stack([np.array(img) for img in images])
        else:
            video = None

        image_tensor = self.process_images(video, self.processor, self.model.config).cuda()
        if type(image_tensor) is list:
            image_tensor = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensor]
        elif image_tensor is not None:
            image_tensor = image_tensor.to(dtype=torch.float16, device=self.model.device)
        else:
            image_tensor = None

        if image_tensor is not None and isinstance(image_tensor, list) and DEFAULT_IMAGE_TOKEN not in context:
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(image_tensor)
            image_tokens = " ".join(image_tokens)
            question = image_tokens + "\n" + context
        elif image_tensor is not None and DEFAULT_IMAGE_TOKEN in context:
            image_tokens = [DEFAULT_IMAGE_TOKEN]
            question = image_tokens + "\n" + context
        else:
            question = context
            
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()


        if image_tensor is not None:
            kwargs["image_sizes"] = [video[idx].size for idx in range(len(video))]
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = 1024
        if "temperature" not in kwargs:
            kwargs["temperature"] = 0
        if "top_p" not in kwargs:
            kwargs["top_p"] = None
        if "num_beams" not in kwargs:
            kwargs["num_beams"] = 1



        input_ids_list = [tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")]
        pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.model.device)
        attention_masks = input_ids.ne(pad_token_ids).to(self.model.device)

        data = dict()
        if DATASET_MODALITY(dataset) == 'VIDEO':
            data["pixel_values"] = image_tensor.unsqueeze(0)
        elif DATASET_MODALITY(dataset) == 'IMAGE':
            data["pixel_values"] = image_tensor
        data["input_ids"] = input_ids
        data["attention_mask"] = attention_masks
        self.model.visual_encoder.reset_tome_r(self.token_merge_ratio)
        output = self.model(data, mode="inference")
        cont = self.model.llm.generate(
            **output,
            do_sample=True if kwargs["temperature"] > 0 else False,
            temperature=kwargs["temperature"],
            top_p=kwargs["top_p"],
            num_beams=kwargs["num_beams"],
            max_new_tokens=kwargs["max_new_tokens"],
        )
        answer = self.tokenizer.batch_decode(cont, skip_special_tokens=True)

        return answer
