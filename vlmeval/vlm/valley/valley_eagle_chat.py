import torch
from PIL import Image
from ..base import BaseModel
from ...smp import *
from typing import Dict
import logging
from transformers import set_seed
from transformers import AutoTokenizer, AutoProcessor
import re


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
GANDALF_TOKEN_INDEX = -300
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_FRAME_TOKEN = "<vi_frame>"
DEFAULT_VI_START_TOKEN = "<vi_start>"
DEFAULT_VI_END_TOKEN = "<vi_end>"
DEFAULT_GANDALF_TOKEN = "<gandalf>"
DEFAULT_EOC_TOKEN = "<eoc>"
COT_PROMPT = "\nPlease think step by step."


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def preprocess_multimodal(
    conversations,
    img_num,
    data_args,
) -> Dict:
    for sentence in conversations:
        if data_args.model_class in ["valley-product", "valley-gandalf", "tinyvalley", "valley-product-mistral"]:
            if DEFAULT_VIDEO_TOKEN in sentence["value"]:
                if data_args.use_special_start_end_token:
                    video_replace_token = \
                        (DEFAULT_VI_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_VI_END_TOKEN) * img_num
                else:
                    video_replace_token = DEFAULT_IMAGE_TOKEN * img_num
                sentence["value"] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
                sentence["value"] = video_replace_token + '\n' + sentence["value"]
            else:
                segs = re.split(DEFAULT_IMAGE_TOKEN, sentence["value"])
                if data_args.use_special_start_end_token:
                    sentence["value"] = \
                        (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN).join(segs[: img_num + 1])\
                        + "".join(segs[img_num + 1:])
                else:
                    sentence["value"] = DEFAULT_IMAGE_TOKEN.join(segs[: img_num + 1]) + "".join(
                        segs[img_num + 1:]
                    )
        elif data_args.model_class in ["valley-video", "valley-video-mistral"]:
            if DEFAULT_IMAGE_TOKEN in sentence["value"] or DEFAULT_VIDEO_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
        else:
            raise Exception(f"unknown model class : {data_args.model_class}")

    return conversations


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    gandalf_token_index=GANDALF_TOKEN_INDEX,
    return_tensors=None,
):
    def split_with_token(string, token):
        result = string.split(token)
        for i in range(len(result) - 1):
            result.insert(i * 2 + 1, token)
        return result

    prompt_chunks = split_with_token(prompt, DEFAULT_IMAGE_TOKEN)
    prompt_chunks = sum([split_with_token(chunk, DEFAULT_GANDALF_TOKEN) for chunk in prompt_chunks], [])
    input_ids, offset = ([tokenizer.bos_token_id], 1) if getattr(tokenizer,'bos_token',None) else ([], 0)
    token2index = {DEFAULT_IMAGE_TOKEN: image_token_index, DEFAULT_GANDALF_TOKEN: gandalf_token_index}
    for chunk in prompt_chunks:
        if chunk in token2index:
            input_ids.append(token2index[chunk])
        else:
            chunk_ids = tokenizer(chunk).input_ids
            # For Qwen2-7B, bos token exists but does not appear in the beginning
            if chunk_ids[0] != getattr(tokenizer, 'bos_token_id', None):
                offset = 0
            input_ids.extend(chunk_ids[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


BLACK_IMG_ENV = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00' + \
                b'\x00\x00\x03\x08\x02\x00\x00\x00\xd9J"\xe8\x00\x00\x00' + \
                b'\x12IDAT\x08\x1dcd\x80\x01F\x06\x18`d\x80\x01\x00\x00Z\x00' + \
                b'\x04we\x03N\x00\x00\x00\x00IEND\xaeB`\x82'


class ValleyEagleChat(BaseModel):

    def __init__(self,
                 model_path='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        from .valley_eagle.model.language_model.valley_qwen2 import ValleyQwen2ForCausalLM
        from .valley_eagle.util.mm_utils import process_anyres_image
        from .valley_eagle import conversation as conversation_lib
        from .valley_eagle.util.data_util import dynamic_preprocess, preprocess

        torch_dtype = torch.float16
        padding_side = 'left'
        use_fast = True
        trust_remote_code = True
        output_logits = False
        conversation_tag = 'qwen2'
        max_new_tokens: int = 384
        seed = 42
        black_img = BLACK_IMG_ENV
        disable_torch_init()
        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_logits = output_logits
        self.conversation_tag = conversation_tag
        conversation_lib.default_conversation = conversation_lib.conv_templates[self.conversation_tag]
        # Load model
        logging.info(f"Start loading valley model from {model_path}")
        self.model_path = model_path
        self.model = ValleyQwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
        self.model = self.model.to(self.device).half()
        self.process_anyres_image = process_anyres_image
        self.preprocess = preprocess

        # should check this code
        self.model.config.min_tile_num = 1
        self.model.config.max_tile_num = 9
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code
        )
        self.tokenizer.padding_side = padding_side
        logging.info("Load model success!")

        self.black_img = black_img
        self.max_new_tokens = max_new_tokens
        # Load image preprocessor
        from transformers import SiglipImageProcessor
        self.qwen2vl_processor = None
        self.image_processor = SiglipImageProcessor.from_pretrained(self.model.config.mm_vision_tower)
        self.image_processor.crop_size = self.image_processor.size["height"]
        # self.vision_tower.load_model()  # vision_tower is an instance
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def expand2square(self,pil_img, background_color):
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

    def preprocess_images(
        self, image_binary_list
    ):
        images = [Image.open(binary).convert("RGB") for binary in image_binary_list]
        image_sizes_list = [img.size for img in images]
        video_pad = []
        for img in images:
            if self.model.config.anyres:
                image = self.process_anyres_image(img, self.image_processor, self.model.config.grid_pinpoints)
            else:
                image = self.image_processor(img, return_tensors="pt")["pixel_values"][0]

            video_pad.append(image)

        video_pad = (
            [self.black_img] if len(video_pad) == 0 else video_pad
        )  # black image is not processed by danyres
        # import pdb; pdb.set_trace()
        if not self.model.config.anyres:
            video = torch.stack(video_pad, dim=0)
        else:
            video = [torch.stack(img, dim=0) for img in video_pad]
        return video, image_sizes_list

    def generate_inner(self, message, dataset=None):
        if self.qwen2vl_processor is None:
            if dataset == 'OCRBench':
                self.qwen2vl_processor = AutoProcessor.from_pretrained(
                    self.model.config.eagle_vision_tower,
                    max_pixels=1280 * 28 * 28,
                    min_pixels=10 * 10 * 28 * 28
                )
            else:
                self.qwen2vl_processor = AutoProcessor.from_pretrained(
                    self.model.config.eagle_vision_tower,
                    max_pixels=1280 * 28 * 28
                )

        messages = []
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])

        if dataset in ["MMMU_DEV_VAL", "MMStar", "OCRBench", "MMVet"]:
            messages.append({"from": 'human', "value": text + COT_PROMPT})
        else:
            messages.append({"from": 'human', "value": text})

        messages_qwen = []
        image_list = []
        for image_file in images:
            from qwen_vl_utils import fetch_image
            image = fetch_image({"image": image_file})
            image_list.append(image)
        messages_qwen.append({'role': 'user', "content": [{"type": "text", "text": text}]})
        messages_qwen.append({"role": "assistant", "content": [{"type": "text", "text": ""}]})

        text = self.qwen2vl_processor.apply_chat_template(
            messages_qwen[:-1],
            tokenize=False,
            add_generation_prompt=True
        )
        text_segs = re.split("<image>", text)
        text = "<|vision_start|><|image_pad|><|vision_end|>".join(text_segs[: len(image_list) + 1]) + \
            "".join(text_segs[len(image_list) + 1:])

        sources = self.qwen2vl_processor(text=[text], images=image_list, padding=True, return_tensors="pt")
        mask_len = len(self.qwen2vl_processor(
            text=[re.sub(r"assistant\\\n[\s\S]*", "assistant\n", text)],
            images=image_list,
            padding=True,
            return_tensors="pt"
        )["input_ids"][0])
        sources["input_ids"] = sources["input_ids"][0]
        sources["labels"] = torch.cat([torch.tensor([-100] * mask_len), sources["input_ids"][mask_len:]], dim=0)
        data_dict_qwen2vl = sources
        video_images_tensor, image_sizes_list = self.preprocess_images(images)

        img_length = len(video_images_tensor)
        source = preprocess_multimodal(messages, img_length, self.model.config)

        data_dict = self.preprocess(
            source,
            self.tokenizer,
            has_image=True,
            only_mask_system=False,
            inference=True,
        )

        input_ids = data_dict['input_ids']

        input_ids = input_ids.unsqueeze(0).to(self.device)
        if img_length:
            images = [item.to(self.device).half() for item in video_images_tensor]
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=[images],
                image_sizes=[image_sizes_list],
                pixel_values=data_dict_qwen2vl['pixel_values'].to(self.device),
                image_grid_thw=data_dict_qwen2vl['image_grid_thw'].to(self.device),
                pixel_values_videos=None,
                video_grid_thw=None,
                do_sample=False,
                max_new_tokens=2048,
                repetition_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True, output_scores=True)

        input_token_len = input_ids.shape[1]
        generation_text = self.tokenizer.batch_decode(output_ids.sequences[:, input_token_len:])[0]

        generation_text = generation_text.replace("<|im_end|>", "")
        return generation_text
