import math
import pandas as pd
import random
import re
import string
import torch
import torch.distributed as dist
import torchvision.transforms as T
import transformers
import warnings
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor

from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY
from ...smp import *

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def get_local_rank_and_local_world_size():
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        return 0, 1

    if 'SLURM_LOCALID' in os.environ:
        local_rank = int(os.environ['SLURM_LOCALID'])
        local_world_size = int(os.environ['SLURM_NTASKS_PER_NODE'])
        return local_rank, local_world_size

    if 'LOCAL_RANK' in os.environ and 'LOCAL_WORLD_SIZE' in os.environ:
        return int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])

    raise NotImplementedError(
        "Fail to get local_rank and local_world_size! "
        "Please ensure that you set the environment variable "
        "`LOCAL_RANK` and `LOCAL_WORLD_SIZE`"
    )


def split_model(model_path):
    num_gpus_per_node = 8
    rank, world_size = get_rank_and_world_size()
    try:
        local_rank, local_world_size = get_local_rank_and_local_world_size()
    except:
        local_rank = rank

    if 'GPUS_PER_PROCESS' in os.environ:
        gpus_per_process = int(os.environ['GPUS_PER_PROCESS'])
    else:
        gpus_per_process = 8  # default to use 8 GPUs for one model

    start_gpu = local_rank * gpus_per_process
    end_gpu = start_gpu + gpus_per_process

    assert end_gpu <= num_gpus_per_node, f"Process {local_rank} tries to access GPU {end_gpu}, " \
                                         f"but only {num_gpus_per_node} GPUs are available per node."

    visible_devices = list(range(start_gpu, end_gpu))

    device_map = {}
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    num_gpus_for_vit = 0.5
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (len(visible_devices) - num_gpus_for_vit))
    num_layers_per_gpu = [num_layers_per_gpu] * len(visible_devices)
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = visible_devices[i]
            layer_cnt += 1
    device_map['vision_model'] = visible_devices[0]
    device_map['mlp1'] = visible_devices[0]
    device_map['language_model.model.tok_embeddings'] = visible_devices[0]
    device_map['language_model.model.embed_tokens'] = visible_devices[0]
    device_map['language_model.output'] = visible_devices[0]
    device_map['language_model.model.norm'] = visible_devices[0]
    device_map['language_model.lm_head'] = visible_devices[0]
    device_map[f'language_model.model.layers.{num_layers - 1}'] = visible_devices[0]

    return device_map, visible_devices


def split_model_old(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers_map = {
        'InternVL2-8B': 32,
        'InternVL2-26B': 48,
        'InternVL2-40B': 60,
        'InternVL2-Llama3-76B': 80
    }

    if model_name not in num_layers_map:
        return 'cuda'
    num_layers = num_layers_map[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.5 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map['language_model.model.rotary_emb'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    return device_map


def build_mcq_cot_prompt(line, prompt):
    cot_prompt = (
        "Answer the preceding multiple choice question. The last line of your response should follow "
        "this format: 'Answer: \\boxed{$LETTER}' (without quotes), where LETTER is one of the options. "
        "If you are uncertain or the problem is too complex, make a reasoned guess based on the "
        "information provided. Avoid repeating steps indefinitely—provide your best guess even if "
        "unsure. Think step by step logically, considering all relevant information before answering."
    )
    prompt = prompt.replace("Answer with the option's letter from the given choices directly.", '').strip()
    prompt = prompt + '\n' + cot_prompt

    return prompt


def build_qa_cot_prompt(line, prompt):
    cot_prompt = (
        "Answer the preceding question. The last line of your response should follow this format: "
        "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your conclusion "
        "based on the reasoning provided. If you are uncertain or the problem is too complex, make "
        "a reasoned guess based on the information provided. Avoid repeating steps indefinitely—"
        "provide your best guess even if unsure. Think step by step logically, considering all "
        "relevant information before answering."
    )
    prompt = prompt + '\n' + cot_prompt

    return prompt


def build_multi_choice_prompt(line, dataset=None):
    question = line['question']
    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    if hint is not None:
        question = hint + '\n' + question

    options = {
        cand: line[cand]
        for cand in string.ascii_uppercase
        if cand in line and not pd.isna(line[cand])
    }
    for key, item in options.items():
        question += f'\n{key}. {item}'
    prompt = question

    if len(options):
        prompt += '\n请直接回答选项字母。' if cn_string(
            prompt) else "\nAnswer with the option's letter from the given choices directly."
    else:
        prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

    return prompt


def build_video_prompt(prompt, dataset=None, max_frames=64):
    for start in range(0, max_frames, 8):
        images_to_remove = ''.join([f'<Image-{i}>' for i in range(start + 1, start + 9)])
        prompt = prompt.replace(images_to_remove, '')
    for i in range(max_frames):
        prompt = prompt.replace(f'Image-{i + 1}', f'Frame-{i + 1}')
    if listinstr(['MMBench-Video'], dataset):
        prompt = prompt.replace('\nAnswer:', '')
    elif listinstr(['Video-MME'], dataset):
        prompt = prompt.replace('\nAnswer:', '')
        prompt += "\nAnswer with the option's letter from the given choices directly."
    elif listinstr(['MVBench'], dataset):
        prompt = prompt.replace('Best option:(', '')

    return prompt


def reorganize_prompt(message, image_num, dataset=None):
    if dataset is not None and listinstr(['MUIRBench'], dataset):
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        images_to_remove = ' '.join(['<image>'] * image_num)
        prompt = prompt.replace(images_to_remove, '')
        for i in range(image_num):
            prompt = prompt.replace('<image>', f'<Image-{i + 1}>', 1)
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
    elif image_num == 1:
        prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
    else:
        prompt, image_idx = '', 1
        for x in message:
            if x['type'] == 'text':
                prompt += x['value']
            elif x['type'] == 'image':
                prompt += f'<Image-{image_idx}>'
                image_idx += 1
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
        images_to_remove = ''.join([f'<Image-{i + 1}>' for i in range(image_num)])
        prompt = prompt.replace(images_to_remove, '')
    return prompt


mpo_prompt_with_final_answer = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format \"Final answer: ..\""
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)

mpo_prompt_without_final_answer = (
    "Your task is to answer the question below. "
    "Give step by step reasoning. "
    "\n\n"
    "Question:"
    "\n\n"
    "{question}"
)


def mpo_post_processing(response, dataset):

    def extract_answer(text):
        match = re.search(r'(Final answer:|Answer:)\s*(.*)', text, re.IGNORECASE)
        if match:
            return match.group(2).strip()
        return text

    if dataset is not None and (DATASET_TYPE(dataset) in ['Y/N', 'MCQ'] or listinstr(['CRPE'], dataset)):
        response = extract_answer(response).strip()
    return response


def build_mpo_prompt(message, line, dataset):
    if not listinstr(['LLaVABench'], dataset):

        if listinstr(['MMVet'], dataset):
            cot_prompt = mpo_prompt_without_final_answer
        else:
            cot_prompt = mpo_prompt_with_final_answer

        question_orig = line['question']
        if listinstr(['MathVerse', 'MathVision'], dataset):
            question_orig = question_orig.split('Question:', 1)[-1].strip()
            question_orig = question_orig.replace('Choices:\n', '').strip()

        prompt = cot_prompt.format(question=question_orig)
    else:
        prompt = line['question']
    message[0]['value'] = prompt
    return message
