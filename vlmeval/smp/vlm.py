import math
import os
import io
import pandas as pd
import numpy as np
import string
from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
import sys

Image.MAX_IMAGE_PIXELS = 1e9
im_data = np.ones([224, 224, 3])
im_data = im_data.astype(np.uint8)
im_data *= 255
EMPTY_IMAGE = Image.fromarray(im_data)


def rescale_img(img, tgt=None):
    assert isinstance(tgt, tuple) and -1 in tgt
    w, h = img.size
    if tgt[0] != -1:
        new_w, new_h = tgt[0], int(tgt[0] / w * h)
    elif tgt[1] != -1:
        new_w, new_h = int(tgt[1] / h * w), tgt[1]
    img = img.resize((new_w, new_h))
    return img


def concat_images_vlmeval(images, target_size=-1, mode='h', return_image=False):
    from .file import md5

    ims = [Image.open(im) for im in images]
    if target_size != -1:
        ims = [
            rescale_img(im, (-1, target_size) if mode == 'h' else (target_size, -1))
            for im in ims
        ]

    ws, hs = [x.width for x in ims], [x.height for x in ims]
    if mode == 'h':
        new_w, new_h = sum(ws), max(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i]), 0))
    elif mode == 'v':
        new_w, new_h = max(ws), sum(hs)
        dst = Image.new('RGB', (new_w, new_h))
        for i, im in enumerate(ims):
            dst.paste(im, (sum(ws[:i], 0)))
    if return_image:
        return dst
    else:
        _str = '\n'.join(images)
        str_md5 = md5(_str)
        tgt = osp.join('/tmp', str_md5 + '.jpg')
        dst.save(tgt)
        return tgt


def mmqa_display(question, target_size=-1):
    question = {k.lower(): v for k, v in question.items()}
    keys = list(question.keys())
    keys = [k for k in keys if k not in ['index', 'image']]

    if 'image' in question:
        images = question.pop('image')
        if images[0] == '[' and images[-1] == ']':
            images = eval(images)
        else:
            images = [images]
    else:
        images = question.pop('image_path')
        if images[0] == '[' and images[-1] == ']':
            images = eval(images)
        else:
            images = [images]
        images = [encode_image_file_to_base64(x) for x in images]

    idx = question.pop('index', 'XXX')
    print(f'INDEX: {idx}')

    for im in images:
        image = decode_base64_to_image(im, target_size=target_size)
        display(image)  # noqa: F821

    for k in keys:
        try:
            if not pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')
        except ValueError:
            if False in pd.isna(question[k]):
                print(f'{k.upper()}. {question[k]}')


def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img


def resize_image_by_short_edge(img, short_edge=-1):
    if short_edge == -1:
        return img
    assert short_edge > 0, short_edge
    factor = short_edge / min(img.size)
    return resize_image_by_factor(img, factor)


def _round_to_factor(value: float, factor: int) -> int:
    """将数值调整为最接近的factor的倍数"""
    return round(value / factor) * factor


def _floor_to_factor(value: float, factor: int) -> int:
    """将数值向下调整为factor的最大倍数"""
    return math.floor(value / factor) * factor


def _ceil_to_factor(value: float, factor: int) -> int:
    """将数值向上调整为factor的最小倍数"""
    return math.ceil(value / factor) * factor


def resize_image_by_pixel_limits(img, image_pixel_limit, factor=42):
    width, height = img.size
    min_pixels = image_pixel_limit['min_pixels']
    max_pixels = image_pixel_limit['max_pixels']

    resized_height = height
    resized_width = width
    pixels = height * width
    if pixels > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_to_factor(height / scale, factor)
        resized_width = _floor_to_factor(width / scale, factor)
    elif pixels < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_to_factor(height * scale, factor)
        resized_width = _ceil_to_factor(width * scale, factor)
    img = img.resize((resized_width, resized_height))
    return img


def convert_image_to_base64(img, fmt='PNG'):
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    return base64.b64encode(image_data).decode('utf-8')


# The `fmt` arg is only needed when writing PIL Image to io buffer
def encode_image_to_base64(img, target_size=-1, fmt='PNG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    ret = convert_image_to_base64(img, fmt)
    # The max size of the image after encoding to base64 string, 1e7 is approximately 10MB
    max_bytes = os.environ.get('VLMEVAL_MAX_IMAGE_BYTES', 1e7)
    # The max size of the image, 1e8 is a huge number, 10000 x 10000
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e8)
    # The min edge length of the image, default to 100
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 100)
    max_size = int(max_size)
    min_edge = int(min_edge)
    max_bytes = int(max_bytes)

    if min(img.size) < min_edge:
        img = resize_image_by_short_edge(img, min_edge)
        ret = convert_image_to_base64(img, fmt)

    if img.size[0] * img.size[1] > max_size:
        # For images that exceeds max_size, JPEG encoding is more efficient
        fmt = 'JPEG'
        img = resize_image_by_pixel_limits(img, {'min_pixels': 1, 'max_pixels': max_size})
        ret = convert_image_to_base64(img, fmt)

    if len(ret) > max_bytes:
        fmt = 'JPEG'
        ret = convert_image_to_base64(img, fmt)

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        img = resize_image_by_factor(img, factor)
        ret = convert_image_to_base64(img, fmt)

    if factor < 1:
        new_w, new_h = img.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret


def encode_image_file_to_base64(image_path, target_size=-1, fmt=None):
    image = Image.open(image_path)
    if fmt is None:
        suffix = osp.splitext(image_path)[1][1:]
        fmt = 'PNG' if suffix.lower() == 'png' else 'JPEG'
    return encode_image_to_base64(image, target_size=target_size, fmt=fmt)


def decode_base64_to_image(base64_string, target_size=-1):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
    except Exception as e:
        print(f'Warning: failed to decode base64 string: {type(e)} {e}')
        if os.environ.get('SKIP_CORRUPTED_IMAGE', 0):
            print('Warning: the image is corrupted, return empty image')
            image = EMPTY_IMAGE
        else:
            image = None

    if image.mode in ('RGBA', 'P', 'LA', 'CMYK'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)


def build_option_str(option_dict):
    s = ''
    for c, content in option_dict.items():
        if not pd.isna(content):
            s += f'{c}. {content}\n'
    return s


def isimg(s):
    return osp.exists(s) or s.startswith('http')


def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False


def compare_outputs(model1, model2, dataset_name, fields=[], root=None):
    import vlmeval
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import load
    default_root = osp.join(vlmeval.__path__[0], '../outputs/')
    if root is None:
        root = default_root
    if fields == []:
        fields = ['prediction', 'hit', 'score']
    dataset = build_dataset(dataset_name)
    judge_format = dataset.JUDGE_FORMAT
    pred_format = dataset.PRED_FORMAT
    kwargs = dict(dataset_name=dataset_name)
    if 'judge_name' in judge_format:
        kwargs['judge_name'] = dataset.DEFAULT_JUDGE
    kwargs1 = {'model_name': model1, **kwargs}
    kwargs2 = {'model_name': model2, **kwargs}
    pth1 = osp.join(root, model1, judge_format.format(**kwargs1))
    pth2 = osp.join(root, model2, judge_format.format(**kwargs2))
    if not osp.exists(pth1) and not osp.exists(pth2):
        print(f'Both judge files {pth1} and {pth2} not found in {root}')
        pth1 = osp.join(root, model1, pred_format.format(**kwargs1))
        pth2 = osp.join(root, model2, pred_format.format(**kwargs2))

    if not osp.exists(pth1) or not osp.exists(pth2):
        raise FileNotFoundError(f'One of {pth1} or {pth2} not found in {root}')

    data = load(pth1)
    data2 = load(pth2)
    for k in fields:
        if k in data:
            data[f'm1_{k}'] = data.pop(k)
            data[f'm2_{k}'] = data2.pop(k)
    return data
