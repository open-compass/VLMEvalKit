    # flake8: noqa: F401, F403
import abc
import argparse
import csv
import json
import multiprocessing as mp
import numpy as np
import os, sys, time, base64, io
import os.path as osp
import copy as cp
import pickle
import random as rd
import requests
import shutil
import string
import subprocess
import warnings
import pandas as pd
from collections import OrderedDict, defaultdict
from multiprocessing import Pool, current_process
from tqdm import tqdm
from PIL import Image
from uuid import uuid4
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
from tabulate import tabulate_formats, tabulate
from huggingface_hub import scan_cache_dir
import logging

def bincount(lst):
    bins = defaultdict(lambda: 0)
    for item in lst:
        bins[item] += 1
    return bins

def read_ok(img_path):
    if not osp.exists(img_path):
        return False
    try:
        im = Image.open(img_path)
        assert im.size[0] > 0 and im.size[1] > 0
        return True
    except:
        return False

def get_cache_path(repo_id):
    hf_cache_info = scan_cache_dir()
    repos = list(hf_cache_info.repos)
    repo = None
    for r in repos:
        if r.repo_id == repo_id:
            repo = r
            break
    if repo is None:
        return None
    revs = list(repo.revisions)
    rev2keep, last_modified = None, 0
    for rev in revs:
        if rev.last_modified > last_modified:
            rev2keep, last_modified = rev, rev.last_modified 
    if rev2keep is None:
        return None
    return str(rev2keep.snapshot_path)

def md5(file_pth):
    with open(file_pth, 'rb') as f:
        hash = hashlib.new('md5')
        for chunk in iter(lambda: f.read(2**20), b''):
            hash.update(chunk)
    return str(hash.hexdigest())

def proxy_set(s):
    import os
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s

logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
    except ImportError:
        rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger

def get_rank_and_world_size():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, world_size

def circular_pred(df, extract_func=None):
    if extract_func is None:
        extract_func = lambda x: x
    df = df.sort_values('index')
    from vlmeval.utils import can_infer_option
    shift = int(1e6)

    choices = [extract_func(x) for x in df['prediction']]
    pred_map = {i: c for i, c in zip(df['index'], choices)}
    flag_map = {i: True for i in pred_map if i < 1e6}
    valid_map = {i: True for i in pred_map if i < 1e6}
    for i in df['index']:
        if i >= shift and pred_map[i] and pred_map[i - shift]:
            if pred_map[i] not in 'ABCDE' or pred_map[i - shift] not in 'ABCDE':
                valid_map[i % shift] = False
                continue
            if (ord(pred_map[i]) - ord(pred_map[i - shift])) % 4 == 1:
                continue
            else:
                flag_map[i % shift] = False
    flag_map = {k: v for k, v in flag_map.items() if valid_map[k]}
    flags = list(flag_map.values())
    return np.mean(flags)

def splitlen(s, sym='/'):
    return len(s.split(sym))

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def d2df(D):
    return pd.DataFrame({x: [D[x]] for x in D})

def LMUDataRoot():
    if 'LMUData' in os.environ and osp.exists(os.environ['LMUData']):
        return os.environ['LMUData']
    home = osp.expanduser('~')
    root = osp.join(home, 'LMUData')
    os.makedirs(root, exist_ok=True)
    return root

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

try:
    import decord
except ImportError:
    pass

def build_options(option_list):
    chars = string.ascii_uppercase
    s = 'There are several options: \n'
    for c, opt in zip(chars, option_list):
        if not pd.isna(opt):
            s += f'{c}. {opt}\n'
        else:
            return s
    return s

def timestr(second=True, minute=False):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')[2:]
    if second:
        return s
    elif minute:
        return s[:-2]
    else:
        return s[:-4]
    
def last_modified(pth):
    stamp = osp.getmtime(pth)
    m_ti = time.ctime(stamp)
    t_obj = time.strptime(m_ti)
    t = time.strftime('%Y%m%d%H%M%S', t_obj)[2:]
    return t

def mmqa_display(question):
    question = {k.lower(): v for k, v in question.items()}
    keys = list(question.keys())
    if 'index' in keys:
        keys.remove('index')
    keys.remove('image')

    images = question['image']
    if isinstance(images, str):
        images = [images]

    idx = 'XXX'
    if 'index' in question:
        idx = question.pop('index')
    print(f'INDEX: {idx}')

    for im in images:
        image = decode_base64_to_image(im)
        w, h = image.size
        ratio = 500 / h
        image = image.resize((int(ratio * w), int(ratio * h)))
        display(image)
        
    for k in keys:
        if not pd.isna(question[k]):
            print(f'{k.upper()}. {question[k]}')

def encode_image_to_base64(img, target_size=-1):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    tmp = osp.join('/tmp', str(uuid4()) + '.jpg')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img.save(tmp)
    ret = encode_image_file_to_base64(tmp)
    os.remove(tmp)
    return ret

def encode_image_file_to_base64(image_path):
    if image_path.endswith('.png'):
        tmp_name = f'{timestr(second=True)}.jpg'
        img = Image.open(image_path)
        img.save(tmp_name)
        result = encode_image_file_to_base64(tmp_name)
        os.remove(tmp_name)
        return result
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()
        
    encoded_image = base64.b64encode(image_data)
    return encoded_image.decode('utf-8')

def decode_base64_to_image_file(base64_string, image_path):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)
    
def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def mrlines(fname, sp='\n'):
    f = open(fname).read().split(sp)
    while f != [] and f[-1] == '':
        f = f[:-1]
    return f

def mwlines(lines, fname):
    with open(fname, 'w') as fout:
        fout.write('\n'.join(lines))

def default_set(self, args, name, default):
    if hasattr(args, name):
        val = getattr(args, name)
        setattr(self, name, val)
    else:
        setattr(self, name, default)

def dict_merge(dct, merge_dct):
    for k, _ in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], dict)):  #noqa
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def youtube_dl(idx):
    cmd = f'youtube-dl -f best -f mp4 "{idx}"  -o {idx}.mp4'
    os.system(cmd)

def run_command(cmd):
    if isinstance(cmd, str):
        cmd = cmd.split()
    return subprocess.check_output(cmd)

def ls(dirname='.', match='', mode='all', level=1):
    if dirname == '.':
        ans = os.listdir(dirname)
    else:
        ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    assert mode in ['all', 'dir', 'file']
    assert level >= 1 and isinstance(level, int)
    if level == 1:
        ans = [x for x in ans if match in x]
        if mode == 'dir':
            ans = [x for x in ans if osp.isdir(x)]
        elif mode == 'file':
            ans = [x for x in ans if not osp.isdir(x)]
    else:
        ans = [x for x in ans if osp.isdir(x)]
        res = []
        for d in ans:
            res.extend(ls(d, match=match, mode=mode, level=level-1))
        ans = res
    return ans

def download_file(url, filename=None):
    import urllib.request

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
        
    if filename is None:
        filename = url.split('/')[-1]

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    return filename

def gen_bash(cfgs, num_gpus, gpus_per_task=1):
    rd.shuffle(cfgs)
    num_bash = num_gpus // gpus_per_task
    cmds_main = []
    for i in range(num_bash):
        cmds = []
        for c in cfgs[i::num_bash]:
            port = rd.randint(30000, 50000)
            gpu_ids = list(range(i, num_gpus, num_bash))
            gpu_ids = ','.join([str(x) for x in gpu_ids])
            cmds.append(
                f'CUDA_VISIBLE_DEVICES={gpu_ids} PORT={port} bash tools/dist_train.sh {c} {gpus_per_task} '
                '--validate --test-last --test-best'
            )
        cmds_main.append('  &&  '.join(cmds) + '  &')
    timestamp = time.strftime('%m%d%H%M%S', time.localtime())
    mwlines(cmds_main, f'train_{timestamp}.sh')

def h2r(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def r2h(rgb):
    return '#%02x%02x%02x' % rgb

def fnp(model, input=None):
    from fvcore.nn import FlopCountAnalysis, parameter_count
    params = parameter_count(model)['']
    print('Parameter Size: {:.4f} M'.format(params / 1024 / 1024))
    if input is not None:
        flops = FlopCountAnalysis(model, input).total()
        print('FLOPs: {:.4f} G'.format(flops / 1024 / 1024 / 1024))
        return params, flops
    return params, None

# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False)

    def dump_csv(data, f, quoting=csv.QUOTE_MINIMAL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_MINIMAL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)

def load(f):
    def load_pkl(pth):
        return pickle.load(open(pth, 'rb'))

    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(pkl=load_pkl, json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](f) 
