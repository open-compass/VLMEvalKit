# flake8: noqa: F401, F403
import abc
import argparse
import csv
import multiprocessing as mp
import os
import os.path as osp
from pathlib import Path
import copy as cp
import random as rd
import requests
import shutil
import subprocess
import warnings
import pandas as pd
from collections import OrderedDict, defaultdict
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
from json import JSONDecoder
from huggingface_hub.utils._cache_manager import _scan_cached_repo
from sty import fg, bg, ef, rs
import uuid
import portalocker
import functools
import logging


def run_once(f):
    """Runs a function (successfully) only once.
    The running can be reset by setting the `has_run` attribute to False
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result
        else:
            logging.info(f'Function {f.__name__} has already been called.')

    wrapper.has_run = False
    return wrapper

def random_uuid():
    return str(uuid.uuid4())

def named_uuid(name):
    return str(uuid.uuid5(name=name, namespace=uuid.NAMESPACE_URL))

def modelscope_flag_set():
    return os.environ.get('VLMEVALKIT_USE_MODELSCOPE', None) in ['1', 'True']

def process_punctuation(inText):
    import re
    outText = inText
    punct = [
        ';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-',
        '>', '<', '@', '`', ',', '?', '!'
    ]
    commaStrip  = re.compile(r'(\d)(,)(\d)')
    periodStrip = re.compile(r'(?<!\d)\.(?!\d)')
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(
                commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub('', outText, re.UNICODE)
    return outText

def h2r(value):
    if value[0] == '#':
        value = value[1:]
    assert len(value) == 6
    return tuple(int(value[i:i + 2], 16) for i in range(0, 6, 2))

def r2h(rgb):
    return '#%02x%02x%02x' % rgb

def colored(s, color):
    if isinstance(color, str):
        if hasattr(fg, color):
            return getattr(fg, color) + s + fg.rs
        color = h2r(color)
    return fg(*color) + s + fg.rs

def istype(s, type_cls):
    if type_cls is int:
        import string
        return all([ch in string.digits for ch in str(s)])
    if isinstance(s, type_cls):
        return True
    try:
        return isinstance(eval(s), type_cls)
    except Exception as _:
        return False

def bincount(lst):
    bins = defaultdict(lambda: 0)
    for item in lst:
        bins[item] += 1
    return bins

def distribution(lst):
    import numpy as np
    assert all([isinstance(x, int) or isinstance(x, float) for x in lst])
    stats = {}
    stats['mean'] = np.mean(lst)
    for i in [1, 10, 100, 500, 900, 990, 999]:
        stats[f'Q-{i}‰'] = np.quantile(lst, i / 1000)
    df = d2df(stats)
    return df

def strlen_distribution(lst):
    lens = [len(x) for x in lst]
    return distribution(lens)

def get_cache_path(repo_id, branch='main', repo_type='datasets'):
    try:
        if modelscope_flag_set():
            from modelscope.hub.file_download import create_temporary_directory_and_cache
            if repo_type == 'datasets':
                repo_type = 'dataset'
            _, cache = create_temporary_directory_and_cache(model_id=repo_id, repo_type=repo_type)
            cache_path = cache.get_root_location()
            return cache_path
        else:
            from .file import HFCacheRoot
            cache_path = HFCacheRoot()
            org, repo_name = repo_id.split('/')
            repo_path = Path(osp.join(cache_path, f'{repo_type}--{org}--{repo_name}/'))
            hf_cache_info = _scan_cached_repo(repo_path=repo_path)
            revs = {r.refs: r for r in hf_cache_info.revisions}
            if branch is not None:
                revs = {refs: r for refs, r in revs.items() if branch in refs}
            rev2keep = max(revs.values(), key=lambda r: r.last_modified)
            return str(rev2keep.snapshot_path)
    except Exception as e:
        logging.warning(f'{type(e)}: {e}')
        return None

def proxy_set(s):
    import os
    for key in ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'HTTPS_PROXY']:
        os.environ[key] = s

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def get_cuda_visible_devices():
    devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if devices is None:
        return None
    devices = [int(x) for x in devices.split(',')]
    return devices

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

def cn_string(s):
    import re
    if re.search(u'[\u4e00-\u9fff]', s):
        return True
    return False

def timestr(granularity='second'):
    s = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    assert granularity in ['second', 'minute', 'hour', 'day']
    if granularity == 'second':
        return s
    elif granularity == 'minute':
        return s[:-2]
    elif granularity == 'hour':
        return s[:-4]
    elif granularity == 'day':
        return s[:-6]

def _minimal_ext_cmd(cmd, cwd=None):
    env = {}
    for k in ['SYSTEMROOT', 'PATH', 'HOME']:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    env['LANGUAGE'] = 'C'
    env['LANG'] = 'C'
    env['LC_ALL'] = 'C'
    out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, cwd=cwd).communicate()[0]
    return out

def githash(fallback='unknown', digits=8):
    if digits is not None and not isinstance(digits, int):
        raise TypeError('digits must be None or an integer')
    try:
        import vlmeval
    except ImportError as e:
        logging.error(f'ImportError: {str(e)}')
        return fallback
    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'], cwd=vlmeval.__path__[0])
        sha = out.strip().decode('ascii')
        if digits is not None:
            sha = sha[:digits]
    except OSError:
        sha = fallback
    return sha

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
    return subprocess.check_output(cmd).decode()

@run_once
def load_env():
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    try:
        import vlmeval
    except ImportError:
        logging.error('VLMEval is not installed. Failed to import environment variables from .env file. ')
        return
    vlmeval_root = osp.realpath(vlmeval.__path__[0])
    pth = osp.join(vlmeval_root, '../.env')
    pth_default = osp.join(vlmeval_root, '../.env_template')

    from dotenv import dotenv_values
    values = dotenv_values(pth_default)
    values_update = dotenv_values(pth)
    values.update(values_update)
    for k, v in values.items():
        if v is not None and len(v):
            if k not in os.environ:
                os.environ[k] = v
            else:
                logging.info(f'Environment variable {k} is already set, will not override it with {v}')
    logging.info(f'API Keys successfully loaded from {pth}')

def pip_install_robust(package):
    import sys
    retry = 3
    while retry > 0:
        try:
            package_base = package.split('=')[0]
            module = __import__(package)
            return True
        except ImportError:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            retry -= 1
    return False


def version_cmp(v1, v2, op='eq'):
    from packaging import version
    import operator
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def toliststr(s):
    if isinstance(s, str) and (s[0] == '[') and (s[-1] == ']'):
        return [str(x) for x in eval(s)]
    elif isinstance(s, str):
        return [s]
    elif isinstance(s, list):
        return [str(x) for x in s]
    raise NotImplementedError


def extract_json_objects(text, decoder=JSONDecoder()):
    pos = 0
    while True:
        match = text.find('{', pos)
        if match == -1: break
        try:
            result, index = decoder.raw_decode(text[match:])
            yield result
            pos = match + index
        except ValueError:
            pos = match + 1


def get_gpu_memory():
    import subprocess
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    except Exception as e:
        print(f'{type(e)}: {str(e)}')
        return []


def _rm(text, key_pair):
    result_text = []
    think_contents = []
    start = 0
    text_length = len(text)
    while start < text_length:
        think_start = text.find(key_pair[0], start)
        if think_start == -1:
            result_text.append(text[start:])
            break

        think_end = text.find(key_pair[1], think_start)
        if think_end == -1:
            think_contents.append(text[start:])
            result_text.append("思考过程过长，被截断")
            break

        # 添加标签前的文本
        result_text.append(text[start:think_start])

        # 提取并存储标签内的内容 (去除<think>和</think>标签)
        content_start = think_start + len(key_pair[0])  # <think> 的长度是7
        think_contents.append(text[content_start:think_end])

        start = think_end + len(key_pair[1])  # </think> 的长度是8

    return "".join(result_text), think_contents


def extract_and_remove_think_tags(text: str, thinking_tags=['<think>', "◁/think▷", "<thinking>"]):
    result_text = text
    think_contents = []
    for bot in thinking_tags:
        eot = bot[0] + '/' + bot[1:]
        result_text, think = _rm(result_text, key_pair=(bot, eot))
        think_contents.extend(think)
    return result_text, think_contents
