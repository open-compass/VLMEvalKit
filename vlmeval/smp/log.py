from __future__ import absolute_import, division, print_function, unicode_literals

import binascii
import ipaddress
import random
import time
from six import ensure_text
import logging

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger_initialized = {}


def judge_router_logging():
    logger = get_logger('Judge')
    import os
    logger.info(
        "In VLMEvalKit, we use env variable `JUDGE_ROUTER` to control the source of judge models. "
        "Feasible values are `default` (if variable not set) and `modelcard`. "
    )
    judge_router = os.environ.get('JUDGE_ROUTER', 'default')
    if judge_router == 'default':
        logger.warning(
            "JUDGE_ROUTER is set to `default`, please make sure you have already set "
            "`OPENAI_API_KEY`, `OPENAI_API_BASE`, and other essential environment variables. "
        )
    elif judge_router == 'modelcard':
        logger.warning(
            "JUDGE_ROUTER is set to `modelcard`, please make sure you have access to the modelcard APIs and "
            "have all internal dependencies properly installed. "
        )
    elif judge_router == 'openrouter':
        logger.warning(
            "JUDGE_ROUTER is set to `openrouter`, please make sure you have already set "
            "`OPENROUTER_API_KEY`, `OPENROUTER_API_BASE`, and other essential environment variables. "
        )
    else:
        raise ValueError(f"Invalid JUDGE_ROUTER value: {judge_router}")


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
        '[%(asctime)s] %(levelname)s - %(name)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.propagate = False
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger


length = 53
lower_rand_num = 1 << 20
upper_rand_num = (1 << 24) - 1
