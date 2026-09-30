from __future__ import annotations
import os
import sys
import warnings

import numpy as np

from vlmeval.api.base import BaseAPI
from vlmeval.smp import get_logger, proxy_set
import time
from http import HTTPStatus
import requests
import base64
import json
import socket

from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection
from urllib3.poolmanager import PoolManager
import asyncio
CONNECT_TIMEOUT_SEC = 10
# 上游 Python 推理可能跑很久，给到 20 分钟
READ_TIMEOUT_SEC = 20 * 60

# 跟 curl 默认对齐：60s 空闲后开始发探测包
TCP_KEEPIDLE_SEC = 60
TCP_KEEPINTERVAL_SEC = 15
TCP_KEEPCOUNT = 4

logger = get_logger(__name__)


def _build_keepalive_socket_options():
    opts = list(HTTPConnection.default_socket_options) + [
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    # Linux：TCP_KEEPIDLE/INTVL/CNT 都在 IPPROTO_TCP 下
    if hasattr(socket, "TCP_KEEPIDLE"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, TCP_KEEPIDLE_SEC))
    # macOS：用 TCP_KEEPALIVE 表示 idle 秒数
    elif hasattr(socket, "TCP_KEEPALIVE"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, TCP_KEEPIDLE_SEC))
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KEEPINTERVAL_SEC))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, TCP_KEEPCOUNT))
    return opts


class KeepAliveAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs["socket_options"] = _build_keepalive_socket_options()
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **kwargs,
        )
        
        
def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')

class FlyAIVLWrapper(BaseAPI):
    is_api: bool = True

    def __init__(
        self,
        model: str = 'flyai-vl',
        key: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=32768,
        temperature=0.01,
        top_p=0.001,
        top_k=1,
        presence_penalty=0.0,
        retry: int = 5,
        use_custom_prompt: bool = False,
        use_vllm: bool = True,
        **kwargs,
    ):
        self.model = model
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            presence_penalty=presence_penalty,
        )

        super().__init__(use_custom_prompt=use_custom_prompt, **kwargs)

    def _convert_image_to_base64(self, data):
        mime_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'webp': 'image/webp',
            'svg': 'image/svg+xml',
        }
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                if key == 'image' and isinstance(value, str):
                    file_path = value
                    if file_path.startswith('file://'):
                        file_path = file_path[7:]

                    if not os.path.exists(file_path):
                        print(f"[Warning] File not found: {file_path}")
                        new_data[key] = value
                        continue

                    try:
                        with open(file_path, 'rb') as f:
                            raw_bytes = f.read()
                            image_name = f.name
                            format_type = image_name.split('.')[-1].lower()

                        b64_str = base64.b64encode(raw_bytes).decode('utf-8')

                        mime_type = mime_map[format_type]
                        new_data[key] = f"data:{mime_type};base64,{b64_str}"

                    except Exception as e:
                        print(f"[Error] Failed to encode {file_path}: {e}")
                        new_data[key] = value
                else:
                    new_data[key] = self._convert_image_to_base64(value)
            return new_data

        elif isinstance(data, list):
            return [self._convert_image_to_base64(item) for item in data]

        return data
   
    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 32 * 32
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['min_pixels', 'max_pixels', 'total_pixels', 'resized_height', 'resized_width']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
            elif s['type'] == 'video':
                value = s['value']
                if isinstance(value, list):
                    item = {
                        'type': 'video',
                        'video': [ensure_image_url(v) for v in value],
                    }
                else:
                    item = {'type': 'video', 'video': ensure_video_url(value)}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                for key in ['resized_height', 'resized_width', 'fps', 'nframes', 'sample_fps']:
                    if key in s and s[key] is not None:
                        item[key] = s[key]
                if not isinstance(value, list):
                    if self.fps is not None and 'fps' not in item:
                        item['fps'] = self.fps
                    elif self.nframe is not None and 'nframes' not in item:
                        import cv2
                        video = cv2.VideoCapture(s['value'])
                        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                        video.release()
                        if frame_count < self.nframe:
                            new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                            print(f"use {new_frame_count} for {s['value']}")
                            item['nframes'] = new_frame_count
                        else:
                            item['nframes'] = self.nframe
            elif s['type'] == 'audio':
                item = {'type': 'audio', 'audio': s['value']}
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_inner(self, inputs, **kwargs) -> str:

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append(
            {'role': 'user', 'content': self._prepare_content(inputs, dataset=kwargs.get('dataset', None))}
        )
        generation_kwargs = self.generate_kwargs.copy()
        dataset = kwargs.get('dataset', None)
        kwargs.pop('dataset', None)
        generation_kwargs.update(kwargs)
        print(generation_kwargs)
        # generate
        address = "https://fliggy-evaluate-platform.alibaba-inc.com"
        request_data = {
            "model": "flyai_vl",
            "messages": messages,
            "dataset": dataset,
            "kwargs" : generation_kwargs
        }
        request_data_b64 = self._convert_image_to_base64(request_data)

        request_data_str = json.dumps(request_data_b64)

        try:
            session = requests.Session()
            adapter = KeepAliveAdapter()
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            headers = {
               "Content-Type": "application/json; charset=UTF-8",
               "Accept": "application/json",
               "Authorization": "Bearer sk-M8ukT9rRKa0eCi71PZITqEiMHCqBJfcd"
            }
            start = time.time()

            # 提交任务
            submit_url = f"{address}/api/tpp/python/invoke"
            submit_resp = session.post(submit_url, data=request_data_str, headers=headers, timeout=50)
            task_id = submit_resp.json()['data']['task_id']
            print(f"task_id: {task_id}")

            # 轮询
            query_url = f"{address}/api/tpp/python/query_task?taskId={task_id}"
            while (time.time() - start) < 3600:
                time.sleep(15)
                resp = session.get(query_url, headers=headers, timeout=200)
                data = resp.json().get('data', {})
                state = data['taskInfo']['taskState']
                if state == 'SUCCESS':
                    answer = data['result']['response']
                    return 0, answer, 'Succeeded! '
                elif state == 'FAILED':
                    raise Exception("Task FAILED")

            raise TimeoutError("轮询超时")

        except Exception as err:
            logger.error(f'{type(err)}: {err}')
            logger.error(f'The input messages are {inputs}.')
            return -1, '', ''

        
class FlyAIVLAPI(FlyAIVLWrapper):
    def generate(self, message, dataset=None):
        print("====FlyAIVLAPI====")
        print(message)
        print(dataset)
        return super(FlyAIVLAPI, self).generate(message,dataset=dataset)
