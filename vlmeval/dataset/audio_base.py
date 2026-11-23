import base64
import binascii
import os
import os.path as osp
import re

import numpy as np
import pandas as pd

from vlmeval.smp import (LMUDataRoot, atomic_write_audio_file, is_valid_audio_file, istype,
                         toliststr)
from vlmeval.smp.audio import is_reusable_audio_file
from .image_base import ImageBaseDataset


def audio_root_map(dataset):
    return dataset


def audio_read_ok(audio_path):
    return is_valid_audio_file(audio_path)


def audio_reuse_ok(audio_path):
    return is_reusable_audio_file(audio_path)


def decode_base64_to_audio_file(audio_b64, audio_path):
    audio_b64 = str(audio_b64).strip()
    if audio_b64.lower().startswith('data:'):
        if ',' not in audio_b64:
            raise ValueError('Invalid base64 audio data URI: missing comma.')
        audio_b64 = audio_b64.split(',', 1)[1]
    try:
        compact = re.sub(r'\s+', '', audio_b64)
        audio_data = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError) as err:
        raise ValueError('Invalid base64 audio payload.') from err
    atomic_write_audio_file(audio_path, lambda f: f.write(audio_data))


def has_audio_payload(audio):
    if isinstance(audio, list):
        return any(has_audio_payload(x) for x in audio)
    if audio is None or pd.isna(audio):
        return False
    return str(audio) != ''


class AudioBaseDataset(ImageBaseDataset):

    MODALITY = 'AUDIO'
    DATASET_URL = {}
    DATASET_MD5 = {}

    def __init__(self, dataset='MMSU', skip_noaudio=True):
        ROOT = LMUDataRoot()
        self.dataset_name = dataset
        self.audio_root = osp.join(ROOT, 'audios', audio_root_map(dataset))

        data = self.load_data(dataset)
        self.skip_noaudio = skip_noaudio
        if skip_noaudio and 'audio' in data and 'audio_path' not in data:
            data = data[~pd.isna(data['audio'])]

        data['index'] = [str(x) for x in data['index']]
        self.meta_only = True

        if 'audio' in data and any(has_audio_payload(x) for x in data['audio']):
            data['audio'] = [str(x) if not pd.isna(x) else '' for x in data['audio']]
            audio_map = {x: y for x, y in zip(data['index'], data['audio'])}
            for k in audio_map:
                if len(audio_map[k]) <= 64 and len(audio_map[k]) > 0:
                    idx = audio_map[k]
                    assert idx in audio_map and len(audio_map[idx]) > 64
                    audio_map[k] = audio_map[idx]

            audios = [toliststr(audio_map[k]) for k in data['index']]
            data['audio'] = [x[0] if len(x) == 1 else x for x in audios]
            self.meta_only = False

        if 'audio_path' in data:
            paths = [toliststr(x) for x in data['audio_path']]
            data['audio_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def dump_audio(self, line):
        os.makedirs(self.audio_root, exist_ok=True)

        if 'audio' in line and has_audio_payload(line['audio']):
            if isinstance(line['audio'], list):
                tgt_path = []
                if 'audio_path' in line:
                    audio_path = toliststr(line['audio_path'])
                else:
                    index = line['index']
                    audio_path = [f'{index}_{i}.wav' for i in range(len(line['audio']))]
                for audio, audio_name in zip(line['audio'], audio_path):
                    path = self._resolve_audio_path(audio_name)
                    if not audio_read_ok(path):
                        decode_base64_to_audio_file(audio, path)
                    tgt_path.append(path)
            elif isinstance(line['audio'], str) and 'audio_path' in line:
                assert isinstance(line['audio_path'], str)
                tgt_path = self._resolve_audio_path(line['audio_path'])
                if not audio_read_ok(tgt_path):
                    decode_base64_to_audio_file(line['audio'], tgt_path)
                tgt_path = [tgt_path]
            else:
                tgt_path = osp.join(self.audio_root, f"{line['index']}.wav")
                if not audio_read_ok(tgt_path):
                    decode_base64_to_audio_file(line['audio'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'audio_path' in line
            tgt_path = toliststr(line['audio_path'])
            read_ok_flag = [audio_read_ok(x) for x in tgt_path]
            if not all(read_ok_flag):
                tgt_path_abs = [self._resolve_audio_path(x) for x in tgt_path]
                read_ok_flag = [audio_read_ok(x) for x in tgt_path_abs]
                assert all(read_ok_flag), (
                    f"Field `audio` is missing and we could not find {tgt_path} "
                    'both as absolute or relative paths. '
                )
                tgt_path = tgt_path_abs

        return tgt_path

    def _resolve_audio_path(self, path):
        if osp.isabs(path):
            return path
        return osp.join(self.audio_root, path)

    def dump_image(self, line):
        return []

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_audio(line)
        question = line['question']

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='audio', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='audio', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        return msgs
