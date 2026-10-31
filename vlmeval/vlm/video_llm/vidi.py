import logging
import os.path as osp
import re
import subprocess
import sys

import numpy as np
import torch

from ..base import BaseModel

# Prompt of the official inference script (Vidi_7B/inference.py).
VIDI_PROMPT = ('Given the frames from a video, answer the time range in percentage that corresponds to query text '
               'split by comma. Video length is: {:.2f} and text query is: {}.')
# Temporal retrieval prompts of VLMEvalKit datasets, from which the plain query is taken.
QUERY_PATTERNS = [
    re.compile(r"match the query: '(.*)'\. The video is", re.S),  # VUE-TR
]


class Vidi(BaseModel):
    """Vidi-7B (https://github.com/bytedance/vidi), a video LLM for temporal retrieval that also listens to the
    audio track.

    The model code is not packaged: clone the repository, install the requirements of its ``Vidi_7B`` folder
    (transformers 4.44.2) and ffmpeg, and set ``root`` to that folder. Inference follows the official
    ``Vidi_7B/inference.py``: frames at 1 fps, the whole audio track, the official prompt and greedy decoding. For
    temporal retrieval datasets the plain query is put into the official prompt, and the answer (time ranges as
    fractions of the video) is returned in seconds, e.g. ``12.30 - 25.00, 40.10 - 52.70``.
    """

    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='bytedance-research/Vidi-7B', root=None, mm_splits=32, max_new_tokens=1024,
                 **kwargs):
        if root is None or not osp.isdir(root):
            raise ValueError('Set `root` to the Vidi_7B folder of a clone of https://github.com/bytedance/vidi.')
        sys.path.insert(0, root)
        try:
            import model.lmm.dattn.xattn as xattn
            from model.builder import load_pretrained_model
            from model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from model.img_utils import process_images
            from model.txt_utils import preprocess_chat, tokenizer_image_token
            from model.vid_utils import load_audio, load_video, process_audio
        except Exception as err:
            logging.critical('Install the requirements of Vidi_7B (https://github.com/bytedance/vidi) to use Vidi.')
            raise err
        # flash-attn >= 2.7 returns one more value from unpad_input than the version the model code was written for.
        unpad_input = xattn.unpad_input
        xattn.unpad_input = lambda *args, **kw: unpad_input(*args, **kw)[:4]

        self.image_token, self.image_token_index = DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        self.process_images, self.preprocess_chat, self.tokenizer_image_token = (process_images, preprocess_chat,
                                                                                 tokenizer_image_token)
        self.load_video, self.load_audio, self.process_audio = load_video, load_audio, process_audio
        # The whole model on the first visible GPU (run.py gives each process its own GPU); the official loader would
        # otherwise spread it over all visible GPUs.
        self.model, self.tokenizer, self.image_processor, self.audio_processor = load_pretrained_model(
            model_path, device='cuda:0')
        self.model.config.mm_splits = mm_splits
        self.max_new_tokens = max_new_tokens

    @staticmethod
    def video_length(path):
        try:
            out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
                                  'default=noprint_wrappers=1:nokey=1', path], capture_output=True, text=True)
            return float(out.stdout)
        except (OSError, ValueError):
            from decord import VideoReader
            vr = VideoReader(path)
            return len(vr) / vr.get_avg_fps()

    @staticmethod
    def plain_query(text):
        for pattern in QUERY_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    def _audio(self, path, length):
        try:
            audio = self.load_audio(path, self.audio_processor.sampling_rate)
        except subprocess.CalledProcessError:  # no audio track: use silence
            audio = np.zeros(int(length * self.audio_processor.sampling_rate), dtype=np.float32)
        features, size = self.process_audio(audio, self.audio_processor)
        return features.unsqueeze(0).half().cuda(), size

    def generate_inner(self, message, dataset=None):
        videos = [m['value'] for m in message if m['type'] == 'video']
        assert len(videos) == 1, 'Vidi takes exactly one video.'
        path = videos[0]
        text = '\n'.join(m['value'] for m in message if m['type'] == 'text')
        length = self.video_length(path)
        query = self.plain_query(text)
        if query is not None:
            text = VIDI_PROMPT.format(length, query[:-1] if query.endswith('.') else query)

        frames = self.process_images(self.load_video(path), self.image_processor, self.model.config)
        frames = frames.unsqueeze(0).half().cuda()
        audio, audio_size = self._audio(path, length)
        prompt = self.preprocess_chat([{'from': 'human', 'value': self.image_token + '\n' + text}], self.tokenizer)
        input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.image_token_index, return_tensors='pt')
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids.unsqueeze(0).cuda(),
                images=frames,
                audios=audio,
                audio_sizes=[audio_size],
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id)
        answer = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if query is None:
            return answer
        spans = []
        for start, end in re.findall(r'([\d|\.]+)-([\d|\.]+)', answer):  # the official parsing of the answer
            try:
                spans.append(f'{float(start) * length:.2f} - {float(end) * length:.2f}')
            except ValueError:
                continue
        return ', '.join(spans) if spans else answer
