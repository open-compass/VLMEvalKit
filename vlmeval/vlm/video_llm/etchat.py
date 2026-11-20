import logging
import os.path as osp
import sys

import torch

from ..base import BaseModel

# Tasks for which the official E.T. Bench script starts the answer with "Best Option: (" and penalizes the
# timestamp token, so that the model picks a letter.
CHOICE_TASKS = ('rar', 'eca', 'rvq')


class ETChat(BaseModel):
    """E.T. Chat (https://github.com/PolyU-ChenLab/ETBench), the time-sensitive video LLM of the E.T. Bench paper.
    It writes and reads timestamps as a special ``<vid>`` token that is matched to a video frame.

    The model code is not packaged: clone the repository, install its requirements (transformers 4.44.2, nncore,
    peft) and set ``root`` to the clone. Inference follows the official ``etchat/eval/infer_etbench.py``: frames at
    1 fps, the phi3 chat template, sampling with temperature 0.2 and up to 2,048 new tokens. On E.T. Bench use the
    ``_vid`` presets, which give the official prompts with ``<vid>`` tokens together with their values.
    """

    INSTALL_REQ = True
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path='PolyU-ChenLab/ETChat-Phi3-Mini-Stage-3', root=None, temperature=0.2,
                 max_new_tokens=2048, **kwargs):
        if root is None or not osp.isdir(root):
            raise ValueError('Set `root` to a clone of https://github.com/PolyU-ChenLab/ETBench.')
        sys.path.insert(0, root)
        try:
            from etchat.constants import DEFAULT_IMAGE_TOKEN
            from etchat.conversation import get_conv
            from etchat.model.builder import build_model
            from etchat.utils.inference import (KeywordsStoppingCriteria,
                                                RepetitionPenaltyLogitsProcessor)
            from etchat.utils.io import load_video
            from etchat.utils.tokenization import detokenize, tokenize
        except Exception as err:
            logging.critical('Install the requirements of https://github.com/PolyU-ChenLab/ETBench to use E.T. Chat.')
            raise err
        self.image_token, self.get_conv, self.load_video = DEFAULT_IMAGE_TOKEN, get_conv, load_video
        self.stopping_criteria, self.repetition_penalty = KeywordsStoppingCriteria, RepetitionPenaltyLogitsProcessor
        self.tokenize, self.detokenize = tokenize, detokenize
        if not osp.isdir(model_path):  # the official loader checks for files inside the model folder
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(model_path)
        # run.py gives each process its own GPU.
        self.device = 'cuda:0'
        self.model, self.tokenizer, self.transform = build_model(model_path, device=self.device)
        self.temperature, self.max_new_tokens = temperature, max_new_tokens

    def generate_inner(self, message, dataset=None):
        videos = [m['value'] for m in message if m['type'] == 'video']
        assert len(videos) == 1, 'E.T. Chat takes exactly one video.'
        texts = [m for m in message if m['type'] == 'text']
        query = '\n'.join(m['value'] for m in texts)
        # Set by the `_vid` presets of E.T. Bench: the values of the <vid> tokens in the prompt, and the task.
        src = next((m['src'] for m in texts if m.get('src') is not None), None)
        task = next((m['task'] for m in texts if m.get('task') is not None), None)

        video, tag = self.load_video(videos[0], num_threads=1)
        video = self.transform(video).half().to(self.device)

        conv = self.get_conv(self.model.config.conv_type)
        conv.append_message(conv.roles[0], self.image_token + '\n' + query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        logits_processor = None
        if task in CHOICE_TASKS:
            logits_processor = [self.repetition_penalty(1.2, self.model.config.match_token_id)]
            prompt += 'Best Option: (' if prompt[-1] == '\n' else ' Best Option: ('
        input_ids = self.tokenize(prompt, self.tokenizer).unsqueeze(0).to(self.device)

        stop_str = conv.seps[-1]
        sampling = dict(do_sample=True, temperature=self.temperature) if self.temperature > 0 else dict(do_sample=False)
        with torch.inference_mode():
            out = self.model.generate(
                input_ids,
                image=[video],
                max_new_tokens=self.max_new_tokens,
                cache_implementation=None,
                logits_processor=logits_processor,
                stopping_criteria=[self.stopping_criteria(self.tokenizer, stop_str)],
                query=[[query]],
                src=[src],
                tag=[tag],
                **sampling)
        response = self.detokenize(out[0, input_ids.size(1):], self.model, self.tokenizer)
        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()
        return response
