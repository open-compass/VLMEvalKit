import ast
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd
import portalocker
from PIL import Image

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from .utils import video_index as vi
from .video_base import VideoBaseDataset


class VideoIndex(VideoBaseDataset):
    """Video-Index: 840 multiple-choice video questions from 76 public video benchmarks, one question
    per video, 210 per capability group (perception, temporal, spatial / physical, reasoning /
    knowledge). Every item passed a screen with text-only, single-frame, options-only and
    shuffled-frame attackers, and its marked answer was verified against the frames.

    Two protocols:

    * video (`Video-Index`): one frame per second of the timeline, at most `max_frames` frames
      (512; uniform thinning beyond), every frame resized to a short side of `short_side` pixels
      (224). `nframe=N` keeps the same frame rule with a cap of N frames. Models that read video
      files (`VIDEO_LLM`) receive the file instead of the frames.
    * blind (`Video-Index_Blind`): the question and the options only, under four option
      permutations per item; the blind score is the mean over the permutations.

    Scoring is rule based (leading option letter, stated answer, or the option text that the reply
    repeats); no judge model is involved. Gain = video accuracy - blind accuracy.

    `limit` and `offset` (counted in items, in the order of the item file) select a slice for
    smoke tests; only the videos of the slice are downloaded.
    """

    TYPE = 'Video-MCQ'
    DEFAULT_JUDGE_MODEL = None
    HF_REPO_ID = 'GMLRVigil/Video-Index'
    ITEMS_FILENAME = 'items/meta_benchmark.jsonl'
    N_ITEMS = 840

    @classmethod
    def supported_datasets(cls):
        return ['Video-Index', 'Video-Index_Blind']

    @classmethod
    def validate_build_config(cls, config: dict) -> None:
        if 'blind' in str(config.get('dataset', '')).lower():
            return
        super().validate_build_config(config)

    def __init__(self,
                 dataset='Video-Index',
                 pack=False,
                 nframe=0,
                 fps=1.0,
                 max_frames=512,
                 short_side=224,
                 limit=None,
                 offset=0):
        self.blind = 'blind' in dataset.lower()
        self.max_frames = max_frames
        self.short_side = short_side
        self.limit = int(limit) if limit else None
        self.offset = int(offset or 0)
        if self.blind:
            nframe, fps = 0, -1
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)
        self.data = self._slice(self.data)
        self.videos = sorted(set(self.data['video']))

    def _slice(self, data):
        if self.limit is None and not self.offset:
            return data
        item_ids = list(dict.fromkeys(data['item_id']))
        stop = None if self.limit is None else self.offset + self.limit
        keep = set(item_ids[self.offset:stop])
        return data[data['item_id'].isin(keep)].reset_index(drop=True)

    # ------------------------------------------------------------------ data
    @classmethod
    def _items_file(cls, bench_root):
        path = bench_root / cls.ITEMS_FILENAME
        if path.exists():
            return path
        from huggingface_hub import hf_hub_download
        return Path(
            hf_hub_download(repo_id=cls.HF_REPO_ID,
                            repo_type='dataset',
                            filename=cls.ITEMS_FILENAME,
                            local_dir=str(bench_root)))

    @staticmethod
    def _read_items(items_file):
        with open(items_file, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    @classmethod
    def _build_tsv(cls, items_file, data_file, blind):
        records = []
        for item in cls._read_items(items_file):
            options = [str(o) for o in item['options']]
            answer_idx = int(item['answer_idx'])
            if blind:
                perms = vi.permutations_for(item['item_id'], len(options))
            else:
                perms = [list(range(len(options)))]
            for perm_idx, perm in enumerate(perms):
                records.append({
                    'index': len(records),
                    'item_id': item['item_id'],
                    'video': item['video_id'],
                    'video_path': item['video'],
                    'question': item['question'],
                    'candidates': repr([options[j] for j in perm]),
                    'answer': vi.LETTERS[perm.index(answer_idx)],
                    'perm': repr(perm),
                    'perm_idx': perm_idx,
                    'benchmark': item['benchmark'],
                    'capability_group': item['capability_group'],
                    'fine_category': item.get('fine_category', ''),
                    'duration': item.get('duration_s', ''),
                })
        dump(pd.DataFrame(records), str(data_file))

    def _missing_videos(self, bench_root, data_file):
        data = self._slice(load(str(data_file)))
        return sorted(p for p in set(data['video_path']) if not (bench_root / p).exists())

    @classmethod
    def _download_videos(cls, bench_root, missing):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=cls.HF_REPO_ID,
                          repo_type='dataset',
                          allow_patterns=list(missing),
                          local_dir=str(bench_root))

    def prepare_dataset(self, dataset):
        blind = 'blind' in dataset.lower()
        bench_root = Path(LMUDataRoot()) / 'Video-Index'
        bench_root.mkdir(parents=True, exist_ok=True)
        items_file = self._items_file(bench_root)
        data_file = bench_root / f'{dataset}.tsv'
        if not data_file.exists() or items_file.stat().st_mtime > data_file.stat().st_mtime:
            self._build_tsv(items_file, data_file, blind)
        if not blind:
            missing = self._missing_videos(bench_root, data_file)
            if missing:
                self._download_videos(bench_root, missing)
            missing = self._missing_videos(bench_root, data_file)
            if missing:
                raise FileNotFoundError(
                    f'{len(missing)} Video-Index videos are missing under {bench_root} (first: {missing[0]}). '
                    f'Download https://huggingface.co/datasets/{self.HF_REPO_ID} into that directory.')
        return dict(root=str(bench_root), data_file=str(data_file))

    # ---------------------------------------------------------------- frames
    def _video_file(self, line):
        return osp.join(self.data_root, line['video_path'])

    def save_video_frames(self, line):
        """Frames of the video protocol for one item, written once as JPEG files."""
        import decord
        vid_path = self._video_file(line)
        vid = decord.VideoReader(vid_path)
        duration = line['duration']
        duration = float(duration) if str(duration) not in ('', 'nan') else None
        if self.fps > 0:
            rate, cap, tag = self.fps, self.max_frames, f'{self.fps:g}fps-cap{self.max_frames}'
        else:
            rate, cap, tag = 1.0, self.nframe, f'1fps-cap{self.nframe}'
        indices = vi.one_fps_indices(len(vid), vid.get_avg_fps(), duration, rate=rate, cap=cap)
        frame_root = osp.join(self.frame_root, str(line['video']))
        os.makedirs(frame_root, exist_ok=True)
        n = len(indices)
        frame_paths = [osp.join(frame_root, f'frame-{i}-of-{n}-{tag}-s{self.short_side}.jpg') for i in range(1, n + 1)]
        if np.all([osp.exists(p) for p in frame_paths]):
            return frame_paths
        lock_path = osp.join(self.frame_root, str(line['video']) + '.lock')
        with portalocker.Lock(lock_path, 'w', timeout=30):
            for frame, pth in zip(vi.read_frames(vid, indices), frame_paths):
                if osp.exists(pth):
                    continue
                img = Image.fromarray(frame)
                w, h = img.size
                if self.short_side and min(w, h) > self.short_side:
                    s = self.short_side / min(w, h)
                    img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)
                img.save(pth, quality=85)
        return frame_paths

    # ---------------------------------------------------------------- prompt
    @staticmethod
    def _candidates(line):
        candidates = line['candidates']
        if isinstance(candidates, str):
            candidates = ast.literal_eval(candidates)
        return [str(c) for c in candidates]

    def build_prompt(self, line, video_llm=False, **kwargs):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        options = vi.render_options(self._candidates(line))
        message = []
        if self.blind:
            intro = vi.INTRO_BLIND
        elif video_llm:
            message.append(dict(type='video', value=self._video_file(line)))
            intro = vi.INTRO_VIDEO
        else:
            frames = self.save_video_frames(line)
            message.extend(dict(type='image', value=frame) for frame in frames)
            intro = vi.INTRO_FRAMES.format(n=len(frames))
        message.append(dict(type='text', value=vi.PROMPT.format(intro=intro, q=line['question'], opts=options)))
        return message

    # -------------------------------------------------------------- evaluate
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        letters, scores = [], []
        for _, row in data.iterrows():
            prediction = row.get('prediction', '')
            prediction = '' if pd.isna(prediction) else str(prediction)
            options = self._candidates(row)
            letters.append(vi.extract_letter(prediction, options) or '')
            scores.append(vi.score_reply(prediction, row['answer'], options))
        data['extracted'] = letters
        data['score'] = scores

        detail_file = get_intermediate_file_path(eval_file, '_judge', 'xlsx')
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(data, detail_file)

        # blind: several rows (option permutations) per item, averaged per item first
        per_item = data.groupby('item_id').agg(score=('score', 'mean'), group=('capability_group', 'first'))
        rating = {'Overall': round(float(per_item['score'].mean()) * 100, 2)}
        for key, label in vi.GROUPS:
            part = per_item[per_item['group'] == key]
            if len(part):  # a slice (`limit`) may hold no item of a group
                rating[label] = round(float(part['score'].mean()) * 100, 2)
        rating['Items'] = int(len(per_item))
        rating['Replies'] = int(len(data))
        rating['Replies without an option'] = int(sum(1 for x in letters if not x))
        dump(rating, score_file)
        return rating

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'Overall' in metrics:
            return {'Overall': metrics['Overall']}
        return {}
