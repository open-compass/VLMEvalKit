import json
import os
import os.path as osp
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from .utils import temporal_grounding as tg
from .video_base import VideoBaseDataset

# Official TimeLens-Bench prompts (https://github.com/TencentARC/TimeLens, evaluation/utils.py).
TIMELENS_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'.")
# Used when the video is given as frames, each preceded by its timestamp.
TIMELENS_PROMPT_FRAMES = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). ") + TIMELENS_PROMPT


def _safe_extract(archive, target):
    target = Path(target).resolve()
    with tarfile.open(archive, 'r:*') as tar:
        members = []
        for member in tar.getmembers():
            dest = (target / member.name).resolve()
            if member.isfile() and str(dest).startswith(str(target) + os.sep):
                members.append(member)
        tar.extractall(target, members=members)


class TimeLensBench(VideoBaseDataset):
    """TimeLens-Bench: manually refined Charades-STA, ActivityNet Captions and QVHighlights for video
    temporal grounding (https://huggingface.co/datasets/TencentARC/TimeLens-Bench).

    Each sample is a (video, query) pair with one ground-truth span. The model answers in free form;
    the first predicted span is scored with temporal IoU (R1@{0.3, 0.5, 0.7} and mIoU), as in the
    official evaluation.
    """

    TYPE = 'Video-Temporal-Grounding'
    DEFAULT_JUDGE_MODEL = None
    HF_REPO_ID = 'TencentARC/TimeLens-Bench'
    SUBSETS = {
        'Charades-TimeLens': 'charades',
        'ActivityNet-TimeLens': 'activitynet',
        'QVHighlights-TimeLens': 'qvhighlights',
    }

    @classmethod
    def supported_datasets(cls):
        return list(cls.SUBSETS)

    def __init__(self, dataset='Charades-TimeLens', pack=False, nframe=0, fps=2.0):
        self._video_meta = {}
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    # ------------------------------------------------------------------ data
    @classmethod
    def _build_tsv(cls, annotation_file, data_file):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)
        records = []
        for video, item in annos.items():
            for query, span in zip(item['queries'], item['spans']):
                records.append(
                    dict(
                        index=len(records),
                        video=video,
                        question=query,
                        answer=json.dumps([float(span[0]), float(span[1])]),
                        duration=float(item['duration']),
                    ))
        tmp = str(data_file) + '.tmp.tsv'
        dump(pd.DataFrame(records), tmp)
        os.replace(tmp, data_file)

    @staticmethod
    def _videos_ready(video_root, data_file):
        data = load(str(data_file))
        return all(osp.exists(osp.join(video_root, f'{v}.mp4')) for v in set(data['video']))

    @classmethod
    def _download_videos(cls, bench_root, subset):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=cls.HF_REPO_ID,
                          repo_type='dataset',
                          allow_patterns=f'video_shards/{subset}/*.tar.gz',
                          local_dir=str(bench_root))
        shard_dir = bench_root / 'video_shards' / subset
        for shard in sorted(shard_dir.glob('*.tar.gz')):
            _safe_extract(shard, bench_root / 'videos')  # shards contain "<subset>/<video>.mp4"

    def prepare_dataset(self, dataset):
        subset = self.SUBSETS[dataset]
        bench_root = Path(LMUDataRoot()) / 'TimeLens-Bench'
        bench_root.mkdir(parents=True, exist_ok=True)

        annotation_file = bench_root / f'{subset}-timelens.json'
        if not annotation_file.exists():
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=self.HF_REPO_ID,
                            repo_type='dataset',
                            filename=annotation_file.name,
                            local_dir=str(bench_root))
        data_file = bench_root / f'{dataset}.tsv'
        if not data_file.exists() or annotation_file.stat().st_mtime > data_file.stat().st_mtime:
            self._build_tsv(annotation_file, data_file)

        video_root = bench_root / 'videos' / subset
        if not video_root.exists() or not self._videos_ready(video_root, data_file):
            self._download_videos(bench_root, subset)
        if not self._videos_ready(video_root, data_file):
            raise FileNotFoundError(f'{dataset} videos are incomplete under {video_root}.')
        return dict(root=str(video_root), data_file=str(data_file))

    # ---------------------------------------------------------------- prompt
    def _frame_times(self, video):
        if video not in self._video_meta:
            import decord
            vr = decord.VideoReader(osp.join(self.data_root, f'{video}.mp4'))
            self._video_meta[video] = (len(vr), vr.get_avg_fps())
        n_frames, video_fps = self._video_meta[video]
        return tg.sampled_frame_times(n_frames, video_fps, fps=self.fps, nframe=self.nframe)

    def build_prompt(self, line, video_llm=False, **kwargs):
        if isinstance(line, int):
            line = self.data.iloc[line]
        video = str(line['video'])
        if video_llm:
            item = dict(type='video', value=osp.join(self.data_root, f'{video}.mp4'))
            if self.fps > 0:
                item['fps'] = self.fps
            elif self.nframe > 0:
                item['nframes'] = self.nframe
            return [item, dict(type='text', value=TIMELENS_PROMPT.format(line['question']))]
        frames = self.save_video_frames(video)
        message = tg.frames_with_timestamps(frames, self._frame_times(video))
        message.append(dict(type='text', value=TIMELENS_PROMPT_FRAMES.format(line['question'])))
        return message

    # ------------------------------------------------------------ evaluation
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        ious, parsed = [], []
        for _, row in data.iterrows():
            gt = row['answer']
            gt = json.loads(gt) if isinstance(gt, str) else list(gt)
            spans = tg.extract_time_spans(row.get('prediction', ''))
            # The official evaluation scores the first predicted span only.
            parsed.append(json.dumps(spans[0]) if spans else '')
            ious.append(tg.temporal_iou(spans[0], gt) if spans else 0.0)
        data['pred_span'] = parsed
        data['iou'] = ious

        metrics = tg.recall_at(ious)
        metrics['mIoU'] = float(np.mean(ious) * 100) if ious else 0.0
        metrics = {k: round(v, 2) for k, v in metrics.items()}
        metrics['num_samples'] = len(ious)
        metrics['num_unparsed'] = int(sum(1 for p in parsed if not p))

        dump(data, get_intermediate_file_path(eval_file, '_judge', 'xlsx'))
        dump(metrics, get_intermediate_file_path(eval_file, '_score', 'json'))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'mIoU' in metrics:
            return {'mIoU': metrics['mIoU']}
        return {}
