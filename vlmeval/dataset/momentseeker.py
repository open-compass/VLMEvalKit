import io
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

# Official MomentSeeker prompts for generation-based MLLMs (arXiv 2502.12558, Appendix 6.1.2).
_FORMAT = ('Format: [[start_1, end_1], …, [start_n, end_n]], where 1≤n≤5. '
           'Examples: Single interval: [[0.2, 7.8]] Multiple intervals: [[0, 10.3], [65.4, 67.3]] ')
_RULES = ('IMPORTANT: 1. Return only the list of relevant intervals in Video 1. '
          '2. Do not return more than 5 intervals.')
_TMR = 'Identify the most relevant time interval(s) in the video that match the given query or caption. '
_IMR = 'Identify one or more time intervals in the video that match the given query paired with an image. '
_VMR = 'Identify one or more time intervals in the video that match the given query paired with a reference video. '
MOMENTSEEKER_PROMPTS = {
    'TMR': _TMR + _FORMAT + 'Now, here is the textual query: {} ' + _RULES,
    'IMR': _IMR + _FORMAT + 'Here is the textual query: {} and the accompanying image: Image 1. ' + _RULES,
    'VMR': _VMR + _FORMAT + 'Here is the textual query: {} and the accompanying video: Video 2. ' + _RULES,
}
TIME_INSTRUCTION = ('Video 1 lasts for {:.2f} seconds, and {} frames are uniformly sampled from it. These frames '
                    'correspond to the following timestamps: {}. Please answer the following questions based on '
                    'this video.')
TIME_INSTRUCTION_VIDEO = 'Video 1 lasts for {:.2f} seconds. Please answer the following questions based on this video.'
ANNOTATION_FILES = {'TMR': 't2v.json', 'IMR': 'ti2v.json', 'VMR': 'tv2v.json'}


class _ChainedReader(io.RawIOBase):
    """Read several files (the parts of a split archive) as one stream."""

    def __init__(self, paths):
        self._paths = list(paths)
        self._file = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self._file is None:
                if not self._paths:
                    return 0
                self._file = open(self._paths.pop(0), 'rb')
            n = self._file.readinto(buffer)
            if n:
                return n
            self._file.close()
            self._file = None


def _safe_extract_stream(paths, target):
    """Extract a (possibly split) .tar.gz archive, skipping members that would leave ``target``."""
    target = Path(target).resolve()
    with tarfile.open(fileobj=io.BufferedReader(_ChainedReader(paths), 1 << 20), mode='r|gz') as tar:
        for member in tar:
            dest = (target / member.name).resolve()
            if member.isfile() and str(dest).startswith(str(target) + os.sep):
                tar.extract(member, target)


class MomentSeeker(VideoBaseDataset):
    """MomentSeeker: moment retrieval in long videos (avg. > 500 s) with text, image-conditioned and
    video-conditioned queries (https://huggingface.co/datasets/avery00/MomentSeeker).

    The model lists up to 5 intervals ``[[start, end], ...]`` in order of relevance. Following the paper,
    R@1 counts a query as solved when the first interval has IoU > 0.3 with any ground-truth moment, and
    mAP@5 averages, over queries, the precision at each correct interval among the first 5 (each ground
    truth matched at most once).
    """

    TYPE = 'Video-Temporal-Grounding'
    DEFAULT_JUDGE_MODEL = None
    HF_REPO_ID = 'avery00/MomentSeeker'
    IOU_THRESHOLD = 0.3
    SUBSETS = {
        'MomentSeeker': ('TMR', 'IMR', 'VMR'),
        'MomentSeeker-TMR': ('TMR', ),
        'MomentSeeker-IMR': ('IMR', ),
        'MomentSeeker-VMR': ('VMR', ),
    }
    QUERY_CLIP_FRAMES = 8  # frames shown for the reference clip of VMR queries to frame-list models

    @classmethod
    def supported_datasets(cls):
        return list(cls.SUBSETS)

    def __init__(self, dataset='MomentSeeker', pack=False, nframe=0, fps=1.0):
        self._video_meta = {}
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    # ------------------------------------------------------------------ data
    def _hf_download(self, bench_root, **kwargs):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=self.HF_REPO_ID, repo_type='dataset', local_dir=str(bench_root), **kwargs)

    def _build_tsv(self, bench_root, data_file):
        import decord
        records, durations = [], {}
        for query_type in ('TMR', 'IMR', 'VMR'):
            with open(bench_root / ANNOTATION_FILES[query_type], 'r', encoding='utf-8') as f:
                annos = json.load(f)
            for item in annos:
                video = osp.splitext(osp.basename(item['src_video_path']))[0]
                if video not in durations:
                    vr = decord.VideoReader(str(bench_root / 'videos' / f'{video}.mp4'))
                    durations[video] = len(vr) / vr.get_avg_fps()
                records.append(
                    dict(
                        index=len(records),
                        video=video,
                        question=item['qry_text'],
                        query_type=query_type,
                        query_image=item.get('qry_img_path') or '',
                        query_video=item.get('qry_video_path') or '',
                        task=item['task'],
                        answer=json.dumps([[float(s), float(e)] for s, e in item['answering_time_interval']]),
                        duration=round(durations[video], 3),
                    ))
        tmp = str(data_file) + '.tmp.tsv'
        dump(pd.DataFrame(records), tmp)
        os.replace(tmp, data_file)

    def prepare_dataset(self, dataset):
        bench_root = Path(LMUDataRoot()) / 'MomentSeeker'
        bench_root.mkdir(parents=True, exist_ok=True)
        if not all((bench_root / f).exists() for f in ANNOTATION_FILES.values()):
            self._hf_download(bench_root, allow_patterns=['*.json'])
        if not (bench_root / 'videos').exists():
            self._hf_download(bench_root, allow_patterns=['videos.tar.gz.part_*'])
            parts = sorted(bench_root.glob('videos.tar.gz.part_*'))
            _safe_extract_stream(parts, bench_root)  # archive contains "videos/<video>.mp4"
        for folder in ('query_images', 'query_videos'):
            if not (bench_root / folder).exists():
                self._hf_download(bench_root, allow_patterns=[f'{folder}.tar.gz'])
                _safe_extract_stream([bench_root / f'{folder}.tar.gz'], bench_root)

        data_file = bench_root / 'MomentSeeker.tsv'
        if not data_file.exists():
            self._build_tsv(bench_root, data_file)
        videos = set(load(str(data_file))['video'])
        missing = [v for v in videos if not (bench_root / 'videos' / f'{v}.mp4').exists()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} MomentSeeker videos are missing under {bench_root / "videos"}, '
                                    f'e.g. {missing[:3]}')
        self.bench_root = str(bench_root)
        subset_file = bench_root / f'{dataset}.tsv'
        if dataset != 'MomentSeeker' and not subset_file.exists():
            data = load(str(data_file))
            data = data[data['query_type'].isin(self.SUBSETS[dataset])].reset_index(drop=True)
            dump(data, str(subset_file))
        return dict(root=str(bench_root / 'videos'),
                    data_file=str(subset_file if dataset != 'MomentSeeker' else data_file))

    # ---------------------------------------------------------------- prompt
    def _frame_times(self, video):
        if video not in self._video_meta:
            import decord
            vr = decord.VideoReader(osp.join(self.data_root, f'{video}.mp4'))
            self._video_meta[video] = (len(vr), vr.get_avg_fps())
        n_frames, video_fps = self._video_meta[video]
        return tg.sampled_frame_times(n_frames, video_fps, fps=self.fps, nframe=self.nframe)

    def _query_clip_frames(self, clip_path):
        """Uniformly sampled frames of a reference clip, cached next to the other extracted frames."""
        import decord
        from PIL import Image
        name = osp.splitext(clip_path.replace('./', '').replace('/', '__'))[0]
        root = osp.join(self.frame_root, '_query_clips', name)
        paths = [
            osp.join(root, f'frame-{i + 1}-of-{self.QUERY_CLIP_FRAMES}.jpg') for i in range(self.QUERY_CLIP_FRAMES)
        ]
        if not all(osp.exists(p) for p in paths):
            os.makedirs(root, exist_ok=True)
            vr = decord.VideoReader(osp.normpath(osp.join(self.bench_root, clip_path)))
            step = len(vr) / (self.QUERY_CLIP_FRAMES + 1)
            for i, p in enumerate(paths):
                Image.fromarray(vr[int((i + 1) * step)].asnumpy()).save(p)
        return paths

    def build_prompt(self, line, video_llm=False, **kwargs):
        if isinstance(line, int):
            line = self.data.iloc[line]
        video, query_type = str(line['video']), str(line['query_type'])
        duration = float(line['duration'])
        if video_llm:
            item = dict(type='video', value=osp.join(self.data_root, f'{video}.mp4'))
            if self.fps > 0:
                item['fps'] = self.fps
            elif self.nframe > 0:
                item['nframes'] = self.nframe
            message = [item, dict(type='text', value=TIME_INSTRUCTION_VIDEO.format(duration))]
        else:
            frames, times = self.save_video_frames(video), self._frame_times(video)
            message = [dict(type='image', value=f) for f in frames]
            stamps = ', '.join(f'{t:.2f}' for t in times)
            message.append(dict(type='text', value=TIME_INSTRUCTION.format(duration, len(frames), stamps)))
        if query_type == 'IMR':
            message.append(dict(type='text', value='Image 1:'))
            message.append(dict(type='image', value=osp.normpath(osp.join(self.bench_root, str(line['query_image'])))))
        elif query_type == 'VMR':
            clip = str(line['query_video'])
            message.append(dict(type='text', value='Video 2:'))
            if video_llm:
                message.append(dict(type='video', value=osp.normpath(osp.join(self.bench_root, clip))))
            else:
                message.extend(dict(type='image', value=f) for f in self._query_clip_frames(clip))
        message.append(dict(type='text', value=MOMENTSEEKER_PROMPTS[query_type].format(line['question'])))
        return message

    # ------------------------------------------------------------ evaluation
    @classmethod
    def score_one(cls, spans, gts, threshold=None):
        """R@1 and AP@5 of one ranked prediction list against the ground-truth moments."""
        threshold = cls.IOU_THRESHOLD if threshold is None else threshold
        preds = spans[:5]
        r1 = float(bool(preds) and max(tg.temporal_iou(preds[0], g) for g in gts) > threshold)
        matched, hits, precisions = set(), 0, []
        for rank, pred in enumerate(preds, start=1):
            candidates = [(tg.temporal_iou(pred, g), j) for j, g in enumerate(gts) if j not in matched]
            if candidates:
                best, j = max(candidates)
                if best > threshold:
                    matched.add(j)
                    hits += 1
                    precisions.append(hits / rank)
        return r1, float(np.mean(precisions)) if precisions else 0.0

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        r1s, aps, r1_05, r1_07, mious, parsed = [], [], [], [], [], []
        for _, row in data.iterrows():
            gts = row['answer']
            gts = json.loads(gts) if isinstance(gts, str) else list(gts)
            spans = tg.extract_time_spans(row.get('prediction', ''))
            r1, ap = self.score_one(spans, gts)
            r1s.append(r1)
            aps.append(ap)
            r1_05.append(self.score_one(spans, gts, 0.5)[0])
            r1_07.append(self.score_one(spans, gts, 0.7)[0])
            mious.append(tg.multi_span_iou(spans[:5], gts))
            parsed.append(json.dumps(spans[:5]))
        data['pred_spans'] = parsed
        data['R@1'], data['AP@5'], data['mIoU'] = r1s, aps, mious

        def summary(mask):
            n = int(mask.sum())
            if n == 0:
                return {}
            return {
                'R@1': round(float(np.mean(np.array(r1s)[mask]) * 100), 2),
                'mAP@5': round(float(np.mean(np.array(aps)[mask]) * 100), 2),
                'R@1 (IoU>0.5)': round(float(np.mean(np.array(r1_05)[mask]) * 100), 2),
                'R@1 (IoU>0.7)': round(float(np.mean(np.array(r1_07)[mask]) * 100), 2),
                'mIoU': round(float(np.mean(np.array(mious)[mask]) * 100), 2),
                'num_samples': n,
            }

        metrics = {'overall': summary(np.ones(len(data), bool))}
        for query_type in ('TMR', 'IMR', 'VMR'):
            if (data['query_type'] == query_type).any():
                metrics[query_type] = summary((data['query_type'] == query_type).values)
        for task in sorted(data['task'].unique()):
            metrics[f'task/{task}'] = summary((data['task'] == task).values)
        metrics['overall']['num_unparsed'] = int(sum(1 for p in parsed if p == '[]'))

        dump(data, get_intermediate_file_path(eval_file, '_judge', 'xlsx'))
        dump(metrics, get_intermediate_file_path(eval_file, '_score', 'json'))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        # The run summary passes the metrics flattened ('overall|R@1'); evaluate() returns them nested.
        if not isinstance(metrics, dict):
            return {}
        if isinstance(metrics.get('overall'), dict):
            return {'R@1': metrics['overall']['R@1']}
        if 'overall|R@1' in metrics:
            return {'R@1': metrics['overall|R@1']}
        return {}
