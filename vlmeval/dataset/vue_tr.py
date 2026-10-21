import json
import os
import os.path as osp
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import download_file, dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from .utils import temporal_grounding as tg
from .video_base import VideoBaseDataset

VUE_TR_PROMPT = (
    "Find all the moments in the video that match the query: '{query}'. The video is {duration:.1f} seconds long. "
    "Answer with a list of time intervals in seconds, e.g. [[12.0, 30.5], [40.0, 52.0]].")
VUE_TR_PROMPT_FRAMES = (
    "You are given a video with multiple frames. The numbers before each video frame indicate its sampling "
    "timestamp (in seconds). ") + VUE_TR_PROMPT
ATTRIBUTES = {
    'duration_category': ('ultra-short', 'short', 'medium', 'long', 'ultra-long'),
    'query_format': ('keyword', 'phrase', 'sentence'),
    'query_modality': ('vision', 'audio', 'vision+audio'),
}


def _intersection_length(a, b):
    """Overlap of two interval lists, walked together in the given order (as the official script does)."""
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0.0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def vue_tr_query_scores(pred, gt, version=1):
    """Per-query IoU, precision and recall of the official VUE-TR evaluation (bytedance/vidi qa_eval.py).

    Predicted starts are floored and ends ceiled. IoU merges the predicted intervals; precision and
    recall are the overlap divided by the predicted / ground-truth length (None when undefined, so the
    query is left out of that average; in version 2, no prediction for an empty ground truth has
    precision 1). As in the official script, precision and recall keep the intervals in the order the
    model gave them, which reproduces the published leaderboard numbers exactly.
    """
    pred = [[float(np.floor(s)), float(np.ceil(e))] for s, e in pred]
    gt = [[float(min(s, e)), float(max(s, e))] for s, e in gt]
    iou = tg.multi_span_iou(pred, gt)
    pred_pr = [[min(s, e), max(s, e)] for s, e in pred]
    inter = _intersection_length(gt, pred_pr)
    len_gt = sum(e - s for s, e in gt)
    len_pred = sum(e - s for s, e in pred_pr)
    recall = inter / len_gt if len_gt else None
    if len_pred:
        precision = inter / len_pred
    else:
        precision = 1.0 if (version >= 2 and not len_gt) else None
    return iou, precision, recall


def vue_tr_auc(ious, precisions, recalls):
    """Areas under the IoU (IoU > t), precision (>= t) and recall (>= t) curves, t in [0, 1], in %."""
    thresholds = np.linspace(0, 1, 101)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    ious = np.asarray(ious, dtype=float)
    precisions = np.asarray([p for p in precisions if p is not None], dtype=float)
    recalls = np.asarray([r for r in recalls if r is not None], dtype=float)

    def area(values, strict):
        if len(values) == 0:
            return 0.0
        curve = [np.mean(values > t) if strict else np.mean(values >= t) for t in thresholds]
        return float(trapezoid(curve, thresholds) * 100)

    return dict(IoU=area(ious, True), Precision=area(precisions, False), Recall=area(recalls, False))


class VUETR(VideoBaseDataset):
    """VUE-TR and VUE-TR-V2 (https://github.com/bytedance/vidi): temporal retrieval in long videos (up to
    two hours) with keyword, phrase and sentence queries about vision, audio or both.

    The model lists the matching intervals; the official metrics are the areas under the precision,
    recall and IoU curves, overall and per video length, query format and query modality. The
    ``-Vision`` subsets keep only the queries about visual content. The videos are YouTube videos; they are
    downloaded with yt-dlp (``pip install yt-dlp``; ffmpeg is needed to keep the audio) at up to 480p into
    ``$LMUData/VUE-TR/videos/<video_id>.mp4``. Videos that are no longer online are listed in
    ``unavailable_videos.txt`` (delete it to try again) and their queries are skipped with a warning.
    """

    TYPE = 'Video-Temporal-Grounding'
    DEFAULT_JUDGE_MODEL = None
    GT_URL = 'https://raw.githubusercontent.com/bytedance/vidi/main/{folder}/{file}'
    VERSIONS = {
        'VUE-TR': ('VUE_TR', 'VUE-TR_ground_truth.json', 1),
        'VUE-TR-V2': ('VUE_TR_V2', 'VUE-TRv2_ground_truth.json', 2),
    }

    @classmethod
    def supported_datasets(cls):
        return ['VUE-TR', 'VUE-TR-Vision', 'VUE-TR-V2', 'VUE-TR-V2-Vision']

    def __init__(self, dataset='VUE-TR', pack=False, nframe=0, fps=1.0):
        self._video_meta = {}
        self.version = self.VERSIONS[dataset.replace('-Vision', '')][2]
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    def prepare_dataset(self, dataset):
        base = dataset.replace('-Vision', '')
        folder, gt_name, _ = self.VERSIONS[base]
        bench_root = Path(LMUDataRoot()) / 'VUE-TR'
        video_root = bench_root / 'videos'
        video_root.mkdir(parents=True, exist_ok=True)
        gt_file = bench_root / gt_name
        if not gt_file.exists():
            download_file(self.GT_URL.format(folder=folder, file=gt_name), str(gt_file))

        with open(gt_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)
        if dataset.endswith('-Vision'):
            annos = [a for a in annos if a['query_modality'] == 'vision']
        missing = {a['video_id'] for a in annos if not (video_root / f"{a['video_id']}.mp4").exists()}
        unavailable_file = bench_root / 'unavailable_videos.txt'
        to_download = missing - self._read_ids(unavailable_file)
        if to_download:
            self._download_videos(to_download, video_root, unavailable_file)
            missing = {a['video_id'] for a in annos if not (video_root / f"{a['video_id']}.mp4").exists()}
        if missing:
            warnings.warn(f'{dataset}: {len(missing)} of {len({a["video_id"] for a in annos})} videos are missing '
                          f'under {video_root} (no longer online, or yt-dlp is not installed); their queries are '
                          'skipped.')
        records = [
            dict(index=a['query_id'],
                 video=a['video_id'],
                 question=a['query'],
                 answer=json.dumps(a['gt']),
                 duration=float(a['duration']),
                 duration_category=a['duration_category'],
                 query_format=a['query_format'],
                 query_modality=a['query_modality']) for a in annos if a['video_id'] not in missing
        ]
        if not records:
            raise FileNotFoundError(f'No {dataset} videos found under {video_root}.')
        data_file = bench_root / f'{dataset}.tsv'
        # Reuse an up-to-date TSV: run.py builds the dataset on rank 0 first and then on every rank, so the file is
        # only written when it is missing, older than the ground truth, or more videos have been downloaded since.
        if not self._tsv_up_to_date(data_file, gt_file, records):
            tmp = str(data_file) + '.tmp.tsv'
            dump(pd.DataFrame(records), tmp)
            os.replace(tmp, data_file)
        return dict(root=str(video_root), data_file=str(data_file))

    @staticmethod
    def _read_ids(path):
        return set(Path(path).read_text().split()) if Path(path).exists() else set()

    @classmethod
    def _download_videos(cls, video_ids, video_root, unavailable_file):
        """Download YouTube videos as <video_id>.mp4 with yt-dlp; the ones that fail are added to
        ``unavailable_file`` so that later builds do not try them again."""
        try:
            import yt_dlp
        except ImportError:
            warnings.warn(f'{len(video_ids)} VUE-TR videos are missing and yt-dlp is not installed, so they cannot be '
                          f'downloaded. Install it with `pip install yt-dlp` (ffmpeg is recommended), or put the '
                          f'videos in {video_root} as <video_id>.mp4.')
            return
        import shutil

        # H.264 video is preferred: decord, which most video models use, often fails on AV1 and VP9.
        if shutil.which('ffmpeg'):
            fmt = 'bv*[vcodec^=avc1]+ba[ext=m4a]/bv*[vcodec^=avc1]+ba/bv*+ba/b'
        else:
            # Most YouTube videos only have separate video and audio streams, which need ffmpeg to be merged.
            warnings.warn('ffmpeg is not installed, so VUE-TR videos are downloaded without audio (enough for the '
                          '-Vision subsets). Install ffmpeg to also get the audio.')
            fmt = 'b[ext=mp4]/bv*[vcodec^=avc1][ext=mp4]/bv*[ext=mp4]'
        # Up to 480p on the shorter side ("res").
        options = dict(format=fmt, format_sort=['res:480'], merge_output_format='mp4',
                       outtmpl=str(Path(video_root) / '%(id)s.%(ext)s'), ignoreerrors=True, quiet=True,
                       no_warnings=True, noprogress=True)
        with yt_dlp.YoutubeDL(options) as ydl:
            ydl.download([f'https://www.youtube.com/watch?v={vid}' for vid in sorted(video_ids)])
        failed = sorted(v for v in video_ids if not (Path(video_root) / f'{v}.mp4').exists())
        if failed:
            with open(unavailable_file, 'a') as f:
                f.write(''.join(f'{v}\n' for v in failed))

    @staticmethod
    def _tsv_up_to_date(data_file, gt_file, records):
        if not data_file.exists() or gt_file.stat().st_mtime > data_file.stat().st_mtime:
            return False
        try:
            existing = load(str(data_file))
        except Exception:
            return False
        return sorted(map(str, existing['index'])) == sorted(str(r['index']) for r in records)

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
            text = VUE_TR_PROMPT.format(query=line['question'], duration=float(line['duration']))
            return [item, dict(type='text', value=text)]
        frames = self.save_video_frames(video)
        message = tg.frames_with_timestamps(frames, self._frame_times(video))
        text = VUE_TR_PROMPT_FRAMES.format(query=line['question'], duration=float(line['duration']))
        message.append(dict(type='text', value=text))
        return message

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        scores, parsed = [], []
        for _, row in data.iterrows():
            gt = row['answer']
            gt = json.loads(gt) if isinstance(gt, str) else list(gt)
            spans = tg.extract_time_spans(row.get('prediction', ''))  # reversed spans handled as in qa_eval.py
            parsed.append(json.dumps(spans))
            scores.append(vue_tr_query_scores(spans, gt, self.version))
        data['pred_spans'] = parsed
        data['iou'] = [s[0] for s in scores]
        data['precision'] = [s[1] for s in scores]
        data['recall'] = [s[2] for s in scores]

        def summary(mask):
            sub = [s for s, m in zip(scores, mask) if m]
            res = vue_tr_auc(*zip(*sub)) if sub else {}
            res = {k: round(v, 2) for k, v in res.items()}
            res['num_samples'] = len(sub)
            return res

        metrics = {'overall': summary([True] * len(data))}
        for column, values in ATTRIBUTES.items():
            for value in values:
                mask = (data[column] == value).tolist()
                if any(mask):
                    metrics[value] = summary(mask)
        metrics['overall']['num_unparsed'] = int(sum(1 for p in parsed if p == '[]'))

        dump(data, get_intermediate_file_path(eval_file, '_judge', 'xlsx'))
        dump(metrics, get_intermediate_file_path(eval_file, '_score', 'json'))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        # The run summary passes the metrics flattened ('overall|IoU'); evaluate() returns them nested.
        if not isinstance(metrics, dict):
            return {}
        if isinstance(metrics.get('overall'), dict):
            return {'IoU': metrics['overall']['IoU']}
        if 'overall|IoU' in metrics:
            return {'IoU': metrics['overall|IoU']}
        return {}
