import json
import os
import os.path as osp
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from .utils import temporal_grounding as tg
from .video_base import VideoBaseDataset

ANNOTATION_URL = 'https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl'
VIDEO_URL = 'https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz'
CLIP_LENGTH = 2  # QVHighlights annotates saliency per 2-second clip

MR_PROMPT = ("Find every moment in the video that matches the query: '{}'. "
             'List all matching time spans in seconds, the most relevant first, '
             'in the format [[start, end], [start, end], ...].')
# Zero-shot highlight detection prompt of TimeChat (https://github.com/RenShuhuai-Andy/TimeChat,
# prompts/vhd_description_zeroshot.txt), also used by later video LLM papers on QVHighlights.
HD_PROMPT = (
    'You are given a video from the QVHighlights dataset. Please find the highlight contents in the video '
    'described by a sentence query, determining the highlight timestamps and its saliency score on a scale '
    "from 1 to 5. The output format should be like: 'The highlight timestamps are in the 82, 84, 86, 88, 90, "
    "92, 94, 96, 98, 100 seconds. Their saliency scores are 1.3, 1.7, 1.7, 1.7, 1.7, 1.3, 1.7, 2.3, 2.3, 2.3'. "
    'Now I will give you the sentence query: {}. Please return the query-based highlight timestamps and '
    'salient scores.')
FRAMES_PREFIX = ('You are given a video with multiple frames. '
                 'The numbers before each video frame indicate its sampling timestamp (in seconds). ')

MR_IOU_THRESHOLDS = [round(t, 2) for t in np.linspace(0.5, 0.95, 10)]
MR_LENGTH_RANGES = {'short': (0, 10), 'middle': (10, 30), 'long': (30, 150), 'full': (0, 150)}
HD_MIN_SCORES = {'Fair': 2, 'Good': 3, 'VeryGood': 4}

_CLOCK = re.compile(r'\b(?:(\d{1,2}):)?(\d{1,2}):(\d{2}(?:\.\d+)?)\b')
_NUMBER = re.compile(r'\d+(?:\.\d+)?')


# ------------------------------------------------------------------ parsing
def parse_windows(answer, max_windows=10):
    """Predicted moments as a ranked list [(start, end), ...]; reversed spans are swapped."""
    spans = tg.extract_time_spans(answer)
    return [(min(s, e), max(s, e)) for s, e in spans][:max_windows]


def _clock_to_seconds(match):
    hours, minutes, seconds = match.groups()
    return str(int(hours or 0) * 3600 + int(minutes) * 60 + float(seconds))


def parse_highlights(answer):
    """Read "timestamps ... scores ..." answers into ([t1, t2, ...], [s1, s2, ...]).

    Numbers before the first "score" are timestamps and numbers after it are their saliency scores. If no
    score is given every timestamp gets 1; missing scores at the end count as 0.
    """
    text = _CLOCK.sub(_clock_to_seconds, str(answer).lower())
    cut = text.find('score')
    time_part, score_part = (text, '') if cut < 0 else (text[:cut], text[cut:])
    times = [float(x) for x in _NUMBER.findall(time_part)]
    scores = [float(x) for x in _NUMBER.findall(score_part)]
    if not scores:
        scores = [1.0] * len(times)
    scores = (scores + [0.0] * len(times))[:len(times)]
    return times, scores


def highlights_to_clip_scores(times, scores, duration):
    """Per-clip saliency (clip i covers [2i, 2i + 2) s); several scores in one clip are averaged."""
    num_clips = int(duration / CLIP_LENGTH)
    total, count = np.zeros(num_clips), np.zeros(num_clips)
    for t, s in zip(times, scores):
        if 0 <= t < num_clips * CLIP_LENGTH:
            total[int(t // CLIP_LENGTH)] += s
            count[int(t // CLIP_LENGTH)] += 1
    return np.divide(total, count, out=np.zeros(num_clips), where=count > 0)


# --------------------------------------------------------- moment retrieval
def _iou_matrix(spans1, spans2):
    spans1, spans2 = np.asarray(spans1, dtype=float), np.asarray(spans2, dtype=float)
    inter = np.clip(
        np.minimum(spans1[:, None, 1], spans2[None, :, 1]) - np.maximum(spans1[:, None, 0], spans2[None, :, 0]), 0,
        None)
    union = (spans1[:, 1] - spans1[:, 0])[:, None] + (spans2[:, 1] - spans2[:, 0])[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _voc_ap(precision, recall):
    """Area under the precision-recall curve with VOC 2011 interpolation."""
    precision = np.concatenate([[0], precision, [0]])
    recall = np.concatenate([[0], recall, [1]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(np.sum((recall[idx] - recall[idx - 1]) * precision[idx]))


def window_ap(pred_windows, gt_windows, thresholds=MR_IOU_THRESHOLDS):
    """Average precision of ranked predicted windows for one query, at each IoU threshold. Each
    ground-truth window can be matched once; a prediction is matched to the free window it overlaps most."""
    ap = np.zeros(len(thresholds))
    if not pred_windows or not gt_windows:
        return ap
    ious = _iou_matrix(pred_windows, gt_windows)
    for t_idx, thr in enumerate(thresholds):
        taken = np.zeros(len(gt_windows), dtype=bool)
        tp = np.zeros(len(pred_windows))
        for p_idx in range(len(pred_windows)):
            for g_idx in ious[p_idx].argsort()[::-1]:
                if ious[p_idx, g_idx] < thr:
                    break
                if not taken[g_idx]:
                    taken[g_idx] = True
                    tp[p_idx] = 1
                    break
        tp_cum = np.cumsum(tp)
        ap[t_idx] = _voc_ap(tp_cum / np.arange(1, len(tp) + 1), tp_cum / len(gt_windows))
    return ap


def moment_retrieval_metrics(preds, gts):
    """Official QVHighlights moment retrieval metrics (mAP over IoU 0.5:0.05:0.95 with up to 10 ranked
    windows, and R1 at IoU 0.5 / 0.7), overall and by ground-truth window length. ``preds`` and ``gts`` are
    lists with the ranked predicted windows and the ground-truth windows of each query. Queries without a
    predicted window score 0."""
    out = {}
    for name, (lo, hi) in MR_LENGTH_RANGES.items():
        aps, ious = [], []
        for pred, gt in zip(preds, gts):
            if name != 'full':
                gt = [w for w in gt if lo < w[1] - w[0] <= hi]
                if not gt:
                    continue
            aps.append(window_ap(pred, gt))
            ious.append(float(_iou_matrix(pred[:1], gt).max()) if pred else 0.0)
        ap = np.mean(aps, axis=0) if aps else np.zeros(len(MR_IOU_THRESHOLDS))
        out[name] = {
            'mAP': float(np.mean(ap) * 100),
            'mAP@0.5': float(ap[0] * 100),
            'mAP@0.75': float(ap[5] * 100),
            'R1@0.5': float(np.mean(np.array(ious) >= 0.5) * 100) if ious else 0.0,
            'R1@0.7': float(np.mean(np.array(ious) >= 0.7) * 100) if ious else 0.0,
            'num_queries': len(aps),
        }
    return out


# ------------------------------------------------------- highlight detection
def gt_clip_scores(duration, relevant_clip_ids, saliency_scores):
    """(#clips, 3) saliency of every clip from the 3 annotators (0 outside the relevant clips)."""
    scores = np.zeros((int(duration / CLIP_LENGTH), 3))
    scores[np.asarray(relevant_clip_ids, dtype=int)] = np.asarray(saliency_scores, dtype=float)
    return scores


def _binary_ap(y_true, y_score):
    """AP as computed by the official QVHighlights script (precision made monotone along sklearn's
    precision-recall curve, averaged where recall changes)."""
    from sklearn.metrics import precision_recall_curve
    if len(set(y_true)) == 1:
        return float(y_true[0] != 0)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    recall = recall.astype(np.float32)
    precision = np.maximum.accumulate(precision)
    return float(np.mean(precision[np.where(np.diff(recall))]))


def highlight_metrics(pred_scores, gt_scores):
    """Official QVHighlights highlight detection metrics: HIT@1 and mAP, with a clip counted as a highlight
    when an annotator gave it at least 2 (Fair), 3 (Good) or 4 (VeryGood). ``pred_scores`` and
    ``gt_scores`` hold per query the predicted (#clips,) and annotated (#clips, 3) clip scores."""
    out = {}
    for name, min_score in HD_MIN_SCORES.items():
        hits, aps = [], []
        for pred, gt in zip(pred_scores, gt_scores):
            gt = (gt >= min_score).astype(float)
            pred = np.asarray(pred, dtype=float)
            pred = np.concatenate([pred, np.zeros(max(0, len(gt) - len(pred)))])[:len(gt)]
            hits.append(gt[int(np.argmax(pred))].max())
            aps.extend(_binary_ap(gt[:, k], pred) for k in range(gt.shape[1]))
        out[name] = {'mAP': float(np.mean(aps) * 100), 'HIT@1': float(np.mean(hits) * 100)}
    return out


class QVHighlights(VideoBaseDataset):
    """QVHighlights (Lei et al., NeurIPS 2021) validation set: 1,550 queries over 1,519 YouTube clips of
    150 s, with all the matching moments of each query and a 1-4 saliency score per 2-second clip from three
    annotators (https://github.com/jayleicn/moment_detr).

    Two tasks, scored with the metrics of the official evaluation script:
    - ``QVHighlights-MR`` (moment retrieval): the model lists the matching moments, most relevant first.
      mAP over IoU 0.5:0.05:0.95 and R1@0.5 / 0.7.
    - ``QVHighlights-HD`` (highlight detection): the model lists highlight timestamps with saliency scores
      (TimeChat's prompt). mAP and HIT@1, mainly with the "VeryGood" (score >= 4) threshold.
    """

    TYPE = 'Video-Temporal-Grounding'
    DEFAULT_JUDGE_MODEL = None

    @classmethod
    def supported_datasets(cls):
        return ['QVHighlights-MR', 'QVHighlights-HD']

    def __init__(self, dataset='QVHighlights-MR', pack=False, nframe=0, fps=2.0):
        self._video_meta = {}
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    # ------------------------------------------------------------------ data
    @staticmethod
    def _build_tsv(annotation_file, data_file):
        records = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                records.append(
                    dict(
                        index=len(records),
                        qid=item['qid'],
                        video=item['vid'],
                        question=item['query'].strip(),
                        duration=float(item['duration']),
                        answer=json.dumps(item['relevant_windows']),
                        relevant_clip_ids=json.dumps(item['relevant_clip_ids']),
                        saliency_scores=json.dumps(item['saliency_scores']),
                    ))
        tmp = str(data_file) + '.tmp.tsv'
        dump(pd.DataFrame(records), tmp)
        os.replace(tmp, data_file)

    @staticmethod
    def _missing_videos(video_root, data_file):
        videos = set(load(str(data_file))['video'])
        return {v for v in videos if not osp.exists(osp.join(video_root, f'{v}.mp4'))}

    @staticmethod
    def _download_videos(video_root, wanted):
        """Stream the official archive (all splits, about 144 GB) and keep only the wanted clips."""
        import requests
        with requests.get(VIDEO_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            response.raw.decode_content = True
            with tarfile.open(fileobj=response.raw, mode='r|gz') as tar:
                for member in tar:
                    name = osp.basename(member.name)
                    if not member.isfile() or osp.splitext(name)[0] not in wanted:
                        continue
                    src = tar.extractfile(member)
                    tmp = video_root / (name + '.tmp')
                    with open(tmp, 'wb') as dst:
                        while True:
                            chunk = src.read(1 << 20)
                            if not chunk:
                                break
                            dst.write(chunk)
                    os.replace(tmp, video_root / name)

    def prepare_dataset(self, dataset):
        root = Path(LMUDataRoot()) / 'QVHighlights'
        root.mkdir(parents=True, exist_ok=True)
        annotation_file = root / 'highlight_val_release.jsonl'
        if not annotation_file.exists():
            import requests
            response = requests.get(ANNOTATION_URL, timeout=60)
            response.raise_for_status()
            annotation_file.write_bytes(response.content)
        data_file = root / 'QVHighlights.tsv'
        if not data_file.exists() or annotation_file.stat().st_mtime > data_file.stat().st_mtime:
            self._build_tsv(annotation_file, data_file)

        video_root = root / 'videos'
        video_root.mkdir(exist_ok=True)
        missing = self._missing_videos(video_root, data_file)
        if missing:
            self._download_videos(video_root, missing)
            missing = self._missing_videos(video_root, data_file)
        if missing:
            raise FileNotFoundError(f'{len(missing)} QVHighlights videos are missing under {video_root}.')
        return dict(root=str(video_root), data_file=str(data_file))

    # ---------------------------------------------------------------- prompt
    @property
    def task(self):
        return 'HD' if self.dataset_name.endswith('-HD') else 'MR'

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
        query = str(line['question']).strip('.')
        prompt = (HD_PROMPT if self.task == 'HD' else MR_PROMPT).format(query)
        if video_llm:
            item = dict(type='video', value=osp.join(self.data_root, f'{video}.mp4'))
            if self.fps > 0:
                item['fps'] = self.fps
            elif self.nframe > 0:
                item['nframes'] = self.nframe
            return [item, dict(type='text', value=prompt)]
        frames = self.save_video_frames(video)
        message = tg.frames_with_timestamps(frames, self._frame_times(video))
        message.append(dict(type='text', value=FRAMES_PREFIX + prompt))
        return message

    # ------------------------------------------------------------ evaluation
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        answers = [str(a) if a is not None else '' for a in data['prediction']]
        if self.task == 'MR':
            preds = [parse_windows(a) for a in answers]
            gts = [json.loads(g) if isinstance(g, str) else list(g) for g in data['answer']]
            data['pred_windows'] = [json.dumps(p) for p in preds]
            results = moment_retrieval_metrics(preds, gts)
            metrics = {k: round(v, 2) for k, v in results['full'].items() if k != 'num_queries'}
            for name in ('short', 'middle', 'long'):
                metrics[f'mAP_{name}'] = round(results[name]['mAP'], 2)
            metrics['num_unparsed'] = int(sum(1 for p in preds if not p))
        else:
            pred_scores, gt_scores, unparsed = [], [], 0
            for answer, (_, row) in zip(answers, data.iterrows()):
                times, scores = parse_highlights(answer)
                unparsed += not times
                pred_scores.append(highlights_to_clip_scores(times, scores, row['duration']))
                gt_scores.append(
                    gt_clip_scores(row['duration'], json.loads(row['relevant_clip_ids']),
                                   json.loads(row['saliency_scores'])))
            data['pred_clip_scores'] = [json.dumps(np.round(s, 3).tolist()) for s in pred_scores]
            results = highlight_metrics(pred_scores, gt_scores)
            metrics = {
                'mAP': round(results['VeryGood']['mAP'], 2),
                'HIT@1': round(results['VeryGood']['HIT@1'], 2),
            }
            for name in ('Fair', 'Good'):
                metrics[f'mAP_{name}'] = round(results[name]['mAP'], 2)
                metrics[f'HIT@1_{name}'] = round(results[name]['HIT@1'], 2)
            metrics['num_unparsed'] = unparsed
        metrics['num_samples'] = len(data)

        dump(data, get_intermediate_file_path(eval_file, '_judge', 'xlsx'))
        dump(metrics, get_intermediate_file_path(eval_file, '_score', 'json'))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if isinstance(metrics, dict) and 'mAP' in metrics:
            return {'mAP': metrics['mAP']}
        return {}
