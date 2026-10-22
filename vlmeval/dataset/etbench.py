import json
import os
import os.path as osp
import random
import re
import string
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, load
from vlmeval.smp.file import LMUDataRoot
from .utils import temporal_grounding as tg
from .video_base import VideoBaseDataset

IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
# Headline metrics of E.T. Bench, in the order of the paper's main table: (task, metric).
HEADLINE = [('rar', 'Acc'), ('eca', 'Acc'), ('rvq', 'Acc'), ('tvg', 'F1'), ('epm', 'F1'), ('tal', 'F1'), ('evs', 'F1'),
            ('vhd', 'F1'), ('dvc', 'F1'), ('dvc', 'Sim'), ('slc', 'F1'), ('slc', 'Sim'), ('tem', 'Rec'), ('gvq', 'Rec')]
FRAMES_PREFIX = ('You are given a video with multiple frames. '
                 'The numbers before each video frame indicate its sampling timestamp (in seconds). ')

_RANGE = re.compile(r'(\d+\.*\d*)\s*-\s*(\d+\.*\d*)')
_RANGE_STR = re.compile(r'(\d+\.*\d*\s*-\s*\d+\.*\d*)')
_TO = re.compile(r'(\d+\.*\d*)\s* to \s*(\d+\.*\d*)')
_FIRST_NUMBER = re.compile(r'\b(\d+\.\d+\b|\b\d+)\b')
_KEYWORDS = ('starts', 'ends', 'happens in', 'start time', 'end time', 'start', 'end', 'happen')
_OPTION_LETTERS = 'ABCDE'


# ----------------------------------------------------------------- parsing
# The parsing rules follow the official E.T. Bench evaluation (evaluation/compute_metrics.py), so that the
# scores are comparable with published E.T. Bench results.
def _floats(pairs):
    out = []
    for a, b in pairs:
        try:
            out.append((float(a), float(b)))
        except ValueError:
            continue
    return out


def parse_spans(answer):
    """Time spans [(start, end), ...] in an answer, or None. "a - b" ranges anywhere in the answer come
    first, then "a to b"; otherwise the first number of each sentence that mentions a start, an end or a
    "happen" is read, and these numbers are paired in order. Reversed spans are swapped."""
    text = str(answer).lower()
    candidates = [s for s in re.split(r'[!?\n]', text) if any(k in s for k in _KEYWORDS)]
    spans = _floats(_RANGE.findall(text)) or _floats(_TO.findall(text))
    if not spans:
        times = []
        for sentence in candidates:
            found = _FIRST_NUMBER.findall(sentence)
            if found:
                times.append(float(found[0]))
        spans = [(times[i], times[i + 1]) for i in range(0, len(times) // 2 * 2, 2)]
    return [(min(s, e), max(s, e)) for s, e in spans] or None


def _time_part_to_range(time_part, radius=20):
    """Read the time of one "<time> seconds, <caption>" line; a single time t becomes [t - 20, t + 20]."""
    ranges = re.findall(r'\d+\.*\d*\s*-\s*\d+\.*\d*', time_part)
    if not ranges:
        clocks = re.findall(r'\d+\.*\d*:\d+\.*\d*', time_part)
        if time_part.count(':') == 1:
            minutes, seconds = clocks[0].split(':')
            t = int(minutes) * 60 + int(seconds)
            ranges = [f'{t - radius} - {t + radius}' if t > radius else f'{t} - {t + 2 * radius}']
        elif time_part.count(':') == 2:
            start, end = clocks
            start = int(start.split(':')[0]) * 60 + int(start.split(':')[1])
            end = int(end.split(':')[0]) * 60 + int(end.split(':')[1])
            ranges = [f'{start} - {end}']
    if not ranges:
        numbers = re.findall(r'\d+\.*\d*(?!\.)', time_part)
        if len(numbers) == 1:
            t = float(numbers[0])
            ranges = [f'{t - radius} - {t + radius}' if t > radius else f'{t} - {t + 2 * radius}']
        elif len(numbers) == 2:
            ranges = [f'{numbers[0]} - {numbers[1]}']
    return ranges


def _paragraph_events(paragraph):
    text = paragraph.lower()
    timestamps, captions = [], []
    found = _RANGE.findall(text)
    if found:
        timestamps = [[float(a), float(b)] for a, b in found]
        rest = text
        for time_string in _RANGE_STR.findall(text):
            rest = rest.replace(time_string, '\n')
        captions = rest.replace('seconds', '').split('\n')
    if not timestamps:
        starts = re.findall(r'(?:start(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)', text, re.DOTALL | re.IGNORECASE)
        ends = re.findall(r'(?:end(?:ing)? time: (\d+\.*\d*)(?:s| seconds)?)', text, re.DOTALL | re.IGNORECASE)
        if starts and ends:
            timestamps = [[float(s), float(e)] for s, e in zip(starts, ends)]
            captions = re.findall(r'description: (.*)', text) or re.findall(r'\*\s*(.*)', text)
    if not timestamps and re.search(r'start time (\d+\.*\d*), end time (\d+\.*\d*)', text):
        raise ValueError('unsupported "start time x, end time y" format')  # rejected by the official script too
    captions = [c.strip().strip(', ').rstrip() for c in captions if len(c) > 5]
    n = min(len(timestamps), len(captions))
    return timestamps[:n], captions[:n]


def parse_events(answer):
    """Dense captions "<start> - <end> seconds, <caption>" as ([[start, end], ...], [caption, ...]), or
    (None, None) if nothing could be read."""
    answer = str(answer)
    try:
        timestamps, captions = _paragraph_events(answer)
    except Exception:
        return None, None
    if not timestamps:
        if '\n' in answer:
            lines = [c for c in answer.split('\n') if len(c) > 7]
        else:
            lines = [c + '.' for c in answer.split('.') if len(c) > 7]
        for line in lines:
            if timestamps:
                break
            try:
                parts = [p.strip(',') for p in line.split('seconds')]
                ranges = _time_part_to_range(parts[0])
                if not ranges:
                    continue
                start, end = ranges[0].split('-')[:2]
                timestamps.append([round(float(start.strip()), 2), round(float(end.strip()), 2)])
                captions.append(parts[-1].strip())
            except Exception:
                continue
    if not timestamps:
        return None, None
    return [[min(t), max(t)] for t in timestamps], captions


# ----------------------------------------------------------------- scoring
def _iou(windows1, windows2):
    """IoU matrix in float32, as computed by the official script (nncore.ops.temporal_iou)."""
    w1 = np.asarray(windows1, dtype=np.float32).reshape(-1, 2)
    w2 = np.asarray(windows2, dtype=np.float32).reshape(-1, 2)
    inter = np.clip(np.minimum(w1[:, None, 1], w2[None, :, 1]) - np.maximum(w1[:, None, 0], w2[None, :, 0]), 0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter / ((w1[:, 1] - w1[:, 0])[:, None] + (w2[:, 1] - w2[:, 0])[None, :] - inter)
    return np.nan_to_num(iou, nan=0.0)


def _recall_scores(ious, total):
    hits = [sum(1 for v in ious if v >= thr) for thr in IOU_THRESHOLDS]
    recall = [h / total for h in hits]
    return dict(mIoU=round(sum(ious) / total, 5), F1=round(sum(recall) / len(recall), 5))


def score_grounding(samples):
    """TVG / EPM: recall of the first predicted span at IoU 0.1 / 0.3 / 0.5 / 0.7 (called F1 in the paper)."""
    ious, failed = [], 0
    for s in samples:
        spans = parse_spans(s['a'])
        if spans is None:
            failed += 1
            continue
        ious.append(float(_iou([s['tgt']], [spans[0]])[0, 0]))
    return dict(Total=len(samples), Failed=failed, **_recall_scores(ious, len(samples)))


def score_highlight(samples):
    """VHD: the first number in the answer must fall inside a highlight of one of the annotators."""
    hits, failed = 0, 0
    for s in samples:
        gt = s['tgt'] if isinstance(s['tgt'][0][0], (list, tuple)) else [s['tgt']]
        match = re.search(r'[-+]?\d*\.\d+|[-+]?\d+', str(s['a']))
        if not match:
            failed += 1
            continue
        t = float(match.group(0))
        hits += any(a <= t <= b for annotator in gt for a, b in annotator)
    return dict(Total=len(samples), Failed=failed, F1=round(hits / len(samples), 5))


def score_temporal_matching(samples):
    """TEM: recall of the first predicted span against the best-matching ground-truth span."""
    ious, failed = [], 0
    for s in samples:
        spans = parse_spans(s['a'])
        if spans is None:
            failed += 1
            continue
        ious.append(float(_iou(s['tgt'], [spans[0]]).max()))
    out = _recall_scores(ious, len(samples))
    return dict(Total=len(samples), Failed=failed, mIoU=out['mIoU'], Rec=out['F1'])


def score_action_localization(samples):
    """TAL: F1 of all predicted spans against all ground-truth spans at each IoU threshold."""
    f1, failed = [0.0] * len(IOU_THRESHOLDS), 0
    for s in samples:
        spans = parse_spans(s['a'])
        if spans is None:
            failed += 1
            continue
        iou = _iou(s['tgt'], spans)
        for i, thr in enumerate(IOU_THRESHOLDS):
            if iou.max() >= thr:
                rec = float((iou.max(axis=1) >= thr).mean())
                prc = float((iou.max(axis=0) >= thr).mean())
                f1[i] += 2 * prc * rec / (prc + rec)
    f1 = [v / len(samples) for v in f1]
    return dict(Total=len(samples), Failed=failed, F1=round(sum(f1) / len(f1), 5))


def score_summarization(samples):
    """EVS: per-second F1 between the predicted and the ground-truth summary (answers that cannot be read
    are left out of the average, as in the official script)."""
    f1, failed = [], 0
    for s in samples:
        spans = parse_spans(s['a'])
        if spans is None:
            failed += 1
            continue
        gt_map, pred_map = np.zeros(1000), np.zeros(1000)
        for a, b in s['tgt']:
            gt_map[max(0, round(a)):round(b)] = 1
        for a, b in spans:
            pred_map[max(0, round(a)):round(b)] = 2
        both = gt_map + pred_map
        tp, fp, fn = int((both == 3).sum()), int((both == 2).sum()), int((both == 1).sum())
        if tp == 0:
            f1.append(0)
        else:
            rec, prc = tp / (tp + fn), tp / (tp + fp)
            f1.append(2 * prc * rec / (prc + rec))
    return dict(Total=len(samples), Failed=failed, F1=round(sum(f1) / len(f1), 5) if f1 else 0)


class SentenceSimilarity:
    """Dot product of all-MiniLM-L6-v2 sentence embeddings, the similarity used by E.T. Bench."""

    def __init__(self):
        self._model = None

    @property
    def model(self):  # loaded on first use
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        return self._model

    def pairs(self, a, b):
        ea = self.model.encode(list(a), convert_to_numpy=True)
        eb = self.model.encode(list(b), convert_to_numpy=True)
        return np.sum(ea * eb, axis=1)

    def best_option(self, answer, options):
        labelled = [f'({_OPTION_LETTERS[i]}) ' + (f'{o[0]} - {o[1]}' if isinstance(o, (list, tuple)) else o)
                    for i, o in enumerate(options)]
        return int(np.argmax(self.pairs([answer] * len(labelled), labelled)))


def _letter(answer, num_options):
    """The option index given by a leading letter ("B", "B.", "(B)", ...) after "Best Option:", or None."""
    text = answer.lower()
    if text.startswith('best option:'):
        text = text[12:]
    text = text.lstrip().lstrip('(').lstrip()
    if not text:
        return 'empty'
    if len(text) == 1 or text[1] in ('.', ',', ' ', ')'):
        idx = 'abcde'.find(text[0])
        return idx if 0 <= idx < num_options else 'letter'
    return None


def score_choice(samples, sim):
    """RAR / ECA / RVQ multiple choice accuracy. Answers whose letter is missing or wrong are matched to the
    most similar option text, as in the official script."""
    hits, failed = 0, 0
    num_options = len(samples[0]['o'])
    for s in samples:
        answer = str(s['a'])
        match = re.search(r'\(([A-Za-z])\)', answer)
        if match and 'abcde'.find(match.group(1).lower()) == s['p'] and s['p'] < num_options:
            hits += 1
            continue
        letter = _letter(answer, num_options)
        if letter == 'empty':
            failed += 1
            continue
        if isinstance(letter, int) and letter == s['p']:
            hits += 1
            continue
        if match is None and letter is None:
            failed += 1
        text = answer.lower()
        text = (text[12:] if text.startswith('best option:') else text).lstrip().lstrip('(').lstrip()
        hits += sim.best_option(text, s['o']) == s['p']
    return dict(Total=len(samples), Failed=failed, Acc=round(hits / len(samples), 5))


def score_grounded_qa(samples, sim):
    """GVQ: the option must be right, then the first predicted span is scored like TEM (recall of correctly
    answered questions at each IoU threshold)."""
    correct, failed = [], 0
    for s in samples:
        answer = str(s['a'])
        text = answer[12:] if answer.lower().startswith('best option:') else answer
        text = text.lstrip().lstrip('(').lstrip()
        if not text:
            failed += 1
            continue
        if (len(text) == 1 or text[1] in ('.', ',', ' ', ')')) and text[0].lower() in 'abcde'[:len(s['o'])]:
            if 'abcde'.find(text[0].lower()) == s['p']:
                correct.append(s)
            continue
        if sim.best_option(text, s['o']) == s['p']:
            correct.append(s)
    ious = []
    for s in correct:
        spans = parse_spans(s['a'])
        if spans is None:
            failed += 1
            continue
        ious.append(float(_iou([s['tgt']], [spans[0]])[0, 0]))
    out = _recall_scores(ious, len(samples))
    return dict(Total=len(samples), Failed=failed, mIoU=out['mIoU'], Rec=out['F1'],
                Acc=round(len(correct) / len(samples), 5))


def _caption_iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = min(max(a[1], b[1]) - min(a[0], b[0]), a[1] - a[0] + b[1] - b[0])
    return inter / (union + 1e-8)


def _tokenize(captions):
    """PTB tokenization as in the COCO caption tools; a simple equivalent is used when pycocoevalcap (and
    Java) are not available."""
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        out = PTBTokenizer(verbose=False).tokenize({i: [{'caption': c}] for i, c in enumerate(captions)})
        return [out[i][0] for i in range(len(captions))]
    except Exception:
        punct = {"''", "'", '``', '`', '.', '?', '!', ',', ':', '-', '--', '...', ';'}
        tokens = [re.findall(r"\w+|[^\w\s]+", c.lower()) for c in captions]
        return [' '.join(t for t in toks if t not in punct) for toks in tokens]


def score_dense_captions(samples, sim, seed=0):
    """DVC / SLC: F1 of the event spans (precision and recall at IoU 0.1 / 0.3 / 0.5 / 0.7, scaled by the share
    of readable answers) and the sentence similarity of matched captions. As in the official script, a
    predicted caption without a matching event is compared with a random string; the strings come from a
    fixed seed here, so the similarity is reproducible."""
    gt, pred, failed = {}, {}, 0
    for s in samples:
        times, caps = parse_events(s['a'])
        if times is None:
            failed += 1
            continue
        gt[s['video']] = (s['tgt'], s['g'])
        pred[s['video']] = list(zip(times, caps))[:1000]
    scale = len(pred) / len(samples)
    out = dict(Total=len(samples), Failed=failed, F1=0.0, Sim=0.0)
    if not gt:
        return out

    f1 = []
    for thr in IOU_THRESHOLDS:
        precision, recall = [], []
        for vid, (spans, _) in gt.items():
            ref_covered, pred_covered = set(), set()
            for p_idx, (p_span, _) in enumerate(pred[vid]):
                for r_idx, r_span in enumerate(spans):
                    if _caption_iou(p_span, r_span) > thr:
                        ref_covered.add(r_idx)
                        pred_covered.add(p_idx)
            precision.append(len(pred_covered) / len(pred[vid]))
            recall.append(len(ref_covered) / len(spans))
        p, r = float(np.mean(precision)) * scale, float(np.mean(recall)) * scale
        f1.append(0 if p + r == 0 else 2 * p * r / (p + r))
    out['F1'] = round(sum(f1) / len(f1), 5)

    def ascii_only(text):
        return ''.join(c if ord(c) < 128 else ' ' for c in text)

    rng = random.Random(seed)
    sims = []
    for thr in IOU_THRESHOLDS:
        res, ref, owner = [], [], []
        for vid in sorted(gt):
            spans, captions = gt[vid]
            for p_span, p_cap in pred[vid]:
                matched = [captions[i] for i, span in enumerate(spans) if _caption_iou(p_span, span) >= thr]
                if not matched:
                    length = rng.randint(10, 20)
                    matched = [''.join(rng.choice(string.ascii_lowercase) for _ in range(length))]
                for cap in matched:
                    res.append(ascii_only(p_cap))
                    ref.append(ascii_only(cap))
                    owner.append(vid)
        scores = sim.pairs(_tokenize(ref), _tokenize(res))
        per_video = pd.Series(scores).groupby(owner).mean()
        sims.append(float(per_video.reindex(sorted(gt)).fillna(0).mean()))
    out['Sim'] = round(sum(sims) / len(sims), 5)
    return out


def etbench_metrics(samples):
    """All E.T. Bench metrics for a list of samples (dicts with the annotation fields and the answer 'a').
    Returns the per (task, source) results and the 14 headline numbers (task averages over sources, in %)."""
    groups = {}
    for s in samples:
        groups.setdefault(s['task'], {}).setdefault(s['source'], []).append(s)
    sim = SentenceSimilarity()
    per_source = {}
    for task, sources in groups.items():
        for source, items in sources.items():
            if task in ('tvg', 'epm'):
                res = score_grounding(items)
            elif task == 'vhd':
                res = score_highlight(items)
            elif task == 'tem':
                res = score_temporal_matching(items)
            elif task == 'tal':
                res = score_action_localization(items)
            elif task == 'evs':
                res = score_summarization(items)
            elif task in ('dvc', 'slc'):
                res = score_dense_captions(items, sim)
            elif task in ('rar', 'eca', 'rvq'):
                res = score_choice(items, sim)
            elif task == 'gvq':
                res = score_grounded_qa(items, sim)
            else:
                raise ValueError(f'Unknown E.T. Bench task: {task}')
            per_source[f'{task}_{source}'] = res
    headline = {}
    for task, metric in HEADLINE:
        values = [r[metric] for key, r in per_source.items() if key.split('_', 1)[0] == task]
        if values:
            headline[f'{task.upper()}_{metric}'] = round(sum(values) / len(values), 5)
    return per_source, headline


class ETBench(VideoBaseDataset):
    """E.T. Bench (Liu et al., NeurIPS 2024): 7,289 samples in 12 event-level, time-sensitive tasks over
    7K videos (https://huggingface.co/datasets/PolyU-ChenLab/ETBench).

    Each sample comes with its own prompt. The model answers in free text and the answers are scored with
    the rules of the official evaluation script. The 14 headline numbers are accuracy for referring (RAR, ECA,
    RVQ), F1 for grounding (TVG, EPM, TAL, EVS, VHD), F1 and sentence similarity for dense captioning (DVC,
    SLC) and recall for complex understanding (TEM, GVQ). ``ETBench-Subset`` is the official 470-sample subset.
    """

    TYPE = 'Video-Temporal-Grounding'
    DEFAULT_JUDGE_MODEL = None
    HF_REPO_ID = 'PolyU-ChenLab/ETBench'
    ANNOTATION_FILE = 'annotations/etbench_txt_v1.0.json'
    SUBSET_FILE = 'evaluation/subset.json'
    # Videos in the official archives whose timestamps are broken, so that decord cannot seek in them and loading
    # the video fails. They are re-encoded once after extraction, frame for frame at the same frame rate; the
    # original is kept next to it as <name>.orig.mp4.
    BROKEN_VIDEOS = ('summe/cooking', )

    @classmethod
    def supported_datasets(cls):
        return ['ETBench', 'ETBench-Subset']

    def __init__(self, dataset='ETBench', pack=False, nframe=0, fps=1.0):
        self._video_meta = {}
        super().__init__(dataset=dataset, pack=pack, nframe=nframe, fps=fps)

    # ------------------------------------------------------------------ data
    @classmethod
    def _build_tsv(cls, annotation_file, subset_file, data_file, subset=False):
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)
        if subset:
            with open(subset_file, 'r', encoding='utf-8') as f:
                keep = json.load(f)
            annos = [a for a in annos if a['idx'] in keep.get(a['task'], {}).get(a['source'], [])]
        records = []
        for a in annos:
            records.append(
                dict(
                    index=len(records),
                    idx=a['idx'],
                    task=a['task'],
                    source=a['source'],
                    video=osp.splitext(a['video'])[0],
                    question=a['q'],
                    duration=float(a['duration']),
                    **{k: json.dumps(a[k]) if k in a else '' for k in ('tgt', 'p', 'o', 'g', 'src')},
                ))
        tmp = str(data_file) + '.tmp.tsv'
        dump(pd.DataFrame(records), tmp)
        os.replace(tmp, data_file)

    @staticmethod
    def _missing_videos(video_root, data_file):
        videos = set(load(str(data_file))['video'])
        return {v for v in videos if not osp.exists(osp.join(video_root, f'{v}.mp4'))}

    @staticmethod
    def _extract(archive, target):
        target = Path(target).resolve()
        with tarfile.open(archive, 'r:*') as tar:  # the archives are plain tar files despite the .tar.gz name
            for member in tar:
                dest = (target / member.name).resolve()
                if member.isfile() and str(dest).startswith(str(target) + os.sep):
                    tar.extract(member, target)

    def prepare_dataset(self, dataset):
        from huggingface_hub import hf_hub_download
        root = Path(LMUDataRoot()) / 'ETBench'
        root.mkdir(parents=True, exist_ok=True)
        for name in (self.ANNOTATION_FILE, self.SUBSET_FILE):
            if not (root / name).exists():
                hf_hub_download(repo_id=self.HF_REPO_ID, repo_type='dataset', filename=name, local_dir=str(root))
        annotation_file = root / self.ANNOTATION_FILE
        data_file = root / f'{dataset}.tsv'
        if not data_file.exists() or annotation_file.stat().st_mtime > data_file.stat().st_mtime:
            self._build_tsv(annotation_file, root / self.SUBSET_FILE, data_file, subset=dataset.endswith('-Subset'))

        video_root = root / 'videos'
        missing = self._missing_videos(video_root, data_file)
        for source in sorted({v.split('/')[0] for v in missing}):
            archive = f'videos/{source}.tar.gz'
            if not (root / archive).exists():
                hf_hub_download(repo_id=self.HF_REPO_ID, repo_type='dataset', filename=archive, local_dir=str(root))
            self._extract(root / archive, video_root)
        missing = self._missing_videos(video_root, data_file)
        if missing:
            raise FileNotFoundError(f'{len(missing)} E.T. Bench videos are missing under {video_root}.')
        self._repair_videos(video_root, data_file)
        return dict(root=str(video_root), data_file=str(data_file))

    @staticmethod
    def _reencode(path):
        from fractions import Fraction

        import av
        tmp = path.with_suffix('.tmp.mp4')
        with av.open(str(path)) as src:
            stream = src.streams.video[0]
            rate = Fraction(stream.average_rate)
            with av.open(str(tmp), 'w') as dst:
                codec = 'libx264' if 'libx264' in av.codecs_available else 'mpeg4'
                enc = dst.add_stream(codec, rate=rate)
                enc.width, enc.height, enc.pix_fmt = stream.codec_context.width, stream.codec_context.height, 'yuv420p'
                for i, frame in enumerate(src.decode(stream)):
                    frame = frame.reformat(format='yuv420p')
                    frame.pts, frame.time_base = i, 1 / rate
                    for packet in enc.encode(frame):
                        dst.mux(packet)
                for packet in enc.encode():
                    dst.mux(packet)
        os.replace(path, path.with_suffix('.orig.mp4'))
        os.replace(tmp, path)

    def _repair_videos(self, video_root, data_file):
        videos = set(load(str(data_file))['video'])
        for video in self.BROKEN_VIDEOS:
            path = Path(video_root) / f'{video}.mp4'
            if video not in videos or path.with_suffix('.orig.mp4').exists():
                continue
            try:
                self._reencode(path)
            except Exception as e:
                warnings.warn(f'Could not re-encode {path} ({e}). decord cannot read this video; install PyAV '
                              '(pip install av) so that it can be repaired.')

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
            return [item, dict(type='text', value=line['question'])]
        frames = self.save_video_frames(video)
        message = tg.frames_with_timestamps(frames, self._frame_times(video))
        message.append(dict(type='text', value=FRAMES_PREFIX + line['question']))
        return message

    # ------------------------------------------------------------ evaluation
    @staticmethod
    def _samples(data):
        samples = []
        for _, row in data.iterrows():
            s = dict(task=row['task'], source=row['source'], idx=int(row['idx']), video=row['video'])
            for key in ('tgt', 'p', 'o', 'g', 'src'):
                value = row.get(key, None)
                if isinstance(value, str):
                    if value:
                        s[key] = json.loads(value)
                elif value is not None and not pd.isna(value):  # the answer index is read back as a number
                    s[key] = int(value) if key == 'p' else value
            prediction = row.get('prediction', '')
            s['a'] = '' if prediction is None or (isinstance(prediction, float) and np.isnan(prediction)) \
                else str(prediction)
            samples.append(s)
        return samples

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            per_source, headline = etbench_metrics(self._samples(data))

        metrics = {k: round(v * 100, 2) for k, v in headline.items()}
        groups = {
            'Acc_ref': ['RAR_Acc', 'ECA_Acc', 'RVQ_Acc'],
            'F1_gnd': ['TVG_F1', 'EPM_F1', 'TAL_F1', 'EVS_F1', 'VHD_F1'],
            'F1_cap': ['DVC_F1', 'SLC_F1'],
            'Sim_cap': ['DVC_Sim', 'SLC_Sim'],
            'Rec_com': ['TEM_Rec', 'GVQ_Rec'],
        }
        for name, keys in groups.items():
            if all(k in headline for k in keys):
                metrics[name] = round(sum(headline[k] for k in keys) * 100 / len(keys), 2)
        metrics['per_source'] = per_source
        metrics['num_samples'] = len(data)

        dump(data, get_intermediate_file_path(eval_file, '_judge', 'xlsx'))
        dump(metrics, get_intermediate_file_path(eval_file, '_score', 'json'))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        for key in ('F1_gnd', 'F1_gnd|F1_gnd'):
            if isinstance(metrics, dict) and key in metrics:
                return {'F1_gnd': metrics[key]}
        return {}
