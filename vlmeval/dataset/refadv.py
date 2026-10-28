"""Ref-Adv-s: referring expression comprehension with hard distractors (ICLR 2026).

Official repo: https://github.com/dddraxxx/Ref-Adv
Data: https://huggingface.co/datasets/dddraxxx/ref-adv-s

The prompt, bbox parsing and IoU computation are ported verbatim from the
official ``run.py`` / ``report.py`` so that scores match the official report.
"""
import base64
import json
import os
import os.path as osp
import re

import numpy as np
import pandas as pd

from vlmeval.smp import LMUDataRoot, dump, get_intermediate_file_path, load, toliststr
from .image_base import ImageBaseDataset

HF_REPO_ID = 'dddraxxx/ref-adv-s'
HF_FILENAME = 'data/train-00000-of-00001.parquet'

# Official `prompts.direct` in configs/qwen.yaml. The literal `<image>` is part of the
# official text prompt and is kept as-is.
DIRECT_PROMPT = ('<image>\n'
                 'Locate every object that matches the description "{ref_sentence}" '
                 'in the image. Report bbox coordinates in JSON format.')

FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', flags=re.S | re.I)

# Dict key priority used by the official `_collect_boxes`
PRIORITY_KEYS = [
    'bboxes',
    'boxes',
    'bbox',
    'bbox_2d',
    'predictions',
    'results',
    'objects',
    'coordinates',
]

BOX_FORMATS = ('abs_xyxy', 'norm_1000_xyxy', 'norm_1_xyxy')

DISTRACTOR_BINS = ('2-3', '4-6', '>=7')


def parse_distractor_count(x):
    """Same as the official `run.py::parse_distractor_count`."""
    if x is None:
        return 0
    if isinstance(x, int):
        return max(0, int(x))
    if isinstance(x, list):
        return len(x)
    s = str(x).strip()
    if not s:
        return 0
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, int):
            return max(0, int(obj))
    except Exception:
        pass
    nums = re.findall(r'\d+', s)
    if nums:
        return len(nums)
    return 0


def build_refadv_tsv(data_path):
    """Convert the HF parquet of Ref-Adv-s into a VLMEvalKit TSV."""
    from huggingface_hub import hf_hub_download

    parquet = hf_hub_download(HF_REPO_ID, HF_FILENAME, repo_type='dataset')
    df = pd.read_parquet(parquet)
    rows = []
    for r in df.itertuples(index=False):
        caption = str(r.normal_caption)
        rows.append({
            'index': int(r.row_idx),
            'image': base64.b64encode(r.image['bytes']).decode('ascii'),
            'question': DIRECT_PROMPT.format(ref_sentence=caption),
            # GT box in absolute xyxy pixels
            'answer': json.dumps([float(v) for v in r.solution]),
            'caption': caption,
            'width': int(r.width),
            'height': int(r.height),
            'distractor_count': parse_distractor_count(r.distractors),
            'image_source': str(r.image_source),
            'human_authored': bool(r.human_authored),
            'use_negation': bool(r.use_negation),
            'file_name': str(r.file_name),
        })
    out = pd.DataFrame(rows)
    # row_idx must equal the row number to keep per-sample comparison with official outputs
    assert list(out['index']) == list(range(len(out))), 'row_idx is not contiguous'
    out.to_csv(data_path, sep='\t', index=False)


def load_last_json_candidate(text):
    """Official `_load_last_json_candidate`: fenced blocks from last to first,
    otherwise a JSON value that extends to the end of the text."""
    fenced_chunks = [m.group(1).strip() for m in FENCE_RE.finditer(text) if m.group(1).strip()]
    for c in reversed(fenced_chunks):
        try:
            return json.loads(c)
        except Exception:
            continue

    t = text.rstrip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch not in '[{':
            continue
        try:
            obj, end = decoder.raw_decode(t[i:])
            if i + end == len(t):
                return obj
        except Exception:
            continue
    return None


def collect_boxes(obj, out, max_boxes=100):
    """Official `_collect_boxes`: recursively collect numeric lists of length 4."""
    if len(out) >= max_boxes:
        return
    if isinstance(obj, list):
        if len(obj) == 4 and all(isinstance(x, (int, float)) for x in obj):
            out.append([float(v) for v in obj])
            return
        for it in obj:
            collect_boxes(it, out, max_boxes=max_boxes)
        return
    if isinstance(obj, dict):
        for k in PRIORITY_KEYS:
            if k in obj:
                collect_boxes(obj[k], out, max_boxes=max_boxes)
        for k, v in obj.items():
            if k in PRIORITY_KEYS:
                continue
            collect_boxes(v, out, max_boxes=max_boxes)


def to_abs_xyxy(box, width, height, fmt):
    x1, y1, x2, y2 = box
    if fmt == 'norm_1000_xyxy':
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    elif fmt == 'norm_1_xyxy':
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif fmt != 'abs_xyxy':
        raise ValueError(f'Unsupported box format: {fmt}')

    xa, xb = sorted([float(x1), float(x2)])
    ya, yb = sorted([float(y1), float(y2)])
    xa = min(max(xa, 0.0), float(width))
    xb = min(max(xb, 0.0), float(width))
    ya = min(max(ya, 0.0), float(height))
    yb = min(max(yb, 0.0), float(height))
    return [xa, ya, xb, yb]


def parse_bboxes(text, width, height, fmt):
    """Official `parse_qwen_bboxes`. Returns (deduplicated absolute boxes, error code)."""
    if not text or not text.strip():
        return [], 'empty_response'
    parsed = load_last_json_candidate(text)
    if parsed is None:
        return [], 'no_bbox_found'

    raw_boxes = []
    collect_boxes(parsed, raw_boxes)
    if not raw_boxes:
        return [], 'no_bbox_found'

    out = []
    seen = set()
    for b in raw_boxes:
        bb = to_abs_xyxy(b, width, height, fmt)
        key = tuple(round(v, 6) for v in bb)
        if key in seen:
            continue
        seen.add(key)
        out.append(bb)
    return out, ''


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def in_distractor_bin(d, name):
    if name == '2-3':
        return 2 <= d <= 3
    if name == '4-6':
        return 4 <= d <= 6
    if name == '>=7':
        return d >= 7
    return False


class RefAdvDataset(ImageBaseDataset):
    """Ref-Adv-s, 1142 samples. Main metric is Acc@0.5 (IoU of the first predicted box
    with the GT box >= 0.5)."""

    TYPE = 'GROUNDING'
    MODALITY = 'IMAGE'

    # No public TSV: the TSV is built from the HF parquet on first use
    DATASET_URL = {'Ref-Adv-s': ''}
    DATASET_MD5 = {}

    # The official repo sets the coordinate format per model: norm_1000_xyxy for
    # Qwen3-VL / Qwen3.5 and abs_xyxy for Qwen2.5-VL. Override with REFADV_BOX_FORMAT.
    DEFAULT_BOX_FORMAT = 'norm_1000_xyxy'

    def load_data(self, dataset):
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        if not osp.exists(data_path):
            os.makedirs(LMUDataRoot(), exist_ok=True)
            build_refadv_tsv(data_path)
        self.data_path = data_path
        return load(data_path)

    @classmethod
    def report_primary_metric(cls, metrics):
        from vlmeval.smp.status_report import is_number, to_number

        if not isinstance(metrics, dict) or not metrics:
            return {}
        val = metrics.get('Acc@0.5')
        if is_number(val):
            return {'Acc@0.5': to_number(val)}
        return super().report_primary_metric(metrics)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)
        # Same message order as the official run.py: text first, then image
        msgs = [dict(type='text', value=line['question'])]
        msgs.extend(dict(type='image', value=p) for p in toliststr(tgt_path))
        return msgs

    @classmethod
    def _box_format(cls):
        fmt = os.environ.get('REFADV_BOX_FORMAT', '') or cls.DEFAULT_BOX_FORMAT
        if fmt not in BOX_FORMATS:
            raise ValueError(f'Invalid REFADV_BOX_FORMAT={fmt!r}, expected one of {BOX_FORMATS}')
        return fmt

    def evaluate(self, eval_file, **judge_kwargs):
        fmt = self._box_format()
        data = load(eval_file)
        meta = self.data.copy()
        meta['index'] = meta['index'].astype(str)
        meta = meta.set_index('index')
        data['index'] = data['index'].astype(str)

        pred_boxes, ious, parse_errors, distractors = [], [], [], []
        for _, row in data.iterrows():
            m = meta.loc[row['index']]
            width, height = int(m['width']), int(m['height'])
            gt = json.loads(m['answer'])
            pred = row['prediction']
            text = '' if pd.isna(pred) else str(pred)
            boxes, err = parse_bboxes(text, width, height, fmt)
            first = boxes[0] if boxes else []
            iou = iou_xyxy(gt, first) if len(gt) == 4 and len(first) == 4 \
                else 0.0
            pred_boxes.append(json.dumps(first))
            ious.append(iou)
            parse_errors.append(err)
            distractors.append(int(m['distractor_count']))

        data['pred_box_xyxy'] = pred_boxes
        data['iou'] = ious
        data['parse_error'] = parse_errors
        data['distractor_count'] = distractors
        dump(data, get_intermediate_file_path(eval_file, '_detail'))

        ious = np.array(ious, dtype=float)
        distractors = np.array(distractors)

        def acc(mask, thr):
            n = int(mask.sum())
            return float((ious[mask] >= thr).sum()) / n * 100 if n else 0.0

        all_mask = np.ones(len(ious), dtype=bool)
        result = {
            'Acc@0.5': acc(all_mask, 0.5),
            'Acc@0.75': acc(all_mask, 0.75),
            'Acc@0.9': acc(all_mask, 0.9),
            'ParseFail': int(sum(1 for e in parse_errors if e)),
            'Samples': int(len(ious)),
        }
        for b in DISTRACTOR_BINS:
            mask = np.array([in_distractor_bin(d, b) for d in distractors], dtype=bool)
            result[f'Acc@0.5(distractor {b})'] = acc(mask, 0.5)

        score_df = pd.DataFrame([result])
        dump(score_df, get_intermediate_file_path(eval_file, '_acc', 'csv'))
        return score_df
