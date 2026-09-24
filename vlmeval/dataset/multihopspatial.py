import json
import re

import pandas as pd

from vlmeval.smp import dump, load
from vlmeval.smp.file import get_intermediate_file_path
from .image_base import ImageBaseDataset

# A single coordinate: an unsigned number like "12", "0.5" or ".5". The pattern
# only matches well-formed floats, so float() below can never raise on it.
_NUM = r"(\d*\.?\d+)"
_BBOX2D_RE = r'"bbox_2d"\s*:\s*\[\s*' + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]"
_BBOX_LINE_RE = r"Bounding Box:\s*\[\s*" + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]"
_BBOX_BARE_RE = r"\[\s*" + r"\s*,\s*".join([_NUM] * 4) + r"\s*\]"
_TAG_RE = re.compile(r"</?(?:ATT|POS|REL)>")

PROMPT_TEMPLATE = """{question}

Please respond in the following format:
Answer: (your choice, e.g., "(a) object name")
Bounding Box: {{"bbox_2d": [x1, y1, x2, y2]}}

Important: Use NORMALIZED coordinates (0.0 to 1.0).
Example: {{"bbox_2d": [0.25, 0.1, 0.75, 0.8]}}"""


def remove_tags(question):
    """Strip the <ATT>/<POS>/<REL> relation markers used during annotation."""
    cleaned = _TAG_RE.sub("", question)
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_mcq_answer(text):
    if not text:
        return None
    m = re.search(r"Answer:\s*(\([a-d]\)\s*[^\n]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"Answer:\s*([a-d])\)\s*([^\n]*)", text, re.IGNORECASE)
    if m:
        letter, desc = m.group(1).lower(), m.group(2).strip()
        return f"({letter}) {desc}" if desc else f"({letter})"
    m = re.search(r"Answer:\s*([a-d])\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return f"({m.group(1).lower()})"
    m = re.search(r"(\([a-d]\)\s*[^\n,\[\]]*)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_choice_letter(text):
    if text is None:
        return None
    m = re.search(r"\(([a-d])\)", text, re.IGNORECASE)
    return m.group(1).lower() if m else None


def parse_bbox(response_text):
    """Extract an xyxy box, then normalize its scale uniformly.

    Axis order is taken as xyxy exactly as prompted (no yxyx accommodation). If
    any coordinate exceeds 1, the whole box is divided by 1000 (0-1000 -> 0-1),
    a uniform units conversion applied to every model. Malformed numbers yield
    None rather than raising.
    """
    m = re.search(_BBOX2D_RE, response_text)
    if m is None:
        m = re.search(_BBOX_LINE_RE, response_text, re.IGNORECASE)
    if m is not None:
        groups = m.groups()
    else:
        found = re.findall(_BBOX_BARE_RE, response_text)
        groups = found[0] if found else None

    if groups is None:
        return None
    try:
        bbox = [float(v) for v in groups]
    except (TypeError, ValueError):
        return None

    if any(v > 1 for v in bbox):
        bbox = [v / 1000.0 for v in bbox]
    return bbox


def is_valid_norm_bbox(bbox):
    if bbox is None or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = bbox
    if any(v < 0 or v > 1 for v in bbox):
        return False
    return x2 > x1 and y2 > y1


def calculate_iou(bbox_gt_xywh, bbox_pred_norm, img_w, img_h):
    """gt is [x, y, w, h] in pixels; pred is [x1, y1, x2, y2] normalized 0-1."""
    if bbox_gt_xywh is None or bbox_pred_norm is None:
        return None
    if len(bbox_gt_xywh) != 4 or len(bbox_pred_norm) != 4:
        return None
    try:
        gx, gy, gw, gh = bbox_gt_xywh
        gt = [gx, gy, gx + gw, gy + gh]
        pred = [
            bbox_pred_norm[0] * img_w,
            bbox_pred_norm[1] * img_h,
            bbox_pred_norm[2] * img_w,
            bbox_pred_norm[3] * img_h,
        ]
        ix1, iy1 = max(gt[0], pred[0]), max(gt[1], pred[1])
        ix2, iy2 = min(gt[2], pred[2]), min(gt[3], pred[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
        area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
        union = area_gt + area_pred - inter
        if union <= 0:
            return 0.0
        return round(inter / union, 4)
    except (TypeError, ValueError):
        return None


class MultihopSpatial(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "MultihopSpatial": "https://huggingface.co/datasets/etri-vilab/MultihopSpatial/resolve/main/multihopspatial.tsv"  # noqa: E501
    }
    DATASET_MD5 = {"MultihopSpatial": "45de7d0c96dc010bf6f26d4ab469bfc8"}

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        prompt = PROMPT_TEMPLATE.format(question=remove_tags(line["question"]))

        if isinstance(tgt_path, list):
            msgs = [dict(type="image", value=p) for p in tgt_path]
        else:
            msgs = [dict(type="image", value=tgt_path)]
        msgs.append(dict(type="text", value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        raw = self.data.set_index("index")

        records = []
        for i in range(len(data)):
            item = data.iloc[i]
            gt = raw.loc[item["index"]]

            response = str(item["prediction"])
            pred_letter = extract_choice_letter(parse_mcq_answer(response))
            gt_letter = extract_choice_letter(str(gt["answer"]))
            mcq_correct = pred_letter is not None and pred_letter == gt_letter

            iou = None
            pred_bbox = parse_bbox(response)
            if is_valid_norm_bbox(pred_bbox):
                gt_bbox = json.loads(gt["bbox"])
                iou = calculate_iou(gt_bbox, pred_bbox, int(gt["width"]), int(gt["height"]))

            records.append(
                {
                    "index": item["index"],
                    "hop": gt["hop"],
                    "view": gt["view"],
                    "mcq_correct": bool(mcq_correct),
                    "iou": iou,
                }
            )

        df = pd.DataFrame(records)
        dump(df, get_intermediate_file_path(eval_file, "_detailed", "pkl"))

        def metrics(sub):
            n = len(sub)
            if n == 0:
                return 0.0, 0.0, 0.0
            correct = sub["mcq_correct"]
            mcq_acc = correct.mean() * 100
            correct_iou = sub[correct & sub["iou"].notna()]["iou"]
            avg_iou = correct_iou.mean() * 100 if len(correct_iou) else 0.0
            acc50 = (correct & sub["iou"].notna() & (sub["iou"] >= 0.5)).sum() / n * 100
            return round(mcq_acc, 2), round(acc50, 2), round(avg_iou, 2)

        row = {}
        row["mcq_acc"], row["acc@50iou"], row["avg_iou"] = metrics(df)
        # Per-hop and per-hop/view breakdown
        for hop in sorted(df["hop"].unique()):
            sub = df[df["hop"] == hop]
            m, a, v = metrics(sub)
            row[f"{hop}_mcq_acc"], row[f"{hop}_acc@50iou"], row[f"{hop}_avg_iou"] = m, a, v
            for view in sorted(sub["view"].unique()):
                sv = sub[sub["view"] == view]
                m2, a2, v2 = metrics(sv)
                row[f"{hop}_{view}_mcq_acc"] = m2
                row[f"{hop}_{view}_acc@50iou"] = a2
                row[f"{hop}_{view}_avg_iou"] = v2

        scores_df = pd.DataFrame({k: [v] for k, v in row.items()})
        dump(scores_df, get_intermediate_file_path(eval_file, "_score", "csv"))
        return scores_df
