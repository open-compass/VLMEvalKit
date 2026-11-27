# Adopted from the DIVE-Bench lmms-eval fork (lmms_eval/tasks/densevideo/utils.py).
# Original lmms-eval source: https://github.com/EvolvingLMMs-Lab/lmms-eval
# Copyright (c) 2024 LMMs-Lab. Licensed under the Apache License, Version 2.0.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0.
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modifications: isolate objective metrics from API scoring and runtime loading.

"""DIVE-Bench objective metrics (ported from the Apache-licensed released evaluator).

No judge APIs, data downloads or model imports are performed by this module.
"""

import ast
import json
import math
import re

import numpy as np

FAST_TEXT_METRICS = False
MAX_EDIT_DISTANCE_CELLS = 25000000


def _normalize_text(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    return " ".join(text.split())


def _levenshtein_distance(seq_a, seq_b):
    len_a = len(seq_a)
    len_b = len(seq_b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    prev = list(range(len_b + 1))
    for i in range(1, len_a + 1):
        curr = [i] + [0] * len_b
        for j in range(1, len_b + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[len_b]


def _compute_cer(pred, ref):
    pred_chars = list(_normalize_text(pred))
    ref_chars = list(_normalize_text(ref))
    if FAST_TEXT_METRICS and len(pred_chars) * len(ref_chars) > MAX_EDIT_DISTANCE_CELLS:
        return float("nan")
    return _levenshtein_distance(pred_chars, ref_chars) / max(len(ref_chars), 1)


def _compute_wer(pred, ref):
    pred_words = _normalize_text(pred).split()
    ref_words = _normalize_text(ref).split()
    if FAST_TEXT_METRICS and len(pred_words) * len(ref_words) > MAX_EDIT_DISTANCE_CELLS:
        return float("nan")
    return _levenshtein_distance(pred_words, ref_words) / max(len(ref_words), 1)


def _compute_exact_match(pred, ref):
    return float(_normalize_text(pred) == _normalize_text(ref))


def _compute_token_f1(pred, ref):
    pred_tokens = _normalize_text(pred).split()
    ref_tokens = _normalize_text(ref).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = {}
    ref_counts = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        if token in ref_counts:
            overlap += min(count, ref_counts[token])

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


GRID_LABELS = {
    "r1c1": (1.0 / 6.0, 1.0 / 6.0),
    "r1c2": (3.0 / 6.0, 1.0 / 6.0),
    "r1c3": (5.0 / 6.0, 1.0 / 6.0),
    "r2c1": (1.0 / 6.0, 3.0 / 6.0),
    "r2c2": (3.0 / 6.0, 3.0 / 6.0),
    "r2c3": (5.0 / 6.0, 3.0 / 6.0),
    "r3c1": (1.0 / 6.0, 5.0 / 6.0),
    "r3c2": (3.0 / 6.0, 5.0 / 6.0),
    "r3c3": (5.0 / 6.0, 5.0 / 6.0),
}


GRID_ALIASES = {
    "1": "r1c1",
    "2": "r1c2",
    "3": "r1c3",
    "4": "r2c1",
    "5": "r2c2",
    "6": "r2c3",
    "7": "r3c1",
    "8": "r3c2",
    "9": "r3c3",
    "topleft": "r1c1",
    "top_left": "r1c1",
    "top left": "r1c1",
    "upperleft": "r1c1",
    "upper_left": "r1c1",
    "upper left": "r1c1",
    "lefttop": "r1c1",
    "left_top": "r1c1",
    "top": "r1c2",
    "topcenter": "r1c2",
    "top_center": "r1c2",
    "top center": "r1c2",
    "topmiddle": "r1c2",
    "top_middle": "r1c2",
    "top middle": "r1c2",
    "upper": "r1c2",
    "uppercenter": "r1c2",
    "upper_center": "r1c2",
    "upper center": "r1c2",
    "topright": "r1c3",
    "top_right": "r1c3",
    "top right": "r1c3",
    "upperright": "r1c3",
    "upper_right": "r1c3",
    "upper right": "r1c3",
    "righttop": "r1c3",
    "right_top": "r1c3",
    "left": "r2c1",
    "middleleft": "r2c1",
    "middle_left": "r2c1",
    "middle left": "r2c1",
    "centerleft": "r2c1",
    "center_left": "r2c1",
    "center left": "r2c1",
    "middle": "r2c2",
    "center": "r2c2",
    "centre": "r2c2",
    "r2c2": "r2c2",
    "right": "r2c3",
    "middleright": "r2c3",
    "middle_right": "r2c3",
    "middle right": "r2c3",
    "centerright": "r2c3",
    "center_right": "r2c3",
    "center right": "r2c3",
    "bottomleft": "r3c1",
    "bottom_left": "r3c1",
    "bottom left": "r3c1",
    "lowerleft": "r3c1",
    "lower_left": "r3c1",
    "lower left": "r3c1",
    "leftbottom": "r3c1",
    "left_bottom": "r3c1",
    "bottom": "r3c2",
    "bottomcenter": "r3c2",
    "bottom_center": "r3c2",
    "bottom center": "r3c2",
    "bottommiddle": "r3c2",
    "bottom_middle": "r3c2",
    "bottom middle": "r3c2",
    "lower": "r3c2",
    "lowercenter": "r3c2",
    "lower_center": "r3c2",
    "lower center": "r3c2",
    "bottomright": "r3c3",
    "bottom_right": "r3c3",
    "bottom right": "r3c3",
    "lowerright": "r3c3",
    "lower_right": "r3c3",
    "lower right": "r3c3",
    "rightbottom": "r3c3",
    "right_bottom": "r3c3",
}


_GRID_MAX_DISTANCE = math.sqrt(2.0)


def _canonical_grid_label(label):
    if label is None:
        return None
    text = str(label).strip().lower()
    text = text.strip("`'\".,;:()[]{}")
    text = re.sub(r"[\s\-]+", " ", text)
    compact = text.replace(" ", "")
    text_us = text.replace(" ", "_")
    for candidate in (text, compact, text_us):
        if candidate in GRID_LABELS:
            return candidate
        if candidate in GRID_ALIASES:
            return GRID_ALIASES[candidate]
    match = re.fullmatch(r"r\s*([1-3])\s*c\s*([1-3])", text)
    if match:
        return f"r{match.group(1)}c{match.group(2)}"
    match = re.fullmatch(r"row\s*([1-3])\s*(?:col|column)\s*([1-3])", text)
    if match:
        return f"r{match.group(1)}c{match.group(2)}"
    return None


def parse_grid_sequence(text):
    """Parse 3x3 grid labels from JSON lists, comma text, or natural language."""
    if text is None:
        return []

    if isinstance(text, (list, tuple)):
        out = []
        for item in text:
            nested = parse_grid_sequence(item)
            if nested:
                out.extend(nested)
            else:
                label = _canonical_grid_label(item)
                if label:
                    out.append(label)
        return out

    if isinstance(text, dict):
        for key in ("answer", "trajectory", "traj", "sequence", "grid", "positions", "labels"):
            if key in text:
                parsed = parse_grid_sequence(text[key])
                if parsed:
                    return parsed
        out = []
        for value in text.values():
            out.extend(parse_grid_sequence(value))
        return out

    raw = str(text).strip()
    if not raw:
        return []

    for parser in (json.loads, ast.literal_eval):
        if raw[:1] in "[{\"'(":
            try:
                parsed = parser(raw)
                if parsed is not raw:
                    seq = parse_grid_sequence(parsed)
                    if seq:
                        return seq
            except Exception:
                pass

    lowered = raw.lower()
    lowered = lowered.replace("_", " ").replace("-", " ")

    matches = []

    for match in re.finditer(r"\br\s*([1-3])\s*c\s*([1-3])\b", lowered):
        matches.append((match.start(), match.end(), f"r{match.group(1)}c{match.group(2)}"))
    for match in re.finditer(r"\brow\s*([1-3])\s*(?:col|column)\s*([1-3])\b", lowered):
        matches.append((match.start(), match.end(), f"r{match.group(1)}c{match.group(2)}"))

    alias_items = sorted(
        ((alias.replace("_", " ").replace("-", " "), canonical)
         for alias, canonical in GRID_ALIASES.items() if not alias.isdigit()),
        key=lambda x: len(x[0]),
        reverse=True,
    )
    for alias, canonical in alias_items:
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        for match in re.finditer(pattern, lowered):
            matches.append((match.start(), match.end(), canonical))

    for match in re.finditer(r"(?<!\d)([1-9])(?!\d)", lowered):
        matches.append((match.start(), match.end(), GRID_ALIASES[match.group(1)]))

    if matches:
        chosen = []
        occupied = []
        for start, end, label in sorted(matches, key=lambda x: (x[0], -(x[1] - x[0]))):
            if any(not (end <= s or start >= e) for s, e in occupied):
                continue
            chosen.append((start, label))
            occupied.append((start, end))
        return [label for _, label in sorted(chosen, key=lambda x: x[0])]

    out = []
    for chunk in re.split(r"[,;/\n]+|\s+then\s+|\s*->\s*", lowered):
        label = _canonical_grid_label(chunk)
        if label:
            out.append(label)
    return out


def grid_label_to_xy(label):
    canonical = _canonical_grid_label(label)
    if canonical is None:
        return None
    return GRID_LABELS.get(canonical)


def _grid_sequence_metrics(pred_seq, ref_seq):
    if not ref_seq:
        return {
            "grid_acc": 0.0,
            "grid_ade": _GRID_MAX_DISTANCE,
            "grid_fde": _GRID_MAX_DISTANCE,
            "grid_transition_acc": 0.0,
        }

    correct = 0
    distances = []
    for idx, ref_label in enumerate(ref_seq):
        pred_label = pred_seq[idx] if idx < len(pred_seq) else None
        if pred_label == ref_label:
            correct += 1
        pred_xy = grid_label_to_xy(pred_label)
        ref_xy = grid_label_to_xy(ref_label)
        if pred_xy is None or ref_xy is None:
            distances.append(_GRID_MAX_DISTANCE)
        else:
            distances.append(math.dist(pred_xy, ref_xy))

    last_pred = pred_seq[len(ref_seq) - 1] if len(pred_seq) >= len(ref_seq) else None
    last_ref = ref_seq[-1]
    last_pred_xy = grid_label_to_xy(last_pred)
    last_ref_xy = grid_label_to_xy(last_ref)
    if last_pred_xy is None or last_ref_xy is None:
        fde = _GRID_MAX_DISTANCE
    else:
        fde = math.dist(last_pred_xy, last_ref_xy)

    if len(ref_seq) <= 1:
        transition_acc = 1.0
    else:
        trans_correct = 0
        for idx in range(len(ref_seq) - 1):
            if idx + 1 >= len(pred_seq):
                continue
            ref_a = grid_label_to_xy(ref_seq[idx])
            ref_b = grid_label_to_xy(ref_seq[idx + 1])
            pred_a = grid_label_to_xy(pred_seq[idx])
            pred_b = grid_label_to_xy(pred_seq[idx + 1])
            if None in (ref_a, ref_b, pred_a, pred_b):
                continue
            ref_delta = (round(ref_b[0] - ref_a[0], 6), round(ref_b[1] - ref_a[1], 6))
            pred_delta = (round(pred_b[0] - pred_a[0], 6), round(pred_b[1] - pred_a[1], 6))
            if pred_delta == ref_delta:
                trans_correct += 1
        transition_acc = trans_correct / float(len(ref_seq) - 1)

    return {
        "grid_acc": correct / float(len(ref_seq)),
        "grid_ade": float(np.mean(distances)) if distances else _GRID_MAX_DISTANCE,
        "grid_fde": float(fde),
        "grid_transition_acc": float(transition_acc),
    }


_HIGHMOTION_GRID_NAMES = {
    "r1c1": "topleft",
    "r1c2": "top",
    "r1c3": "topright",
    "r2c1": "left",
    "r2c2": "middle",
    "r2c3": "right",
    "r3c1": "bottomleft",
    "r3c2": "bottom",
    "r3c3": "bottomright",
}


def _highmotion_count_text(value):
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }
    return words.get(int(value), str(value))
