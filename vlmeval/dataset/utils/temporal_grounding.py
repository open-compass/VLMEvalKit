"""Shared helpers for video temporal grounding (VTG) benchmarks.

A VTG model answers a text query with one or more time spans ``[start, end]`` in seconds. This module
parses such spans out of free-form model outputs, scores them against the ground truth, and computes
the timestamps of sampled frames so that models fed with frame lists can be told when each frame was
taken.
"""
import re

import numpy as np

_CLOCK = re.compile(r'\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b')
_RANGES = (
    re.compile(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)'),  # "18.5 - 23.0"
    re.compile(r'(\d+\.?\d*)\s+to\s+(\d+\.?\d*)'),  # "18.5 to 23.0"
)
_NUMBER = re.compile(r'\b(\d+\.\d+|\d+)\b')


def _clock_to_seconds(text):
    seconds = 0.0
    for part in text.split(':'):
        seconds = seconds * 60 + float(part)
    return seconds


def _pairs(values):
    values = values[:len(values) // 2 * 2]
    return [(float(values[i]), float(values[i + 1])) for i in range(0, len(values), 2)]


def extract_time_spans(text):
    """Return the time spans ``[(start, end), ...]`` found in a model answer, in order of appearance.

    The rules are those of the common VTG evaluation scripts (TimeChat, TimeLens): clock times
    (``MM:SS`` / ``HH:MM:SS``) are read first and paired in order; otherwise ``a - b`` or ``a to b``
    ranges; otherwise consecutive numbers are paired (this also reads ``[[a, b], [c, d]]``).
    Spans are returned as written; invalid spans (start >= end) are kept and score 0.
    """
    text = str(text).lower()
    clocks = _CLOCK.findall(text)
    if len(clocks) >= 2:
        return _pairs([_clock_to_seconds(t) for t in clocks])
    for pattern in _RANGES:
        found = pattern.findall(text)
        if found:
            return [(float(s), float(e)) for s, e in found]
    return _pairs(_NUMBER.findall(text))


def temporal_iou(pred, gt):
    """IoU of two single spans ``(start, end)``; 0 for an empty or invalid prediction."""
    if pred is None or len(pred) != 2 or pred[0] >= pred[1]:
        return 0.0
    inter = min(pred[1], gt[1]) - max(pred[0], gt[0])
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return max(inter, 0.0) / union if union > 0 else 0.0


def merge_spans(spans):
    """Sort spans and merge the ones that overlap or touch; drops invalid spans."""
    spans = sorted([float(s), float(e)] for s, e in spans if e > s)
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def multi_span_iou(pred, gt):
    """IoU between two sets of spans, computed on their total lengths (predicted spans merged first).

    No ground truth and no prediction counts as a perfect match (1.0).
    """
    gt = [[float(s), float(e)] for s, e in gt]
    pred = merge_spans(pred)
    if not gt:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    len_gt = sum(e - s for s, e in gt)
    len_pred = sum(e - s for s, e in pred)
    inter = sum(max(0.0, min(pe, ge) - max(ps, gs)) for ps, pe in pred for gs, ge in gt)
    union = len_pred + len_gt - inter
    return float(np.clip(inter / union, 0.0, 1.0)) if union > 0 else 0.0


def recall_at(ious, thresholds=(0.3, 0.5, 0.7)):
    """``{'R1@t': % of queries with IoU >= t}`` for each threshold ``t``."""
    ious = np.asarray(ious, dtype=float)
    return {f'R1@{t}': float(np.mean(ious >= t) * 100) if len(ious) else 0.0 for t in thresholds}


def sampled_frame_times(num_video_frames, video_fps, fps=-1, nframe=0):
    """Timestamps (s) of the frames that ``VideoBaseDataset.save_video_frames`` extracts.

    Mirrors its sampling: with ``fps > 0`` frame ``int(i * video_fps / fps)`` for each step of the
    target rate; otherwise ``nframe`` frames at ``int(i * num_video_frames / (nframe + 1))``.
    """
    if fps > 0:
        n = int(num_video_frames / video_fps * fps)
        indices = [int(i * video_fps / fps) for i in range(n)]
    else:
        step = num_video_frames / (nframe + 1)
        indices = [int(i * step) for i in range(1, nframe + 1)]
    return [idx / video_fps for idx in indices]


def frames_with_timestamps(frames, times):
    """Interleave each frame with its timestamp in seconds, e.g. ``['0.0', <frame>, '0.5', <frame>]``."""
    message = []
    for frame, t in zip(frames, times):
        message.append(dict(type='text', value=f'{t:.1f}'))
        message.append(dict(type='image', value=frame))
    return message
