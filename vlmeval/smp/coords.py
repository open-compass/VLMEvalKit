"""Coordinate-space helpers for Qwen-VL GUI models.

Qwen's image processor runs every input through ``smart_resize``: dimensions are
rounded to a multiple of 28 and the pixel budget is clamped to
``[min_pixels, max_pixels]``. A GUI model therefore emits absolute pixel
coordinates in the *resized* frame, while benchmarks store ground truth in the
*original* frame. ``pyautogui.click(x=?, y=?)`` is defined in original-image
pixels, so the wrapper maps the answer back before the evaluator normalises the
click point against the image size.
"""

from __future__ import annotations

import math
import re

FACTOR = 28
MIN_PIXELS = 3136  # 56 * 56, Qwen2-VL and Qwen2.5-VL preprocessor_config.json
MAX_PIXELS = 12845056  # 16384 * 28 * 28, both released processor configs

_CLICK_RE = re.compile(r"x\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*y\s*=\s*([0-9]+(?:\.[0-9]+)?)")


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Mirror of ``transformers`` Qwen2VLImageProcessor.smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resized_size(original_size, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Return the (width, height) the processor actually feeds the model."""
    width, height = original_size
    resized_h, resized_w = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    return resized_w, resized_h


def rescale_click(text, original_size, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    """Map ``x=..., y=...`` from the processor's resized frame to original pixels."""
    width, height = original_size
    resized_w, resized_h = resized_size(original_size, min_pixels, max_pixels)
    if (resized_w, resized_h) == (width, height):
        return text
    scale_x, scale_y = width / resized_w, height / resized_h
    return _CLICK_RE.sub(
        lambda m: f"x={round(float(m.group(1)) * scale_x)}, y={round(float(m.group(2)) * scale_y)}",
        text,
    )
