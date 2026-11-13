"""Map parsed GUI clicks into fractions of the original image.

ScreenSpot asks for ``pyautogui.click(x, y)`` in absolute pixels and also
accepts answers that are already fractions of the image. Qwen-VL style
grounding instead emits a 0-1000 grid (see issues #1706 and #1332). Treating
those values as pixels divides them by the screenshot size, so a correct
``(220, 228)`` lands far from the target while ``Format_Err_Rate`` stays 0.

``auto`` keeps fractions as fractions and keeps absolute pixels when a value
cannot fit on a 0-1000 grid. A click that leaves the screenshot but stays
inside the 0-1000 grid is that grid, because a pixel cannot leave the image.
When a click can be read either as absolute pixels or as a 0-1000 point,
``auto`` keeps the pixels. Annotated boxes select the grid only when that
reading matches strictly more boxes, so a file of in-range Qwen clicks is
still divided by 1000 instead of by the screenshot size. Override with
``SCREENSPOT_COORD_SCALE`` or ``evaluate(..., coord_scale=...)``.
"""

import os

COORD_SCALE_ENV = 'SCREENSPOT_COORD_SCALE'
AUTO = 'auto'
NORM_1 = 'norm_1'
NORM_1000 = 'norm_1000'
PIXEL = 'pixel'
_SCALES = (AUTO, NORM_1, NORM_1000, PIXEL)
QWEN_GRID = 1000.0
# Values this far past the normalized image are reported, not treated as a near miss.
_OUT_OF_RANGE_TOL = 1e-3

_ALIASES = {
    '1': NORM_1,
    '0_1': NORM_1,
    '01': NORM_1,
    'fraction': NORM_1,
    'fractions': NORM_1,
    'norm1': NORM_1,
    'unit': NORM_1,
    '1000': NORM_1000,
    '0_1000': NORM_1000,
    'qwen': NORM_1000,
    'grid': NORM_1000,
    'norm1000': NORM_1000,
    'pixels': PIXEL,
    'pixel_coord': PIXEL,
    'absolute': PIXEL,
    'abs': PIXEL,
}


def canonicalize_coord_scale(scale):
    if scale is None:
        return None
    text = str(scale).strip().lower().replace('-', '_').replace(' ', '_')
    if text == '':
        return None
    text = _ALIASES.get(text, text)
    if text not in _SCALES:
        allowed = ', '.join(_SCALES)
        raise ValueError(
            f'Unknown ScreenSpot coordinate scale {scale!r}. Expected one of: {allowed}.')
    return text


def _active_clicks(points, image_sizes, boxes):
    """Clicks that are not fractions and not negative parse sentinels."""
    active = []
    for point, size, box in zip(points, image_sizes, boxes):
        if point is None:
            continue
        x = float(point[0])
        y = float(point[1])
        if x < 0 or y < 0 or (x <= 1 and y <= 1):
            continue
        active.append((x, y, float(size[0]), float(size[1]), box))
    return active


def _in_box(x, y, box):
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _grid_matches_more_boxes(active):
    """True when the 0-1000 reading hits more annotated boxes than pixels do.

    One scale is chosen for the whole file. Boxes are the same [0, 1] xyxy
    rectangles used for scoring. A missing box counts for neither reading, so
    a tie, including a target large enough to contain both readings, stays
    on absolute pixels.
    """
    pixel_hits = 0
    grid_hits = 0
    for x, y, width, height, box in active:
        if box is None or width <= 0 or height <= 0:
            continue
        if _in_box(x / width, y / height, box):
            pixel_hits += 1
        if _in_box(x / QWEN_GRID, y / QWEN_GRID, box):
            grid_hits += 1
    return grid_hits > pixel_hits


def infer_coord_scale(points, image_sizes, boxes=None):
    """Pick ``norm_1``, ``norm_1000``, or ``pixel``.

    ``boxes``, when given, are [0, 1] xyxy rectangles parallel to ``points``.
    Returns ``(scale, reason)``. ``reason`` is ``None`` when the choice matches
    the historical behavior (fractions stay fractions, anything larger is pixels)
    and no warning is needed.
    """
    if boxes is None or len(boxes) != len(points):
        boxes = [None] * len(points)
    active = _active_clicks(points, image_sizes, boxes)
    if not active:
        return NORM_1, None

    if any(x > QWEN_GRID or y > QWEN_GRID for x, y, _w, _h, _box in active):
        # A 0-1000 grid cannot produce this value, so the file is absolute pixels.
        return PIXEL, None

    outside_image = [item for item in active if item[0] > item[2] or item[1] > item[3]]
    if outside_image:
        return NORM_1000, ('parsed click falls outside the screenshot but inside the 0-1000 grid; '
                           'dividing by 1000. Absolute pixels cannot leave the image.')

    larger_than_grid = any(w > QWEN_GRID or h > QWEN_GRID for _x, _y, w, h, _box in active)
    if larger_than_grid and _grid_matches_more_boxes(active):
        return NORM_1000, (
            'clicks fit both absolute pixels and a 0-1000 grid, but the annotated boxes '
            'match the 0-1000 reading more often, so dividing by 1000. '
            f'Set {COORD_SCALE_ENV}=pixel to keep absolute pixels.')

    if larger_than_grid:
        return PIXEL, (
            'clicks fit both absolute pixels and a 0-1000 grid. '
            'Keeping absolute pixels. '
            f'Set {COORD_SCALE_ENV}=norm_1000 to divide by 1000.')

    return PIXEL, (
        'clicks fit both absolute pixels and a 0-1000 grid, and every screenshot is '
        'at most 1000px. Keeping absolute pixels. '
        f'Set {COORD_SCALE_ENV}=norm_1000 to divide by 1000.')


def resolve_click_scale(points, image_sizes, explicit=None, boxes=None):
    """Resolve the scale for one prediction file.

    ``explicit`` (or ``SCREENSPOT_COORD_SCALE`` when it is omitted) selects a
    scale. ``auto`` inspects the parsed clicks and, when both readings fit,
    keeps pixels unless ``boxes`` agree with the 0-1000 grid.
    """
    if explicit is None:
        explicit = os.environ.get(COORD_SCALE_ENV, AUTO)
    scale = canonicalize_coord_scale(explicit)
    if scale is None or scale == AUTO:
        return infer_coord_scale(points, image_sizes, boxes=boxes)
    return scale, None


def normalize_click_point(point, image_size, scale):
    """Return ``([x, y], out_of_range)`` with x and y as fractions of the image.

    A pair already inside ``[0, 1]`` is left unchanged, so a true 0-1 answer
    stays correct even when the rest of the file uses another scale. Negative
    parse sentinels are returned as-is and are not counted as out of range.
    """
    scale = canonicalize_coord_scale(scale)
    if scale in (None, AUTO):
        raise ValueError('normalize_click_point requires a concrete scale, not auto.')
    x = float(point[0])
    y = float(point[1])
    if x < 0 or y < 0:
        return [x, y], False
    if x <= 1 and y <= 1:
        return [x, y], False

    if scale == NORM_1000:
        x /= QWEN_GRID
        y /= QWEN_GRID
    elif scale == PIXEL:
        width, height = float(image_size[0]), float(image_size[1])
        if width <= 0 or height <= 0:
            return [x, y], True
        x /= width
        y /= height
    elif scale != NORM_1:
        raise ValueError(f'Unsupported coordinate scale {scale!r}.')

    out_of_range = (
        x < -_OUT_OF_RANGE_TOL or y < -_OUT_OF_RANGE_TOL or x > 1 + _OUT_OF_RANGE_TOL
        or y > 1 + _OUT_OF_RANGE_TOL)
    return [x, y], out_of_range
