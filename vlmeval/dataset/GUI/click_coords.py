"""Map parsed GUI clicks into fractions of the original image.

ScreenSpot asks for ``pyautogui.click(x, y)`` in absolute pixels and also
accepts answers that are already fractions of the image. Qwen-VL style
grounding instead emits a 0-1000 grid (see issues #1706 and #1332). Treating
those values as pixels divides them by the screenshot size, so a correct
``(220, 228)`` lands far from the target while ``Format_Err_Rate`` stays 0.

``auto`` keeps fractions as fractions, keeps absolute pixels when a value
cannot fit on a 0-1000 grid, and otherwise treats a file of in-range 0-1000
clicks on a larger screenshot as that grid. Override with
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


def _active_clicks(points, image_sizes):
    """Clicks that are not fractions and not negative parse sentinels."""
    active = []
    for point, size in zip(points, image_sizes):
        if point is None:
            continue
        x = float(point[0])
        y = float(point[1])
        if x < 0 or y < 0 or (x <= 1 and y <= 1):
            continue
        active.append((x, y, float(size[0]), float(size[1])))
    return active


def infer_coord_scale(points, image_sizes):
    """Pick ``norm_1``, ``norm_1000``, or ``pixel``.

    Returns ``(scale, reason)``. ``reason`` is ``None`` when the choice matches
    the historical behavior (fractions stay fractions, anything larger is pixels)
    and no warning is needed.
    """
    active = _active_clicks(points, image_sizes)
    if not active:
        return NORM_1, None

    if any(x > QWEN_GRID or y > QWEN_GRID for x, y, _w, _h in active):
        # A 0-1000 grid cannot produce this value, so the file is absolute pixels.
        return PIXEL, None

    outside_image = [(x, y, w, h) for x, y, w, h in active if x > w or y > h]
    if outside_image:
        return NORM_1000, ('parsed click falls outside the screenshot but inside the 0-1000 grid; '
                           'dividing by 1000. Absolute pixels cannot leave the image.')

    larger_than_grid = any(w > QWEN_GRID or h > QWEN_GRID for _x, _y, w, h in active)
    if larger_than_grid:
        return NORM_1000, (
            'every click is in (1, 1000] while a screenshot is larger than 1000px, '
            'so the answers match a 0-1000 normalized grid rather than absolute pixels. '
            f'Set {COORD_SCALE_ENV}=pixel if these clicks are real pixels.')

    return PIXEL, (
        'clicks fit both absolute pixels and a 0-1000 grid, and every screenshot is '
        f'at most 1000px. Keeping absolute pixels. Set {COORD_SCALE_ENV}=norm_1000 to divide by 1000.'
    )


def resolve_click_scale(points, image_sizes, explicit=None):
    """Resolve the scale for one prediction file.

    ``explicit`` (or ``SCREENSPOT_COORD_SCALE`` when it is omitted) selects a
    scale. ``auto`` inspects the parsed clicks.
    """
    if explicit is None:
        explicit = os.environ.get(COORD_SCALE_ENV, AUTO)
    scale = canonicalize_coord_scale(explicit)
    if scale is None or scale == AUTO:
        return infer_coord_scale(points, image_sizes)
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
