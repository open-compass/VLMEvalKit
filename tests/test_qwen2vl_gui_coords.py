"""Regression tests for Qwen-VL GUI coordinate handling.

The helper is loaded by path so these tests run without torch/pandas, matching
tests/test_gpt_image_budget.py.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_COORDS_PATH = Path(__file__).resolve().parents[1] / "vlmeval" / "smp" / "coords.py"
_spec = importlib.util.spec_from_file_location("vlmeval_smp_coords", _COORDS_PATH)
coords = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(coords)

rescale_click = coords.rescale_click
resized_size = coords.resized_size
smart_resize = coords.smart_resize


def test_smart_resize_matches_transformers_reference():
    # Reference values from transformers Qwen2VLImageProcessor.smart_resize.
    assert smart_resize(2160, 3840, max_pixels=1003520) == (728, 1316)
    assert smart_resize(2160, 3840) == (2156, 3836)


def test_resized_size_stays_in_budget_and_on_the_patch_grid():
    width, height = resized_size((3840, 2160), max_pixels=1003520)
    assert width * height <= 1003520
    assert width % 28 == 0 and height % 28 == 0


def test_click_is_mapped_back_to_original_pixels():
    # The model saw a 1316x728 downscale and answered in that frame.
    out = rescale_click("pyautogui.click(x=987, y=546)", (3840, 2160), max_pixels=1003520)
    assert out == "pyautogui.click(x=2880, y=1620)"


def test_click_is_unchanged_when_no_resize_happens():
    text = "pyautogui.click(x=10, y=20)"
    assert rescale_click(text, (840, 560)) == text


@pytest.mark.parametrize(
    "size,cap",
    [((3840, 2160), 1003520), ((5120, 2880), 12845056), ((1920, 1080), 1605632)],
)
def test_perfect_model_click_round_trips(size, cap):
    """A click at the ground-truth centre must survive the resize and come back within 1 px."""
    width, height = size
    resized_w, resized_h = resized_size(size, max_pixels=cap)
    centre_x, centre_y = width * 0.61, height * 0.42
    answer = f"pyautogui.click(x={round(centre_x * resized_w / width)}, y={round(centre_y * resized_h / height)})"

    fixed = rescale_click(answer, size, max_pixels=cap)

    match = re.search(r"x=(\d+), y=(\d+)", fixed)
    assert match is not None
    assert abs(int(match.group(1)) - centre_x) <= 1
    assert abs(int(match.group(2)) - centre_y) <= 1
