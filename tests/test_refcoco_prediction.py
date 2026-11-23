import ast
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_refcoco_dataset():
    path = ROOT / 'vlmeval' / 'dataset' / 'refcoco.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    definition = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'RefCOCODataset'
    )
    namespace = {
        'Dict': Dict,
        'ImageBaseDataset': object,
        'List': List,
        'Optional': Optional,
        'np': np,
        'pd': pd,
        're': re,
    }
    module = ast.Module(body=[definition], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['RefCOCODataset']


RefCOCODataset = _load_refcoco_dataset()


def test_parses_bracketed_bbox():
    parsed = RefCOCODataset._parse_prediction('The answer is [10, 20, 30, 40].')

    np.testing.assert_array_equal(parsed, np.array([10, 20, 30, 40], dtype=float))


def test_parses_internvl_bbox_after_long_chain_of_thought():
    prediction = (
        '1. Locate the television.\n'
        '2. Inspect the person near the center.\n'
        '3. Compute the final region.\n'
        '<bbox>470 402 742 847</bbox>'
    )

    parsed = RefCOCODataset._parse_prediction(prediction)

    np.testing.assert_array_equal(parsed, np.array([470, 402, 742, 847], dtype=float))


@pytest.mark.parametrize(
    'prediction',
    [
        '<bbox> -1.5, 2, 300, 4e2 </bbox>',
        '<BBOX>\n-1.5\n2\n300\n4e2\n</BBOX>',
        'Reasoning mentions 1 and 2. <bbox>-1.5 2,300 4e2</bbox>',
    ],
)
def test_parses_tagged_bbox_number_and_separator_variants(prediction):
    parsed = RefCOCODataset._parse_prediction(prediction)

    np.testing.assert_array_equal(parsed, np.array([-1.5, 2, 300, 400], dtype=float))


def test_bracketed_bbox_keeps_precedence_when_both_containers_are_present():
    prediction = (
        'The intermediate box is [1, 2, 3, 4]. '
        'The final box is <bbox>10 20 30 40</bbox>.'
    )

    parsed = RefCOCODataset._parse_prediction(prediction)

    np.testing.assert_array_equal(parsed, np.array([1, 2, 3, 4], dtype=float))


def test_first_complete_tagged_bbox_is_used():
    prediction = (
        'Example: <bbox>1 2 3 4</bbox>. '
        'Final answer: <bbox>10 20 30 40</bbox>'
    )

    parsed = RefCOCODataset._parse_prediction(prediction)

    np.testing.assert_array_equal(parsed, np.array([1, 2, 3, 4], dtype=float))


def test_incomplete_tag_keeps_legacy_numeric_fallback():
    prediction = 'Incomplete <bbox>10 20 30</bbox>; fallback coordinates 1 2 3 4.'

    parsed = RefCOCODataset._parse_prediction(prediction)

    np.testing.assert_array_equal(parsed, np.array([10, 20, 30, 1], dtype=float))


@pytest.mark.parametrize(
    'prediction',
    [
        '',
        'There is no bounding box in this answer.',
        '<bbox>10 20 30</bbox>',
    ],
)
def test_returns_none_without_four_numeric_values(prediction):
    assert RefCOCODataset._parse_prediction(prediction) is None


def test_non_string_input_returns_none():
    assert RefCOCODataset._parse_prediction(None) is None
