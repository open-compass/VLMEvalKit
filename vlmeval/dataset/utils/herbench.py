import re

import numpy as np

from vlmeval.smp import load
from .multiple_choice import extract_answer_from_item

FAIL_MSG = 'Failed to obtain answer via API.'

# The 12 HERBench compositional task types (as stored in the 'task_type' field)
TASK_TYPES = [
    'Temporal Shot Ordering',
    'Multi-Person Duration Reasoning',
    'Action Sequence Integrity Identification',
    'Appearance-Grounded Behavior & Interactions',
    'Appearance-Grounded Attribute Recognition',
    'Appearance-Grounded Localization & Trajectory',
    'False Action Memory',
    'Scene Verification & Arrangement',
    'False Object Memory',
    'Multi-Entities Grounding and Localization',
    'Action Counting',
    'Region Localized People Counting',
]


def get_dimension_rating(data_path):
    data = load(data_path)

    task_types = sorted(set(data['task_type']))
    rating = {
        'overall': '',
        'task_type': {t: [] for t in task_types},
        'source_dataset': {},
    }

    for i in range(len(data)):
        row = data.iloc[i]
        rating['task_type'][row['task_type']].append(row['score'])
        rating['source_dataset'].setdefault(row['source_dataset'], []).append(row['score'])

    all_scores = [x for scores in rating['task_type'].values() for x in scores if x >= 0]
    rating['overall'] = f'{np.mean(all_scores):.4f}' if all_scores else ''
    for key in ('task_type', 'source_dataset'):
        for name, scores in rating[key].items():
            valid = [x for x in scores if x >= 0]
            rating[key][name] = f'{np.mean(valid):.4f}' if valid else ''
    return rating


def extract_option(model, input_item, dataset_name):
    """LLM-judge fallback: expose the choice texts as per-letter keys so that
    extract_answer_from_item can build the option list."""
    if model is None:
        return ''
    import json

    candidates = input_item.get('candidates', '[]')
    if not isinstance(candidates, list):
        candidates = json.loads(candidates)
    for i, text in enumerate(candidates):
        input_item[chr(ord('A') + i)] = text
    return extract_answer_from_item(model, input_item, dataset_name)['opt']


def extract_characters_regex(s, letters='ABCDE'):
    """Extract the answer letter from a model response, restricted to the
    letters actually offered for this question (a few questions have 4 options).

    Port of the official HERBench answer extraction
    (evaluation/model_wrappers/base_vlm.py::extract_answer_choice),
    returning '' instead of 'ERROR' when no letter is found.
    """
    s = str(s).strip()
    if not s:
        return ''

    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Final answer:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '').replace(answer_prefix.lower(), '').strip()

    letter_set = ''.join(letters)
    patterns = [
        rf'^([{letter_set}])[\s\.\,\)\:]',       # letter at start followed by a delimiter
        rf'^\(([{letter_set}])\)',               # letter in parentheses at start
        rf'^([{letter_set}])$',                  # just the letter alone
        rf'[Aa]nswer:\s*\(?([{letter_set}])\b',  # "Answer: A"
        rf'[Cc]hoice:\s*\(?([{letter_set}])\b',  # "Choice: A"
        rf'\b([{letter_set}])\b[\.\,]',          # standalone letter followed by punctuation
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            return match.group(1).upper()

    # standalone letter surrounded by non-letters (avoids the 'B' in 'Based', etc.)
    upper = s.upper()
    for match in re.finditer(rf'([{letter_set}])', upper):
        start, end = match.start(1), match.end(1)
        before_ok = start == 0 or not upper[start - 1].isalpha()
        after_ok = end == len(upper) or not upper[end].isalpha()
        if before_ok and after_ok:
            return match.group(1)

    return ''
