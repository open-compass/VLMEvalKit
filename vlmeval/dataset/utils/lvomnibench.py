import re

import numpy as np

from vlmeval.smp import load

QUESTION_TYPES = [
    'Attribute Perception',
    'Counting',
    'Event Understanding',
    'Human-Centric Understanding',
    'Logical Reasoning',
    'Music Perception',
    'Sound Inference',
    'Spatial Inference',
    'Temporal Inference',
]

AUDIO_TYPES = [
    'Speech',
    'Music',
    'Sound',
]

DIFFICULTIES = [
    'High',
    'Medium',
    'Low',
]

VIDEO_CATEGORIES = [
    'Diy & Cooking',
    'Entertainment',
    'Film & TV',
    'Lifestyle',
    'Record',
    'Sports',
]


def get_dimension_rating(data_path):
    data = load(data_path)

    rating = {
        'overall': '',
        'question_type': {k: [] for k in QUESTION_TYPES},
        'audio_type': {k: [] for k in AUDIO_TYPES},
        'difficulty': {k: [] for k in DIFFICULTIES},
        'video_category': {k: [] for k in VIDEO_CATEGORIES},
    }

    for i in range(len(data)):
        score = float(data.iloc[i]['score'])
        qt = data.iloc[i]['question_type']
        at = data.iloc[i]['audio_type']
        diff = data.iloc[i]['difficulty']
        vc = data.iloc[i]['video_category']

        rating['question_type'][qt].append(score)
        rating['audio_type'][at].append(score)
        rating['difficulty'][diff].append(score)
        rating['video_category'][vc].append(score)

    # Compute overall
    all_scores = []
    for scores in rating['question_type'].values():
        all_scores.extend(scores)
    valid_scores = [x for x in all_scores if x >= 0]
    rating['overall'] = f'{np.mean(valid_scores):.3f}' if valid_scores else '0.000'

    # Compute per-dimension
    for dim in ['question_type', 'audio_type', 'difficulty', 'video_category']:
        for key in list(rating[dim].keys()):
            scores = rating[dim][key]
            valid = [x for x in scores if x >= 0]
            rating[dim][key] = f'{np.mean(valid):.3f}' if valid else '0.000'

    return rating


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is'
        'The correct option is',
        'Best answer:'
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCD]', s):
        return ''
    matches = re.search(r'[ABCD]', s)
    if matches is None:
        return ''
    return matches[0]
