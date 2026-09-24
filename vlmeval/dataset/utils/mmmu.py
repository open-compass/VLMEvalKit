"""Shared MMMU open-answer parsing.

Adapted from the official evaluator:
https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py
"""

import re


def check_is_number(value):
    """Return whether a value can be interpreted as a number."""
    try:
        float(str(value).replace(',', ''))
        return True
    except ValueError:
        return False


def normalize_str(value):
    """Normalize an MMMU open answer to strings or a rounded number."""
    value = str(value).strip()
    if check_is_number(value):
        return [round(float(value.replace(',', '')), 2)]

    value = value.lower()
    if len(value) == 1:
        return [' ' + value, value + ' ']
    return [value]


def extract_numbers(value):
    """Extract comma-separated, scientific-notation, and simple numbers."""
    pattern_commas = r'-?\b\d{1,3}(?:,\d{3})+\b'
    pattern_scientific = r'-?\d+(?:\.\d+)?[eE][+-]?\d+'
    pattern_simple = r'-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])'
    return (
        re.findall(pattern_commas, value)
        + re.findall(pattern_scientific, value)
        + re.findall(pattern_simple, value)
    )


def parse_open_response(response):
    """Parse candidate strings and numbers from an MMMU open response."""
    if response == 'API Error' or response == '':
        return 'API Error'

    def get_key_subresponses(value):
        value = value.strip().strip('.').lower()
        sub_responses = re.split(r'\.\s(?=[A-Z])|\n', value)
        indicators = [
            'could be ',
            'so ',
            'is ',
            'thus ',
            'therefore ',
            'final ',
            'answer ',
            'result ',
            'are ',
            'in total ',
            'total ',
            'identify ',
            'recognize ',
            'calculated as ',
            'counted as ',
            'measured as ',
            'observed as ',
            'concluded as ',
            'found to be ',
            'equals ',
            'determined to be ',
            'number of ',
            'value is ',
            'adds up to ',
            'have ',
            'has ',
        ]
        key_responses = []
        for index, item in enumerate(sub_responses):
            current_indicators = indicators + (['='] if index == len(sub_responses) - 1 else [])
            candidates = [item.split(indicator)[-1].strip() for indicator in current_indicators if indicator in item]
            if candidates:
                shortest = min(candidates, key=len)
                if shortest not in {':', ',', '.', '!', '?', ';', "'"}:
                    key_responses.append(shortest)
        return key_responses or [value]

    key_responses = get_key_subresponses(str(response))
    predictions = key_responses.copy()
    for item in key_responses:
        predictions.extend(extract_numbers(item))

    normalized = []
    for item in predictions:
        normalized.extend(normalize_str(item))
    return list(dict.fromkeys(normalized))


def eval_open(gold, predictions):
    """Evaluate an MMMU open response using the official matching policy."""
    if predictions == 'API Error':
        return False

    normalized_answers = []
    answers = gold if isinstance(gold, list) else [gold]
    for answer in answers:
        normalized_answers.extend(normalize_str(answer))

    for prediction in predictions:
        if isinstance(prediction, str):
            if any(isinstance(answer, str) and answer in prediction for answer in normalized_answers):
                return True
        elif prediction in normalized_answers:
            return True
    return False
