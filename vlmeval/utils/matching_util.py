import copy as cp
import os
import re
import string
from collections import Counter

from vlmeval.smp.log import get_logger

logger = get_logger(__name__)


# Matches verbose chain-of-thought answers such as:
#   "The correct answer is **B**. Here's why: ..."
#   "The answer is C."
# A lowercase "a" followed by a word is the English article ("the answer is a horse"), not option A.
_VERBOSE_ANSWER_RE = re.compile(r"(?i)(?:correct\s+)?answer\s+is\s+\**(?!(?-i:a)\s+[a-z])([ABCD])\b\**")


def can_infer_option(answer, choices):
    verbose = os.environ.get('VERBOSE', 0)
    # Choices is a dictionary
    if 'Failed to obtain answer via API' in answer:
        return False

    reject_to_answer = [
        "Sorry, I can't help with images of people yet.", "I can't process this file.",
        "I'm sorry, but without the image provided", 'Cannot determine the answer'
    ]
    for err in reject_to_answer:
        if err in answer:
            return 'Z'

    def count_choice(splits, choices, prefix='', suffix=''):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = cp.copy(answer)
    chars = '.()[],:;!*#{}'
    for c in chars:
        answer_mod = answer_mod.replace(c, ' ')

    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)

    if count == 1:
        for ch in choices:
            if 'A' in splits and len(splits) > 3 and verbose:
                logger.info(f'A might be a quantifier in the string: {answer}.')
                return False
            if ch in splits and splits.index(ch) > (len(splits) - 5):
                return ch
    elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
        return 'Z'

    match = _VERBOSE_ANSWER_RE.search(answer or "")
    if match and match.group(1).upper() in choices:
        return match.group(1).upper()

    return False


def can_infer_sequence(answer, choices=None):
    answer_upper = answer.upper()

    sequence_match = re.search(r'\b([A-D]{4})\b', answer_upper)
    if sequence_match:
        candidate = sequence_match.group(1)
        if len(set(candidate)) == 4:
            return candidate

    order_patterns = [
        r'(?:first|1st|首先|第一步).*?([A-D])',
        r'(?:second|2nd|其次|第二步).*?([A-D])',
        r'(?:third|3rd|再次|第三步).*?([A-D])',
        r'(?:fourth|4th|最后|第四步).*?([A-D])',
    ]

    sequence = []
    for pattern in order_patterns:
        match = re.search(pattern, answer_upper, re.IGNORECASE)
        if match:
            option = match.group(1).upper()
            if option not in sequence:
                sequence.append(option)

    if len(sequence) == 4:
        return ''.join(sequence)

    step_pattern = (r'(?:step\s*[\d一二三四]+|'
                    r'步骤\s*[\d一二三四]+|'
                    r'第\s*[\d一二三四]\s*步)'
                    r'.*?([A-D])')
    step_matches = re.findall(step_pattern, answer_upper, re.IGNORECASE)
    if len(step_matches) >= 4:
        unique = []
        for m in step_matches[:4]:
            if m.upper() not in unique:
                unique.append(m.upper())
        if len(unique) == 4:
            return ''.join(unique)

    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


# Explicit answer statements in chain-of-thought output, e.g.
#   "Final Answer: **D. $77,490**", "### **Answer:**\n**B. Project B**",
#   "the correct answer is (C)". Only the keywords are case-insensitive: the
# option itself must be an upper-case letter not followed by another letter,
# so "the answer is a horse" does not read as option A.
_ANSWER_STATEMENT_RE = re.compile(
    r"(?i:final\s+answer|correct\s+answer|answer)(?:\s+is)?\s*[:\uff1a]?[\s*#_]*"
    r"(?i:option\s+)?[\(\[]?([A-Z])(?![A-Za-z])"
)


def parse_answer_statement(response, choices):
    """Return the option named by the last explicit answer statement, else None."""
    response = str(response)
    found = None
    for m in _ANSWER_STATEMENT_RE.finditer(response):
        opt = m.group(1)
        # "Answer: A company ..." / "Answer: I think ..." are words, not options.
        if opt in 'AI' and re.match(r'\s+[a-z]', response[m.end():]):
            continue
        if opt in choices:
            found = opt
    return found


def parse_mmmu_choice(response, choices):
    """Rule-based MCQ parser from the official MMMU evaluation code.

    Port of ``parse_multi_choice_response`` in MMMU-Benchmark/MMMU
    (mmmu/utils/eval_utils.py), with one change: where MMMU picks a random
    option when nothing matches, this returns None so the caller can count
    the item as unanswered instead of scoring a coin flip.
    """
    response = str(response)
    for char in [',', '.', '!', '?', ';', ':', "'"]:
        response = response.strip(char)
    response = ' ' + response + ' '

    index_ans, ans_with_brack, candidates = True, False, []
    for choice in choices:
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in choices:
            if f'{choice} ' in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for index, ans in choices.items():
            if str(ans).lower() in response.lower():
                candidates.append(index)
                index_ans = False
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if index_ans:
        fmt = '({})' if ans_with_brack else ' {} '
        positions = [response.rfind(fmt.format(c)) for c in candidates]
    else:
        positions = [response.lower().rfind(str(choices[c]).lower()) for c in candidates]
    return candidates[positions.index(max(positions))]


def rule_based_mcq_answer(response, choices):
    """Judge-free fallback for MCQ answers that ``can_infer`` cannot read.

    Tries an explicit answer statement first, then the official MMMU parser.
    Returns ``(option, rule)`` or ``(None, None)``.
    """
    opt = parse_answer_statement(response, choices)
    if opt:
        return opt, 'answer statement'
    opt = parse_mmmu_choice(response, choices)
    if opt:
        return opt, 'MMMU parser'
    return None, None


def can_infer_lego(answer, question_type, choices):
    answer = str(answer)
    if question_type == 'sort':
        copt = can_infer_sequence(answer, choices)
    else:  # multiple-choice
        copt = can_infer_option(answer, choices)  # option
    return copt if copt else can_infer_text(answer, choices)


def detect_repetition(
    text: str,
    seg_len: int = 100,
    min_matches: int = 20,
    interval_kinds: int = 3,
    interval_threshold: float = 0.8,
) -> tuple[bool, list[re.Match] | None]:
    '''To detect whether the LLM out is in repetition.

    It cannot detect pattern like AB AC AD AE ...
    '''
    text = text.strip()
    # Skip too short text.
    if len(text) < seg_len * min_matches:
        return False, None
    # Use the last `seg_len` chars as the detect sample
    seg = text[-seg_len:]
    # The repetition only occurs in the end of text, only use the second half.
    text = text[len(text) // 2:]

    matches = list(re.finditer(re.escape(seg), text))
    if len(matches) < min_matches:
        return False, matches

    diff = Counter(post.start() - prev.start() for prev, post in zip(matches[:-1], matches[1:]))
    # Some repetition has multiple intervals, like
    # AB ABCD AB ABCD AB
    # Thus use the most common `interval_kinds` ratio
    if sum(item[1]
           for item in diff.most_common(interval_kinds)) / diff.total() > interval_threshold:
        return True, matches
    else:
        return False, matches
