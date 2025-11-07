import re
import ast


ZW_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')
_NUM_RE = re.compile(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?')

# Strongest signal: explicit <answer>...</answer> block with a leading letter (A–J allowed)
TAGGED_ANSWER_BLOCK = re.compile(
    r'<\s*answer\b[^>]*>\s*([A-Ja-j])(?:\s*[\.．:：\)\]】、])?.*?<\s*/\s*answer\s*>',
    flags=re.IGNORECASE | re.DOTALL
)

TAGGED_NUMERIC_ANSWER = re.compile(
    r'<\s*answer\b[^>]*>\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)(?:[^\d<][^<]*)?<\s*/\s*answer\s*>',
    flags=re.IGNORECASE | re.DOTALL
)


def can_match_option(
    answer_text: str,
    choices=None,
    tail_lines: int = 6,
    tail_window: int = 800
):
    """
    Extract a single-choice option letter from free-form model output.

    Return:
        - 'A'..'F' (or restricted by `choices`) if a reliable match is found
        - False otherwise

    Procedure (1 → 7):
      1) Dynamic letter set: build allowed letters (default A–F; shrink if `choices` given)
      2) Block-level: <answer>...</answer> with leading letter (A–J allowed here)
      3) Tail anchors: a letter immediately before </answer> or </think> within the tail window
      4) Last-lines (after think-tail): scan last N lines after the last </think> (or <think>)
         - full-line single letter (markdown/brackets allowed)
         - labeled line start "A. ..."/"B) ..."
         - unique inline uppercase token not part of words
      5) Last-lines (global tail): same scan on the last N lines of the whole tail window
      6) Phrase-style conclusion (tail window only), e.g. "final answer: B"
         - capture is UPPERCASE only; use the last occurrence
      7) Global fallback (strict): after removing enumeration lines, accept ONLY one
         unique UPPERCASE standalone token across the entire text
    """
    # 1) Dynamic letter set
    if not isinstance(answer_text, str):
        return False
    text = ZW_RE.sub('', answer_text.strip())

    if choices:
        letters_sorted = ''.join(sorted({str(c).strip().upper()[:1] for c in choices if str(c)}))
        letters = ''.join([ch for ch in 'ABCDEFGHIJ' if ch in letters_sorted]) or 'ABCDEF'
    else:
        letters = 'ABCDEF'

    # 2) Block-level <answer>...</answer>
    m_block = TAGGED_ANSWER_BLOCK.search(text)
    if m_block:
        return m_block.group(1).upper()

    # 3) Tail anchors: before </answer> / </think>
    tail_block = text[-tail_window:]
    PAT_ANS = re.compile(
        r'(?<![A-Za-z])[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?\s*</\s*answer\s*>' % letters,
        re.IGNORECASE
    )
    PAT_THINK = re.compile(
        r'(?<![A-Za-z])[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?\s*</\s*think\s*>' % letters,
        re.IGNORECASE
    )
    for pat in (PAT_ANS, PAT_THINK):
        m = pat.search(tail_block)
        if m:
            return m.group(1).upper()

    # Helpers shared by steps 4 & 5
    _PUNC_TIGHT = r"\.,:;!?\)\]】》」』，。；、：）】》」』"
    OPTION_LINE_PREFIX = re.compile(r'^(?:[*_>\-\s]*)(?:option|选项)\s+[A-J]\s*[:：]', re.IGNORECASE)
    MD_SINGLE = re.compile(r'^\s*[*_`>（）\[\]【】\(\)]*\s*([A-Fa-f])\s*[*_`（）\[\]【】\(\)]*\s*$')
    LINE_START_LABELED = re.compile(r'^\s*([A-F])\s*[\.．:：\)\]】、-]\s+', re.IGNORECASE)
    # Uppercase-only inline token (not part of words)
    TOKEN_INLINE = re.compile(
        r'(?<![A-Za-z])[*_`（\[\{\(]*\s*([A-F])\s*[*_`）\]\}\)]*(?=$|[\s%s])' % _PUNC_TIGHT
    )

    def _pick_from_lines(lines):
        for line in reversed([ln.strip() for ln in lines if ln.strip()]):
            if OPTION_LINE_PREFIX.search(line):
                continue
            m = MD_SINGLE.fullmatch(line)
            if m:
                return m.group(1).upper()
            m = LINE_START_LABELED.match(line)
            if m:
                return m.group(1).upper()
            tokens = [t.upper() for t in TOKEN_INLINE.findall(line)]
            if tokens:
                uniq = sorted(set(tokens))
                if len(uniq) == 1:
                    return uniq[0]
        return None

    # 4) Last-lines after think-tail
    if re.search(r'</\s*think\s*>', text, re.IGNORECASE):
        tail_segment = text[list(re.finditer(r'</\s*think\s*>', text, re.IGNORECASE))[-1].end():].strip()
    elif re.search(r'<\s*think\s*>', text, re.IGNORECASE):
        tail_segment = text[list(re.finditer(r'<\s*think\s*>', text, re.IGNORECASE))[-1].end():].strip()
    else:
        tail_segment = text

    pick = _pick_from_lines(tail_segment.splitlines()[-tail_lines:])
    if pick:
        return pick

    # 5) Last-lines in global tail window
    pick = _pick_from_lines(text[-tail_window:].splitlines()[-tail_lines:])
    if pick:
        return pick

    # 6) Phrase-style conclusion (tail window; last match; upper && lower CASE)
    PHRASE_AFTER = re.compile(
        r'(?i)(?:final\s*answer|the\s*answer\s*is|answer(?:\s*is)?|correct\s*answer|'
        r'答案|最终答案|结论|所以|因此|我选(?:择)?|选择|选)\s*[:：>＝=]?\s*'
        r'[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?(?:\b|[.)、。])' % letters
    )
    m = PHRASE_AFTER.search(text)
    if m:
        return m.group(1).upper()

    # 7) Global fallback: uppercase-only & unique (skip enumerations)
    cleaned_lines = []
    for ln in text.splitlines():
        if OPTION_LINE_PREFIX.search(ln):
            continue
        cleaned_lines.append(ln)
    cleaned = "\n".join(cleaned_lines)

    TOKEN_UPPER_GLOBAL = re.compile(
        r'(?<![A-Za-z])[\(\[\{（【]?\s*([%s])\s*[\)\]\}）】]?(?![A-Za-z])' % letters
    )
    tokens = TOKEN_UPPER_GLOBAL.findall(cleaned)  # uppercase-only by pattern
    uniq = sorted(set(tokens))
    if len(uniq) == 1:
        return uniq[0]

    return False


def _after_think(text: str) -> str:
    m_end = list(re.finditer(r'</\s*think\s*>', text, flags=re.IGNORECASE))
    if m_end:
        return text[m_end[-1].end():].strip()
    m_start = list(re.finditer(r'<\s*think\s*>', text, flags=re.IGNORECASE))
    if m_start:
        return text[m_start[-1].end():].strip()
    return text


def _first_number(s: str):
    m = _NUM_RE.search(s)
    return float(m.group()) if m else None


def can_match_na(pred):
    try:
        if isinstance(pred, list):
            candidates = [str(pred[0])] if pred else []
        elif isinstance(pred, str) and pred.strip().startswith('[') and pred.strip().endswith(']'):
            seq = ast.literal_eval(pred)  # safer than eval
            candidates = [str(seq[0])] if isinstance(seq, list) and seq else [pred]
        else:
            candidates = [str(pred)]

        for raw in candidates:
            text = ZW_RE.sub('', raw.strip())

            # <answer>
            m = TAGGED_NUMERIC_ANSWER.search(text)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    pass

            # after </think>
            tail = _after_think(text)
            v = _first_number(tail)
            if v is not None:
                return v

            # first number
            v = _first_number(text)
            if v is not None:
                return v

        return None
    except Exception:
        return None
