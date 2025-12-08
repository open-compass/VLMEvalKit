import re
import ast
import string
import pandas as pd


# ---------------------------------------------------------------------
# Basic text utilities
# ---------------------------------------------------------------------

def _clean_text(x) -> str:
    """Cast to str, collapse whitespace, strip."""
    s = str(x)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def _nonempty(v) -> bool:
    """Return True if v is not None/NaN/empty (after strip for scalars)."""
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return str(v).strip() != ""


# Only whitespace + punctuation (no letters / digits / CJK)
_ONLY_PUNCT_WS_RE = re.compile(r'^[\s\.\,\;\:\!\?\-\_/\\\|\(\)\[\]\{\}【】（）·・、—]+$')


def _normalize_choice_body(raw) -> str:
    """
    Normalize choice text:
      - None / NaN -> ''
      - Collapse whitespace
      - If only whitespace/punctuation -> ''
      - Keep '<image>' etc. as-is
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ''
    txt = _clean_text(raw)
    if not txt or _ONLY_PUNCT_WS_RE.fullmatch(txt):
        return ''
    return txt


def _parse_candidates(val, max_letter: str = 'F'):
    """
    Parse 'options' / 'candidates' into a list of cleaned strings.
    """
    # 1) already a list
    if isinstance(val, list):
        return [_clean_text(x) for x in val]

    # 2) string cases
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None

        # 2a) try stringified Python list
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [_clean_text(x) for x in parsed]
        except Exception:
            # If parsing fails, fall back to extracting choices from the string.
            pass

        # 2b) fallback: treat as a mini "options block"
        mapping = extract_choices_from_question(s, max_letter=max_letter)
        if mapping:
            out = []
            for ch in string.ascii_uppercase:
                if ch in mapping:
                    out.append(_clean_text(mapping[ch]))
                else:
                    if ch == 'A':
                        continue
                    break
            return out or None

        return None

    # 3) other types -> not supported
    return None


def _letters_upto(max_letter: str = 'F', n: int | None = None):
    """
    Letter set helper.

    If n is given and > 0: ['A', ..., up to n].
    Else: ['A', ..., max_letter].
    """
    max_letter = max_letter.upper()
    all_letters = list(string.ascii_uppercase)
    if n is not None and isinstance(n, int) and n > 0:
        return all_letters[:min(n, 26)]
    end_idx = all_letters.index(max_letter) + 1
    return all_letters[:end_idx]


# ---------------------------------------------------------------------
# Extract choices from explicit columns (A/B/C/...)
# ---------------------------------------------------------------------

def _extract_from_columns(item, letter_set=('A', 'B', 'C', 'D', 'E', 'F')):
    """
    Read choices from columns A–F (or extended):

      - column exists & nonempty -> normalized text
      - column exists & empty    -> ''
      - column missing           -> no key
    """
    return {
        ch: (_normalize_choice_body(item[ch]) if _nonempty(item[ch]) else '')
        for ch in letter_set
        if ch in item
    }


# ---------------------------------------------------------------------
# Extract choices from question text
# ---------------------------------------------------------------------

def _slice_by_markers(text: str, markers):
    """
    Given label markers (letter, end_idx, start_idx), slice text into segments:

      [end_i : start_{i+1})  as body for letter_i,
      last marker goes to end of text.
    """
    out = {}
    markers = sorted(markers, key=lambda x: x[2])
    for i, (ch, end_label, start_pos) in enumerate(markers):
        next_start = markers[i + 1][2] if i + 1 < len(markers) else len(text)
        raw = text[end_label:next_start]
        out[ch] = _normalize_choice_body(raw)
    return out


def _contiguous_prefix_len(keys_iterable):
    """
    Count how many letters we have consecutively from 'A'.

    {'A','B','C'} -> 3
    {'B','C'}     -> 0
    {'A','C'}     -> 1
    """
    s = {k.upper() for k in keys_iterable}
    k = 0
    for ch in string.ascii_uppercase:
        if ch in s:
            k += 1
        else:
            break
    return k


def extract_choices_from_question(q: str, max_letter: str = 'F') -> dict:
    """
    Heuristically parse choices from question text.

    Tries:
      - Line-based labels (each option starts a line)
      - Inline labels ("A. xxx B. yyy C. zzz")

    Returns: {'A': '...', 'B': '...', ...} or {}.
    """
    if not isinstance(q, str) or not q.strip():
        return {}
    text = q

    # Drop preamble before "Options:" / "选项:"
    m = re.search(r'(?i)(options?|选项)\s*[:：]?', text)
    if m:
        text = text[m.end():]

    letters = ''.join(_letters_upto(max_letter))

    # ---------------- Line-based: e.g. each option on its own line ----------------
    LINE_LABEL = re.compile(
        rf'(?mi)^[ \t]*'
        rf'(?:[*_`>•·\-]+\s*)?'
        rf'(?:[\(\[\{{（【]\s*)?'
        rf'([{letters}])'                  # A / B / ...
        rf'(?:\s*[\)\]\}}）】])?'
        rf'\s*[\.．:：\)\]】、-]\s*'        # A. / A) / A: ...
    )

    from_lines: dict[str, str] = {}

    for m in LINE_LABEL.finditer(text):
        ch = m.group(1).upper()

        line_end = text.find('\n', m.end())
        if line_end == -1:
            line_end = len(text)
        raw = text[m.end():line_end]
        body = _normalize_choice_body(raw)
        from_lines[ch] = body

    # ---------------- Inline: e.g. "A. foo  B. bar  C. baz" ----------------
    INLINE_LABEL = re.compile(
        rf'(?<![A-Za-z0-9])'
        rf'([{letters}])'
        rf'\s*[\.．:：\)\]】、-]\s*'
        rf'(?![A-Za-z0-9])'
    )
    inline_markers = [(m.group(1).upper(), m.end(), m.start())
                      for m in INLINE_LABEL.finditer(text)]
    from_inline = _slice_by_markers(text, inline_markers) if inline_markers else {}

    # Pick better candidate
    ls_k = _contiguous_prefix_len(from_lines.keys())
    il_k = _contiguous_prefix_len(from_inline.keys())

    if ls_k > il_k:
        chosen = from_lines
    elif il_k > ls_k:
        chosen = from_inline
    else:
        if len(from_lines) > len(from_inline):
            chosen = from_lines
        elif len(from_inline) > len(from_lines):
            chosen = from_inline
        else:
            chosen = from_lines or from_inline

    return chosen


# ---------------------------------------------------------------------
# Top-level: build_choices
# ---------------------------------------------------------------------

def build_choices(item: dict, max_letter: str = 'F') -> dict:
    """
    Build a choice dict for one item (row-like mapping).

    Priority:
      1) 'options' / 'candidates' (options > candidates)
      2) Columns A..max_letter
      3) Parse from question text
      4) Fallback: {}
    """
    # 1) options / candidates
    seq = None
    for key in ('options', 'candidates'):
        if key in item:
            parsed = _parse_candidates(item[key])
            if parsed:
                seq = parsed
                break

    if seq:
        letters = _letters_upto(n=len(seq))
        return {ch: (seq[i] if i < len(seq) else '') for i, ch in enumerate(letters)}

    # 2) A–F (or extended) columns
    letters = _letters_upto(max_letter=max_letter)
    from_cols = _extract_from_columns(item, letter_set=letters)
    if from_cols:
        return from_cols

    # 3) From question text
    q = item.get('question')
    if isinstance(q, str):
        from_q = extract_choices_from_question(q, max_letter=max_letter)
        if from_q:
            return from_q

    # 4) Nothing found
    return {}
