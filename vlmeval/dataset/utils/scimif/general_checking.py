import ast
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

EN_NUM = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10
}
_NUMBER_TOKEN_RE = re.compile('-?\\d+(?:,\\d{3})*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?')
_SCI_NUMBER_TOKEN_RE = re.compile('-?\\d+(?:,\\d{3})*(?:\\.\\d+)?[eE][+-]?\\d+')
_SUPERSCRIPT_TO_ASCII = str.maketrans({
    '⁰': '0',
    '¹': '1',
    '²': '2',
    '³': '3',
    '⁴': '4',
    '⁵': '5',
    '⁶': '6',
    '⁷': '7',
    '⁸': '8',
    '⁹': '9',
    '⁺': '+',
    '⁻': '-'
})


def _sentence_has_final_answer_phrase(s: str) -> bool:
    if not s or not s.strip():
        return False
    if re.search('final(?:\\s+numerical)?\\s+answer|final_answer|final_numerical_answer', s, re.I):
        return True
    return bool(re.search('\\b(?:the\\s+)?answer\\s+is\\b|\\bfinal\\s+conclusion\\b|\\bresult\\s+is\\b', s, re.I))


def _normalize_latex_scientific_for_eval(text: str) -> str:
    if not text:
        return text
    s = text.replace('$', ' ')
    s = re.sub('\\\\(?:text|mathrm|textrm)\\s*\\{e([+-])\\}\\s*(\\d+)', 'e\\1\\2', s, flags=re.I)
    s = re.sub('\\\\(?:text|mathrm|textrm)\\s*\\{[eE]\\}', 'e', s)
    s = re.sub('(-?\\d+(?:\\.\\d+)?)\\s+e\\s*([+-]?\\d+)', '\\1e\\2', s, flags=re.I)
    s = re.sub('(-?\\d+(?:,\\d{3})*(?:\\.\\d+)?)\\s*(?:\\\\times|×|[xX])\\s*10\\s*\\^\\s*\\{?\\s*([+-]?\\d+)\\s*\\}?',
               '\\1e\\2', s)

    def replace_unicode_exponent(match: re.Match) -> str:
        exponent = match.group(2).translate(_SUPERSCRIPT_TO_ASCII)
        return f'{match.group(1)}e{exponent}'

    s = re.sub('(-?\\d+(?:,\\d{3})*(?:\\.\\d+)?)\\s*(?:\\\\times|×|[xX])\\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)',
               replace_unicode_exponent, s)
    return s


def _split_for_final_answer_scope(response: str) -> List[str]:
    return [p.strip() for p in re.split('[。!?\\n]|(?<!\\d)\\.(?!\\d)', response) if p.strip()]


def _mask_latex_sup_sub_for_numeric_scan(s: str) -> str:
    s = re.sub('\\^\\{[^}]*\\}', '^', s)
    s = re.sub('_\\{[^}]*\\}', '_', s)
    return s


def _should_skip_numeric_token_at(text: str, start: int) -> bool:
    if start <= 0:
        return False
    prev = text[start - 1]
    if prev in '^_' or prev.isalpha():
        return True
    if start >= 2 and text[start - 2] in '^_' and (text[start - 1] == '{'):
        return True
    return False


def _last_number_token_in_response(response: str) -> Optional[str]:
    t = _mask_latex_sup_sub_for_numeric_scan(_normalize_latex_scientific_for_eval(response))
    last: Optional[str] = None
    for m in _NUMBER_TOKEN_RE.finditer(t):
        if _should_skip_numeric_token_at(t, m.start()):
            continue
        last = m.group(0)
    return last


def _last_scientific_number_token_in_response(response: str) -> Optional[str]:
    t = _mask_latex_sup_sub_for_numeric_scan(_normalize_latex_scientific_for_eval(response))
    matches = list(_SCI_NUMBER_TOKEN_RE.finditer(t))
    if not matches:
        return None
    return matches[-1].group(0)


def _decimal_digits_in_token(num: str) -> int:
    clean_num = num.replace(',', '')
    if 'e' in clean_num.lower():
        base = clean_num.lower().split('e')[0]
        return len(base.split('.')[-1]) if '.' in base else 0
    return len(clean_num.split('.')[-1]) if '.' in clean_num else 0


def _parse_decimal_places_hint(text: str) -> Optional[int]:
    if not text or not str(text).strip():
        return None
    t = str(text).strip()
    low = t.lower()
    m = re.search('(\\d+)\\s*decimal\\s*places?', low) or re.search('(\\d+)\\s*decimal\\b', low)
    if m:
        return int(m.group(1))
    m = re.search('\\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\\s+decimal\\s*places?\\b', low)
    if m:
        return EN_NUM.get(m.group(1))
    if re.search('nearest\\s+integer|0\\s+decimal\\s*places?|no\\s+decimal', low):
        return 0
    m = re.search('^\\s*(\\d+)\\s*$', t)
    if m:
        return int(m.group(1))
    return None


def _resolve_decimal_required_digits(required_parameters: str, edit_question: str) -> Tuple[Optional[int], str]:
    rp = (required_parameters or '').strip()
    eq = (edit_question or '').strip()
    for label, blob in (('required_parameters', rp), ('edit_question', eq)):
        d = _parse_decimal_places_hint(blob)
        if d is not None:
            return (d, label)
    return (None, '')


def _requires_whole_response_numeric_format(edit_question: str) -> bool:
    low = re.sub('\\s+', ' ', (edit_question or '').lower())
    patterns = (
        '\\ball intermediate and final numerical (?:values|results|quantities)\\b',
        '\\ball intermediate (?:calculations|steps|values|results).{0,100}\\bfinal (?:answer|result)\\b',
        '\\ball numerical (?:quantities|values|results|answers).{0,80}\\b(?:output|response|calculation|calculations|reasoning|solution)\\b',  # noqa: E501
        '\\bevery (?:numeric|numerical) (?:value|quantity|result).{0,80}\\b(?:output|response|calculation|calculations|reasoning|solution)\\b',  # noqa: E501
        '\\b(?:entire|whole) (?:output|response).{0,80}\\b(?:decimal|precision|scientific notation|scientific annotation)\\b'  # noqa: E501
    )
    return any((re.search(pattern, low) for pattern in patterns))


def _extract_final_answer_region(response: str) -> Tuple[Optional[str], str]:
    try:
        parsed = json.loads(response)
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = None
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            normalized_key = re.sub('[\\s-]+', '_', str(key).strip().lower())
            if normalized_key in {'final_answer', 'final_numerical_answer'}:
                if isinstance(value, str):
                    return (value, f'JSON field {key!r}')
                return (json.dumps(value, ensure_ascii=False), f'JSON field {key!r}')
    marker_re = re.compile(
        '\\bfinal(?:\\s+numerical)?\\s+answer\\b|\\bfinal_(?:numerical_)?answer\\b|\\bthe\\s+answer\\s+is\\b|\\bresult\\s+is\\b',  # noqa: E501
        re.I)
    matches = list(marker_re.finditer(response))
    if matches:
        last = matches[-1]
        return (response[last.start():], 'final-answer')
    return (None, '')


def _numeric_tokens(text: str) -> List[str]:
    normalized = _mask_latex_sup_sub_for_numeric_scan(_normalize_latex_scientific_for_eval(text))
    tokens: List[str] = []
    for match in _NUMBER_TOKEN_RE.finditer(normalized):
        if _should_skip_numeric_token_at(normalized, match.start()):
            continue
        line_start = normalized.rfind('\n', 0, match.start()) + 1
        prefix = normalized[line_start:match.start()]
        suffix = normalized[match.end():]
        structural_prefix = re.sub('[\\s#>*_`-]+', '', prefix)
        if not structural_prefix and re.match('\\s*[.)]', suffix):
            continue
        tokens.append(match.group(0))
    return tokens


def _format_bad_tokens(tokens: List[str], limit: int = 5) -> str:
    shown = ', '.join((repr(token) for token in tokens[:limit]))
    if len(tokens) > limit:
        shown += f'total{len(tokens)}'
    return shown


def check_decimal_format(response: str, required_parameters: str, item: Dict = None, **kwargs) -> Dict:
    if not isinstance(response, str) or not response.strip():
        return {'score': 0.0, 'detail': 'The response is empty'}
    eq = item.get('edit_question', '') or '' if item else ''
    required_digits, src = _resolve_decimal_required_digits(required_parameters or '', eq)
    if required_digits is None:
        return {
            'score': 0.0,
            'detail': f'Unable to parsedecimal placesrequired。required_parameters={required_parameters!r}'
        }
    if _requires_whole_response_numeric_format(eq):
        target_text = response
        scope = 'question requiredcheckthe entire response ( intermediate calculations)'
        tokens = _numeric_tokens(target_text)
    else:
        target_text, located_by = _extract_final_answer_region(response)
        if target_text is not None:
            scope = f'checkfinal answer ({located_by}）'
            tokens = _numeric_tokens(target_text)
        else:
            scope = 'question required Not foundanswer , checkresponse'
            all_tokens = _numeric_tokens(response)
            tokens = all_tokens[-1:] if all_tokens else []
    if not tokens:
        return {'score': 0.0, 'detail': f'{scope}, Not detected'}
    bad_tokens = [token for token in tokens if _decimal_digits_in_token(token) != required_digits]
    score = 0.0 if bad_tokens else 1.0
    requirement_detail = (f'{required_digits}required ({src}）'
                          if score else f'does not satisfy{required_digits}required:{_format_bad_tokens(bad_tokens)}')
    return {'score': score, 'detail': f'{scope}；check {len(tokens)},{requirement_detail}'}


def check_scientific_format(response: str, required_parameters: str = None, item: Dict = None, **kwargs) -> Dict:
    if not isinstance(response, str) or not response.strip():
        return {'score': 0.0, 'detail': 'The response is empty'}
    eq = item.get('edit_question', '') or '' if item else ''
    if _requires_whole_response_numeric_format(eq):
        target_text = response
        scope = 'question requiredcheckthe entire response ( intermediate calculations)'
        tokens = _numeric_tokens(target_text)
    else:
        target_text, located_by = _extract_final_answer_region(response)
        if target_text is not None:
            scope = f'checkfinal answer ({located_by}）'
            tokens = _numeric_tokens(target_text)
        else:
            scope = 'question required Not foundanswer , checkresponse'
            all_tokens = _numeric_tokens(response)
            tokens = all_tokens[-1:] if all_tokens else []
    if not tokens:
        return {'score': 0.0, 'detail': f'{scope}, Not detected'}
    bad_tokens = [token for token in tokens if _SCI_NUMBER_TOKEN_RE.fullmatch(token) is None]
    score = 0.0 if bad_tokens else 1.0
    format_detail = 'scientific notation' if score else f'scientific notation:{_format_bad_tokens(bad_tokens)}'
    return {'score': score, 'detail': f'{scope}；check {len(tokens)},{format_detail}'}


def is_box_wrapped(text: str) -> bool:
    lines = text.splitlines()
    if len(lines) < 3:
        return False
    top = lines[0]
    bottom = lines[-1]
    middle = lines[1:-1]
    if not (top.startswith('┌') and top.endswith('┐')):
        return False
    if not (bottom.startswith('└') and bottom.endswith('┘')):
        return False
    for line in middle:
        if not (line.startswith('│') and line.endswith('│')):
            return False
    return True


def is_latex_box_wrapped(text: str) -> bool:
    if not text:
        return False
    return bool(re.search('\\\\(?:boxed|fbox)\\s*\\{', text, re.I))


def check_wrap_up(response: str, required_parameters: str, item: Dict = None, **kwargs) -> Dict:
    if not isinstance(response, str) or not response.strip():
        return {'score': 0.0, 'detail': 'The response is empty'}
    wrapper = required_parameters.strip()
    res = response.strip()
    success = False
    if not wrapper:
        if is_box_wrapped(res):
            return {'score': 1.0, 'detail': 'box (┌─┐ │ │ └─┘)'}
        if is_latex_box_wrapped(res):
            return {'score': 1.0, 'detail': 'Detected a LaTeX box wrapper'}
        if res.startswith('```') and res.endswith('```') and (len(res) >= 6):
            return {'score': 1.0, 'detail': '``` ```'}
        if res.startswith('[') and res.endswith(']'):
            return {'score': 1.0, 'detail': 'Detected square-bracket wrapping'}
        if res.startswith('(') and res.endswith(')'):
            return {'score': 1.0, 'detail': '( )'}
        if res.startswith('{') and res.endswith('}'):
            return {'score': 1.0, 'detail': '{ }'}
        return {'score': 0.0, 'detail': 'No supported wrapper was detected'}
    if wrapper.startswith('```'):
        success = res.startswith(wrapper) and res.endswith('```')
    elif wrapper == '[ ]':
        success = res.startswith('[') and res.endswith(']')
    elif wrapper == '( )':
        success = res.startswith('(') and res.endswith(')')
    elif wrapper == '{ }':
        success = res.startswith('{') and res.endswith('}')
    elif wrapper == 'box':
        success = is_box_wrapped(res)
    elif wrapper.lower() in ('boxed', '\\boxed', 'latex_boxed', 'fbox', '\\fbox'):
        success = is_latex_box_wrapped(res)
    elif wrapper.startswith('<') and wrapper.endswith('>'):
        tag = re.escape(wrapper.strip('<>'))
        success = bool(re.match(f'^<{tag}[^>]*>.*?</{tag}>$', res, re.DOTALL))
    else:
        success = res.startswith(wrapper) and res.endswith(wrapper)
    if success:
        return {'score': 1.0, 'detail': f'{wrapper}'}
    return {'score': 0.0, 'detail': f'{wrapper}correct'}


_CASE_WORD_RE = re.compile('(?:all\\s+)?(?:lower|upper)[\\s\\-]?case|uppercase|lowercase', re.I)
_ANSWER_LOC_RE = re.compile(
    '(final\\s*numerical\\s*answer|final\\s*answer|the\\s+answer\\s+is|final_answer|final_numerical_answer)', re.I)


def _normalize_case_text(text: str) -> str:
    t = (text or '').lower()
    t = t.replace('lower-case', 'lowercase').replace('upper-case', 'uppercase')
    t = t.replace('lower case', 'lowercase').replace('upper case', 'uppercase')
    return t


def _case_windows(text: str, radius: int = 160) -> str:
    wins = []
    for m in _CASE_WORD_RE.finditer(text or ''):
        a = max(0, m.start() - radius)
        b = min(len(text), m.end() + radius)
        wins.append(text[a:b])
    return ' || '.join(wins) if wins else text or ''


def infer_casing_scope(edit_question: str, required_parameters: str = '') -> str:
    full = f"{edit_question or ''}\n{required_parameters or ''}"
    q = _normalize_case_text(full)
    if not _CASE_WORD_RE.search(q):
        return 'skip'
    ctx = _case_windows(q)
    if re.search(
            'output\\s+answer\\s+format|entire\\s+output\\s+answer|final\\s+output\\s+answer|(?:final\\s+)?answer\\s+format|entire\\s+answer\\b|final\\s+answer.{0,60}(?:lower|upper)case|(?:lower|upper)case.{0,40}final\\s+answer|answer\\s+must\\s+be\\s+(?:written\\s+|formatted\\s+|provided\\s+)?in\\s+all\\s+(?:lower|upper)case|format(?:ted)?\\s+your\\s+answer.{0,60}(?:lower|upper)case|answer\\s+as\\b.{0,80}(?:lower|upper)case|label\\s+values?.{0,40}(?:lower|upper)case|values?\\s+in\\s+(?:all\\s+)?(?:lower|upper)case|string\\s+values?.{0,40}(?:lower|upper)case|final\\s+list.{0,40}(?:lower|upper)case|(?:lower|upper)case.{0,40}final\\s+list|output\\s+must\\s+be\\s+the\\s+name.{0,40}(?:lower|upper)case|in\\s+all\\s+(?:lower|upper)case\\s+letters',  # noqa: E501
            ctx,
            re.S):
        if re.search(
                'entire\\s+(?:response|output|reply)\\b(?!\\s+answer)|presented\\s+entirely\\s+in\\s+(?:all\\s+)?(?:lower|upper)case|your\\s+entire\\s+output\\s+must\\s+be\\s+in\\s+all\\s+(?:lower|upper)case',  # noqa: E501
                ctx,
                re.S):
            return 'whole'
        return 'answer'
    if re.search(
            '(?:entire\\s+)?(?:response|output|reply)\\b.{0,50}(?:all\\s+)?(?:lower|upper)case|(?:lower|upper)case.{0,50}(?:entire\\s+)?(?:response|output|reply)\\b|presented\\s+entirely\\s+in\\s+(?:all\\s+)?(?:lower|upper)case|your\\s+entire\\s+output\\s+must\\s+be\\s+in\\s+all\\s+(?:lower|upper)case|reponse\\s+must\\s+be\\s+output\\s+in\\s+all\\s+(?:lower|upper)case|output\\s+must\\s+be\\s+in\\s+all\\s+(?:lower|upper)case|output\\s+(?:should|must)\\s+be\\s+in\\s+(?:all\\s+)?(?:lower|upper)case|ouput\\s+must\\s+be\\s+in\\s+(?:lower|upper)case',  # noqa: E501
            ctx,
            re.S):
        if re.search('output\\s+answer', ctx):
            return 'answer'
        return 'whole'
    if re.search('\\banswer\\b.{0,80}(?:lower|upper)case|(?:lower|upper)case.{0,80}\\banswer\\b', ctx, re.S):
        return 'answer'
    if re.search('(?:lower|upper)case', ctx):
        return 'answer'
    return 'skip'


def _collect_json_string_values(obj: Any) -> List[str]:
    vals: List[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, str):
            vals.append(x)

    walk(obj)
    return vals


def extract_answer_span_for_casing(response: str) -> Tuple[Optional[str], str]:
    if not isinstance(response, str) or not response.strip():
        return (None, 'empty')
    text = response.strip()
    m = re.search('\\\\(?:boxed|fbox)\\s*\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text, re.I)
    if m and m.group(1).strip():
        return (m.group(1).strip(), 'boxed')
    m = _ANSWER_LOC_RE.search(text)
    if m:
        tail = text[m.end():].lstrip(' \t:：-')
        block = re.split('\\n\\s*\\n', tail, maxsplit=1)[0].strip()
        if not block:
            line = tail.split('\n', 1)[0].strip()
            block = line
        if block:
            return (block, 'answer_phrase')
    cleaned = re.sub('^```[a-zA-Z]*\\s*', '', text)
    cleaned = re.sub('\\s*```$', '', cleaned).strip()
    try:
        obj = json.loads(cleaned)
        vals = _collect_json_string_values(obj)
        if vals:
            return (' '.join(vals), 'json_values')
    except Exception:
        pass
    fence = re.search('```(?:json)?\\s*(\\{.*?\\}|\\[.*?\\])\\s*```', text, re.I | re.S)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            vals = _collect_json_string_values(obj)
            if vals:
                return (' '.join(vals), 'json_values')
        except Exception:
            pass
    return (None, 'unresolved')


def _letters_case_ok(text: str, mode: str) -> bool:
    letters = [c for c in text if 'A' <= c <= 'Z' or 'a' <= c <= 'z']
    if not letters:
        return False
    if mode == 'upper':
        return all(('A' <= c <= 'Z' for c in letters))
    return all(('a' <= c <= 'z' for c in letters))


def _check_casing(response: str, mode: str, item: Dict = None, required_parameters: str = '', **kwargs) -> Dict:
    if not isinstance(response, str) or not response.strip():
        return {'score': 0.0, 'detail': 'The content is empty', 'skipped': False}
    eq = ''
    rp = required_parameters or ''
    if item and isinstance(item, dict):
        eq = item.get('edit_question', '') or ''
        if not rp:
            rp = kwargs.get('required_parameters', '') or ''
    rp = required_parameters or kwargs.get('required_parameters', '') or rp
    scope = infer_casing_scope(eq, rp)
    if scope == 'skip':
        return {'score': 0.0, 'detail': 'lowercase , skipped instruction ( )', 'skipped': True, 'casing_scope': 'skip'}
    if scope == 'whole':
        target, src = (response, 'whole_response')
    else:
        target, src = extract_answer_span_for_casing(response)
        if target is None:
            return {
                'score': 0.0,
                'detail': f'casing_scope=answer, answer ({src}）',
                'skipped': False,
                'casing_scope': 'answer'
            }
    ok = _letters_case_ok(target, mode)
    label = 'uppercase' if mode == 'upper' else 'lowercase'
    if ok:
        return {
            'score': 1.0,
            'detail': f'casing_scope={scope}, src={src},{label}',
            'skipped': False,
            'casing_scope': scope
        }
    return {
        'score': 0.0,
        'detail': f'casing_scope={scope}, src={src}, some letters are not {label}',
        'skipped': False,
        'casing_scope': scope
    }


def check_uppercase(response: str, item: Dict = None, required_parameters: str = '', **kwargs) -> Dict:
    return _check_casing(response, 'upper', item=item, required_parameters=required_parameters, **kwargs)


def check_lowercase(response: str, item: Dict = None, required_parameters: str = '', **kwargs) -> Dict:
    return _check_casing(response, 'lower', item=item, required_parameters=required_parameters, **kwargs)


_FINAL_ANSWER_RE = re.compile(
    'final[\\s_-]+(?:numerical[\\s_-]+)?answer\\s*[:：]?|the\\s+answer\\s+is\\s*(?:[:：]|\\.(?=\\s|$))?|(?<!final\\s)\\banswer\\s+is\\s*(?:[:：]|\\.(?=\\s|$))?|final\\s+conclusion\\s*[:：]?|\\bresult\\s+is\\s*(?:[:：]|\\.(?=\\s|$))?',  # noqa: E501
    re.I)


def infer_format_scope(edit_question: str) -> str:
    q = (edit_question or '').lower()
    strong_whole_patterns = [
        'entire\\s+(?:final\\s+)?(?:response|output|model\\s+output|reply)',
        'format\\s+your\\s+entire\\s+(?:response|output|reply)',
        'all\\s+(?:of\\s+the\\s+)?(?:output|content)\\s+must\\s+be\\s+(?:formatted\\s+)?(?:as|in)', 'output\\s+only\\b',
        'no\\s+additional\\s+(?:text|content|explanation)'
    ]
    if any((re.search(p, q, re.I) for p in strong_whole_patterns)):
        return 'whole'
    if re.search('\\bfinal(?:[\\s-]+\\w+){0,4}[\\s-]+(?:answer|output)\\b', q, re.I):
        return 'final'
    weak_whole = '(?:your|the)\\s+(?:response|reply)\\s+must\\s+be\\s+(?:valid\\s+)?(?:formatted\\s+)?(?:as|in)\\s+(?:a\\s+)?(?:json|list|tuple|dictionary|markdown|html|xml|csv)\\b'  # noqa: E501
    if re.search(weak_whole, q, re.I):
        return 'whole'
    return 'final'


def _strip_outer_code_fence(text: str) -> str:
    s = (text or '').strip()
    m = re.fullmatch('```(?:[A-Za-z0-9_+.-]+)?\\s*\\n?([\\s\\S]*?)\\n?```', s)
    return m.group(1).strip() if m else s


def _extract_last_code_fence(text: str) -> Optional[str]:
    matches = list(re.finditer('```(?:[A-Za-z0-9_+.-]+)?\\s*\\n?([\\s\\S]*?)\\n?```', text or ''))
    return matches[-1].group(1).strip() if matches else None


def _extract_balanced_braces(text: str, open_pos: int) -> Optional[str]:
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[open_pos + 1:i].strip()
    return None


def _unwrap_answer_container(text: str) -> str:
    s = _strip_outer_code_fence(text)
    boxed = list(re.finditer('\\\\(?:boxed|fbox)\\s*\\{', s, re.I))
    if boxed:
        open_pos = s.find('{', boxed[-1].start())
        inner = _extract_balanced_braces(s, open_pos)
        if inner:
            s = inner
    s = s.strip()
    if s.startswith('\\[') and s.endswith('\\]'):
        s = s[2:-2].strip()
    if s.startswith('$') and s.endswith('$') and (len(s) >= 2):
        s = s[1:-1].strip()
    return s


def extract_format_target(response: str, edit_question: str) -> Tuple[str, str, str]:
    scope = infer_format_scope(edit_question)
    if scope == 'whole':
        return (_strip_outer_code_fence(response), scope, 'whole_response')
    matches = list(_FINAL_ANSWER_RE.finditer(response or ''))
    if matches:
        tail = (response or '')[matches[-1].end():].strip()
        tail = re.sub(
            '^(?:\\((?:tuple|list|dictionary|dict|json|html|xml|csv|markdown|answer)[^()\\n]{0,40}\\))?\\s*[:：]?\\s*\\*{0,2}\\s*',  # noqa: E501
            '',
            tail,
            flags=re.I).strip()
        if tail:
            return (_unwrap_answer_container(tail), scope, 'answer_phrase')
    boxed = list(re.finditer('\\\\(?:boxed|fbox)\\s*\\{', response or '', re.I))
    if boxed:
        open_pos = (response or '').find('{', boxed[-1].start())
        inner = _extract_balanced_braces(response or '', open_pos)
        if inner:
            return (_unwrap_answer_container(inner), scope, 'boxed_answer')
    fenced = _extract_last_code_fence(response or '')
    if fenced:
        return (_unwrap_answer_container(fenced), scope, 'last_code_fence')
    markup = re.search('(<([A-Za-z_][\\w:.-]*)\\b[^>]*>[\\s\\S]*</\\2>)\\s*$', (response or '').strip(), re.I)
    if markup:
        return (markup.group(1).strip(), scope, 'final_markup')
    blocks = [p.strip() for p in re.split('\\n\\s*\\n', response or '') if p.strip()]
    target = blocks[-1] if blocks else (response or '').strip()
    return (_unwrap_answer_container(target), scope, 'last_block')


def _literal_value(text: str):
    cleaned = _unwrap_answer_container(text)
    try:
        return json.loads(cleaned)
    except Exception:
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None


def _valid_list_target(text: str) -> bool:
    value = _literal_value(text)
    if isinstance(value, list):
        return True
    cleaned = _unwrap_answer_container(text)
    if re.search('\\\\begin\\{(?:itemize|enumerate)\\}[\\s\\S]*\\\\item\\b[\\s\\S]*\\\\end\\{(?:itemize|enumerate)\\}',
                 cleaned):
        return True
    return bool(re.search('^\\s*(?:[-*+]|\\d+[.)])\\s+\\S+', cleaned, re.M))


def _valid_tuple_target(text: str) -> bool:
    value = _literal_value(text)
    if isinstance(value, tuple):
        return True
    cleaned = _unwrap_answer_container(text)
    return bool(re.fullmatch('\\(\\s*[^(),]+(?:\\s*,\\s*[^(),]+)+\\s*\\)', cleaned, re.S))


def _extract_required_dict_keys(edit_question: str) -> List[str]:
    snippets = re.findall('\\{[^{}]{1,1000}\\}', edit_question or '', re.S)
    for snippet in reversed(snippets):
        value = _literal_value(snippet)
        if isinstance(value, dict) and value:
            return [str(k) for k in value.keys()]
    return []


def _valid_dictionary_target(text: str, edit_question: str) -> Tuple[bool, List[str]]:
    value = _literal_value(text)
    if not isinstance(value, dict):
        return (False, [])
    required_keys = _extract_required_dict_keys(edit_question)
    return (all((k in value for k in required_keys)), required_keys)


_VOID_HTML_TAGS = frozenset(
    {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr'})


class _HTMLStructureParser(HTMLParser):

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: List[str] = []
        self.roots: List[str] = []
        self.invalid = False
        self.text_outside = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if not self.stack:
            self.roots.append(tag)
        if tag not in _VOID_HTML_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        if not self.stack:
            self.roots.append(tag.lower())

    def handle_endtag(self, tag):
        tag = tag.lower()
        if not self.stack or self.stack[-1] != tag:
            self.invalid = True
            return
        self.stack.pop()

    def handle_data(self, data):
        if data.strip() and (not self.stack):
            self.text_outside = True


def _expected_root_tag(edit_question: str, language: str) -> Optional[str]:
    keyword_pos = (edit_question or '').lower().rfind(language.lower())
    scope = (edit_question or '')[keyword_pos:] if keyword_pos >= 0 else edit_question or ''
    tags = re.findall('<([A-Za-z_][\\w:.-]*)\\b[^>]*>', scope)
    return tags[0].lower() if tags else None


def _valid_html_target(text: str, edit_question: str) -> Tuple[bool, Optional[str]]:
    cleaned = _unwrap_answer_container(text)
    parser = _HTMLStructureParser()
    try:
        parser.feed(cleaned)
        parser.close()
    except Exception:
        return (False, None)
    expected = _expected_root_tag(edit_question, 'html')
    ok = bool(parser.roots) and (not parser.stack) and (not parser.invalid) and (not parser.text_outside)
    if expected:
        ok = ok and parser.roots[0] == expected
    return (ok, expected)


def _valid_xml_target(text: str, edit_question: str) -> Tuple[bool, Optional[str]]:
    cleaned = _unwrap_answer_container(text)
    expected = _expected_root_tag(edit_question, 'xml')
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        return (False, expected)
    actual = root.tag.split('}')[-1].lower()
    return (not expected or actual == expected, expected)


def check_json_format(response: str, item: Dict = None, **kwargs) -> Dict:
    if not response:
        return {'score': 0.0, 'detail': 'Empty response'}
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    cleaned = _strip_outer_code_fence(target)
    try:
        json.loads(cleaned)
        return {'score': 1.0, 'detail': f'scope={scope}, src={src}, JSON'}
    except Exception as e:
        return {
            'score': 0.0,
            'detail': f'scope={scope}, src={src}, JSON Parsing failed: {str(e)} | cleaned={cleaned[:100]}'
        }


def check_list_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    ok = _valid_list_target(target)
    return {
        'score': 1.0 if ok else 0.0,
        'detail': f"scope={scope}, src={src}, {('Detected a valid' if ok else 'valid')}"
    }


def check_tuple_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    ok = _valid_tuple_target(target)
    return {
        'score': 1.0 if ok else 0.0,
        'detail': f"scope={scope}, src={src}, {('Detected a valid' if ok else 'valid')}"
    }


def check_dictionary_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    ok, keys = _valid_dictionary_target(target, q)
    key_note = f', required_keys={keys}' if keys else ''
    return {
        'score': 1.0 if ok else 0.0,
        'detail': f"scope={scope}, src={src}{key_note}, {('The format is valid' if ok else 'valid')}"
    }


def check_markdown_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    patterns = ['^#{1,6}\\s+', '^\\s*[-*+]\\s+', '^\\s*\\d+\\.\\s+', '```', '\\[.+?\\]\\(.+?\\)']
    for p in patterns:
        if re.search(p, target, re.MULTILINE):
            return {'score': 1.0, 'detail': f'scope={scope}, src={src}, Detected Markdown :{p}'}
    return {'score': 0.0, 'detail': f'scope={scope}, src={src}, Not detected Markdown'}


def check_html_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    ok, expected = _valid_html_target(target, q)
    return {
        'score': 1.0 if ok else 0.0,
        'detail':
        f"scope={scope}, src={src}, expected_root={expected or 'any'}, {('HTML valid' if ok else 'HTML invalid')}"
    }


def check_xml_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    ok, expected = _valid_xml_target(target, q)
    return {
        'score': 1.0 if ok else 0.0,
        'detail':
        f"scope={scope}, src={src}, expected_root={expected or 'any'}, {('XML valid' if ok else 'XML invalid')}"
    }


def check_csv_format(response: str, item: Dict = None, **kwargs) -> Dict:
    q = (item or {}).get('edit_question', '')
    target, scope, src = extract_format_target(response, q)
    try:
        rows = [row for row in csv.reader(io.StringIO(target)) if any((cell.strip() for cell in row))]
    except csv.Error as e:
        return {'score': 0.0, 'detail': f'scope={scope}, src={src}, CSV Parsing failed: {e}'}
    if not rows:
        return {'score': 0.0, 'detail': f'scope={scope}, src={src}, CSV The content is empty'}
    counts = [len(row) for row in rows]
    if len(set(counts)) == 1 and counts[0] > 1:
        return {'score': 1.0, 'detail': f'scope={scope}, src={src}, CSV passed, total{len(rows)},{counts[0]}'}
    line_match = re.search('\\b(\\d+|one|two|three|four|five)\\s+lines?\\b', q, re.I)
    if line_match and len(set(counts)) == 1 and (counts[0] == 1):
        raw_n = line_match.group(1).lower()
        required_lines = int(raw_n) if raw_n.isdigit() else EN_NUM.get(raw_n)
        if required_lines == len(rows):
            return {'score': 1.0, 'detail': f'scope={scope}, src={src}, CSV passed, total{len(rows)}'}
    return {'score': 0.0, 'detail': f'scope={scope}, src={src}, CSV inconsistent Not detected'}


def _normalize_for_match(s: str) -> str:
    return re.sub('\\s+', ' ', re.sub('[^a-z0-9\\u4e00-\\u9fff]+', ' ', s.lower())).strip()


def _extract_gt_blockquote_options(edit_question: str) -> List[str]:
    q = edit_question
    low = q.lower()
    idx = low.find('choice list')
    if idx != -1:
        q = q[idx:]
    out: List[str] = []
    for line in q.splitlines():
        s = line.strip()
        if not s.startswith('>'):
            continue
        rest = s[1:].strip()
        if rest:
            out.append(rest)
    return out


def _extract_choose_options_from_question(edit_question: str) -> Tuple[List[str], str]:
    gt_opts = _extract_gt_blockquote_options(edit_question)
    if gt_opts:
        return (gt_opts, 'CHOICE LIST (> )')
    q_lower = edit_question.lower()
    keywords = ['choice list', 'choices', 'choice', 'options', 'option']
    start_idx = None
    for kw in keywords:
        i = q_lower.find(kw)
        if i != -1:
            start_idx = i + len(kw)
            break
    tail_original = edit_question[start_idx:] if start_idx is not None else edit_question
    parts_original = [p.strip() for p in re.split('[。!?]|(?<!\\d)\\.(?!\\d)+', tail_original) if p.strip()]
    scope_original = ' '.join(parts_original[:2]) if parts_original else tail_original
    contents: List[str] = []
    option_pat = re.compile('(?:^|[\\n\\r;；])\\s*(?:\\(?\\s*)([a-z]|\\d+)\\s*\\)?\\s*[\\.\\)]\\s*([^;\\n\\r]+)', re.I)
    for m in option_pat.finditer(scope_original):
        val = m.group(2).strip()
        if val and len(val) >= 1:
            contents.append(val)
    if contents:
        return (contents, 'question choice/options')
    for m in option_pat.finditer(edit_question):
        val = m.group(2).strip()
        if val:
            contents.append(val)
    if contents:
        return (contents, 'question option')
    loose_gt: List[str] = []
    for line in edit_question.splitlines():
        s = line.strip()
        if s.startswith('>'):
            rest = s[1:].strip()
            if rest:
                loose_gt.append(rest)
    if loose_gt:
        return (loose_gt, 'question >')
    return ([], '')


def check_choose_from(response: str, required_parameters: str = None, item: Dict = None, **kwargs) -> Dict:
    edit_question = item.get('edit_question', '') if item else ''
    if not response or not edit_question:
        return {'score': 0.0, 'detail': 'question'}
    res_norm = _normalize_for_match(response)
    option_texts, scope_note = _extract_choose_options_from_question(edit_question)
    if option_texts:
        for opt in option_texts:
            opt_stripped = opt.strip()
            if opt_stripped and opt_stripped in response:
                return {
                    'score': 1.0,
                    'detail': f"option ({scope_note}): {opt[:120]}{('...' if len(opt) > 120 else '')}"
                }
            o_norm = _normalize_for_match(opt)
            if len(o_norm) < 2:
                continue
            if o_norm in res_norm:
                return {
                    'score': 1.0,
                    'detail': f"option ({scope_note}): {opt[:120]}{('...' if len(opt) > 120 else '')}"
                }
        return {'score': 0.0, 'detail': f'response option ({scope_note}); option ={len(option_texts)}'}
    q_lower = edit_question.lower()
    letters = set(re.findall('\\b([a-z])[\\.\\)]', q_lower))
    nums = set(re.findall('\\b(\\d+)[\\.\\)]', q_lower))
    res_lower = response.lower().strip()
    for ch in sorted(letters):
        if re.search(f'(?:^|\\s){re.escape(ch)}(?:\\s|[\\.\\)]|$)', res_lower):
            return {'score': 1.0, 'detail': f'option :{ch}(question option , )'}
    for n in sorted(nums, key=len, reverse=True):
        if re.search(f'(?:^|\\s){re.escape(n)}(?:\\s|[\\.\\)]|$)', res_lower):
            return {'score': 1.0, 'detail': f'option :{n}(question option , )'}
    return {'score': 0.0, 'detail': 'question option, response'}


def check_judge(response: str, item: Dict = None, **kwargs) -> Dict:
    res = response.strip().lower()
    match = re.search('\\b(yes|no|true|false)\\b', res)
    if match:
        return {'score': 1.0, 'detail': f'Detected :{match.group()}'}
    return {'score': 0.0, 'detail': 'Not detected (Yes/No/True/False)'}


def check_number_response(response: str, required_parameters: str, item: Dict = None, **kwargs) -> Dict:
    if not isinstance(response, str) or not response.strip():
        return {'score': 0.0, 'detail': 'The response is empty'}
    m = re.search('\\d+', str(required_parameters))
    if not m:
        return {'score': 0.0, 'detail': f'Unable to parse required_parameters: {required_parameters}'}
    n = int(m.group())
    if n <= 2:
        return {'score': 0.0, 'detail': f'N > 2, actual:{n}'}
    llm_client = kwargs.get('llm_client')
    judge_model = kwargs.get('judge_model')
    if llm_client is None:
        return {'score': 0.0, 'detail': 'llm_client, cannot LLM-judge'}
    edit_question = item.get('edit_question', '') if isinstance(item, dict) else ''
    prompt = f'You are a strict evaluator.\n\nDecide whether the model output contains EXACTLY {n} distinct response types/categories.\n\nGuidelines:\n- A "response type/category" means a clearly separable class of outputs.\n- Count distinct categories, not wording variations.\n\nReturn ONLY in JSON format:\n{{"answer": "YES or NO", "reason": "explain how many categories you found and why"}}\n\nQuestion:\n{edit_question[:1200]}\n\nModel output:\n{response[:3000]}\n'  # noqa: E501
    try:
        if hasattr(llm_client, 'chat') and hasattr(getattr(llm_client.chat, 'completions', None), 'create'):
            resp = llm_client.chat.completions.create(model=judge_model,
                                                      messages=[{
                                                          'role': 'system',
                                                          'content': 'You are a strict evaluator.'
                                                      }, {
                                                          'role': 'user',
                                                          'content': prompt
                                                      }],
                                                      temperature=0)
            text = (resp.choices[0].message.content or '').strip()
        elif hasattr(llm_client, 'complete'):
            text = str(llm_client.complete(prompt)).strip()
        else:
            text = str(llm_client(prompt)).strip()
        if '{' in text:
            start = text.index('{')
            end = text.rindex('}') + 1
            result = json.loads(text[start:end])
            answer = str(result.get('answer', '')).lower()
            reason = result.get('reason', '')
            if 'yes' in answer:
                return {'score': 1.0, 'detail': f'LLM-judge: YES (N={n}) | {reason}'}
            elif 'no' in answer:
                return {'score': 0.0, 'detail': f'LLM-judge: NO (N={n}) | {reason}'}
        return {'score': 0.0, 'detail': f'LLM-judge Unable to parse:{text[:200]}'}
    except Exception as e:
        return {'score': 0.0, 'detail': f'LLM-judge error: {e}'}


def _split_sentences(text: str) -> List[str]:
    parts = re.split('(?<=[。!?]|(?<!\\d)\\.(?!\\d))\\s+|\\n+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def check_response_structure(response: str, required_parameters: str, item: Dict = None, **kwargs) -> Dict:
    if not isinstance(response, str) or not response.strip():
        return {'score': 0.0, 'detail': 'The model response is empty'}
    pos_match = re.search('\\b(beginning|middle|end)\\b', required_parameters.lower())
    if not pos_match:
        return {'score': 0.0, 'detail': 'required (beginning/middle/end)'}
    target_pos = pos_match.group()
    sentences = _split_sentences(response)
    if not sentences:
        return {'score': 0.0, 'detail': 'cannot'}
    answer_phrase_re = re.compile(
        '(final\\s*numerical\\s*answer|final\\s*answer|final\\s+analysis|final_answer|final_numerical_answer|(?:the\\s+)?answer\\s+is|final\\s+conclusion|result\\s+is)',  # noqa: E501
        re.I)
    idx = None
    for i, sent in enumerate(sentences):
        if answer_phrase_re.search(sent):
            idx = i
            break
    if idx is None:
        return {
            'score':
            0.0,
            'detail':
            'Not foundanswer ( final answer / final numerical answer / final analysis / '
            'The answer is / answer is / final conclusion / result is / )'
        }
    n = len(sentences)
    ratio = idx / max(n - 1, 1) if n > 1 else 0.5
    if target_pos == 'beginning' and ratio < 0.2 or (target_pos == 'middle'
                                                     and 0.2 <= ratio <= 0.8) or (target_pos == 'end' and ratio > 0.8):
        return {'score': 1.0, 'detail': f'answer{idx + 1}/{n},{target_pos}( ≈{ratio:.2f})'}
    return {'score': 0.0, 'detail': f'answer{idx + 1}/{n}, .required:{target_pos}, ≈{ratio:.2f}'}
