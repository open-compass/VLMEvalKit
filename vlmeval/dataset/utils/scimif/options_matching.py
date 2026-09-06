import json
import re
from typing import Any, Dict, List, Optional, Tuple

_JSON_SCHEMA_KEYS = frozenset({'lighting_condition', 'platform', 'view_direction', 'weather'})


def evaluate_options_constraint(response: str,
                                item: Dict[str, Any],
                                instruction_name: str,
                                required_parameters: str = '',
                                **kwargs) -> Dict[str, Any]:
    if not response or not str(response).strip():
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    eq = item.get('edit_question', '') or '' if item else ''
    labels, label_src = _resolve_option_labels(required_parameters or '', eq)
    if not labels:
        return {
            'passed': False,
            'score': 0.0,
            'detail':
            f'cannot required_parameters edit_question validlabel ; required_parameters={required_parameters!r}'
        }
    response_clean = response.strip()
    invalid_found: List[str] = []
    passed = _check_options_in_response(response_clean, labels, invalid_found)
    src_note = f' (labelsource: {label_src})'
    if passed:
        return {'passed': True, 'score': 1.0, 'detail': f'The output satisfiesoption :{labels}{src_note}'}
    return {
        'passed':
        False,
        'score':
        0.0,
        'detail': (f'The output contains option:{invalid_found}'
                   if invalid_found else f'The output does not satisfyoption: {labels}') + src_note
    }


def _parse_labels(required_parameters: str) -> List[str]:
    text = required_parameters.strip()
    for sep in [',', ';', '|']:
        if sep in text:
            parts = [p.strip().strip('"\'') for p in text.split(sep)]
            return [p for p in parts if p]
    return [text] if text else []


def _resolve_option_labels(required_parameters: str, edit_question: str) -> Tuple[List[str], str]:
    rp = (required_parameters or '').strip()
    if rp:
        labels = _parse_labels(rp)
        if labels:
            return (labels, 'required_parameters')
    labels = _parse_labels_from_edit_question(edit_question or '')
    if labels:
        return (labels, 'edit_question')
    return ([], '')


def _parse_labels_from_edit_question(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    seen = set()
    out: List[str] = []

    def add(s: str) -> None:
        s = (s or '').strip()
        if not s or len(s) > 200:
            return
        low = s.lower()
        if low in seen:
            return
        seen.add(low)
        out.append(s)

    for m in re.finditer('\\[([^\\]]+)\\]', text, re.DOTALL):
        inner = m.group(1).strip()
        if not inner:
            continue
        try:
            parsed = json.loads('[' + inner + ']')
            if isinstance(parsed, list):
                for x in parsed:
                    if isinstance(x, str):
                        add(x)
                continue
        except json.JSONDecodeError:
            pass
        for part in re.split(',', inner):
            part = part.strip().strip('"\'')
            if part:
                add(part)
    for line in text.splitlines():
        line = line.strip()
        if '|' not in line or '"' not in line:
            continue
        for q in re.findall('"([^"]+)"', line):
            if q in _JSON_SCHEMA_KEYS:
                continue
            if len(q) > 120:
                continue
            add(q)
    low = text.lower()
    if 'choice list' in low:
        idx = low.find('choice list')
        tail = text[idx:]
        for ln in tail.splitlines():
            s = ln.strip()
            if not s.startswith('>'):
                continue
            rest = s[1:].strip()
            if rest and len(rest) < 500:
                add(rest)
    return out


def _try_parse_json_object_from_response(response: str) -> Optional[Dict[str, Any]]:
    t = response.strip()
    t = re.sub('^```[a-zA-Z]*\\s*', '', t)
    t = re.sub('\\s*```\\s*$', '', t).strip()
    i = t.find('{')
    j = t.rfind('}')
    if i == -1 or j <= i:
        return None
    try:
        obj = json.loads(t[i:j + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _check_options_in_response(response: str, labels: List[str], invalid_found: List[str]) -> bool:
    labels_lower = {s.lower() for s in labels}
    obj = _try_parse_json_object_from_response(response)
    if obj is not None:
        str_vals = [v for v in obj.values() if isinstance(v, str)]
        if str_vals:
            ok = True
            for k, v in obj.items():
                if not isinstance(v, str):
                    continue
                if v.lower() in labels_lower or v in labels:
                    continue
                invalid_found.append(f'{k}={v!r}')
                ok = False
            return ok
    parts = re.split('[,;]\\s*', response)
    answer_like = [p.strip().strip('"\'') for p in parts if p.strip() and len(p.strip()) < 50]
    if len(answer_like) >= 3 and (not ('{' in response and '"' in response)):
        for p in answer_like:
            if p.lower() not in labels_lower and (not p.isdigit()):
                invalid_found.append(p)
        return len(invalid_found) == 0
    words = re.findall('"([^"]+)"', response) or re.findall('\\b(Yes|No|Maybe|Unknown)\\b', response, re.I)
    if not words:
        words = re.findall('\\b([A-Za-z][a-z]{1,25})\\b', response)
    for w in words:
        if w in _JSON_SCHEMA_KEYS:
            continue
        if w.lower() in labels_lower or w in labels:
            continue
        if w.isdigit() or len(w) > 40:
            continue
        if w.lower() in {'the', 'and', 'or', 'is', 'are', 'to', 'of', 'in', 'a', 'an', 'json'}:
            continue
        invalid_found.append(w)
    has_valid = any((re.search(re.escape(label), response, re.I) for label in labels))
    return has_valid and len(invalid_found) == 0
