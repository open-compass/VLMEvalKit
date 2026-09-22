import json
import re
from typing import Any, Dict, List, Optional

_PLACEHOLDER_SUBSTRINGS = ('specific geographical address format', 'specific format of geographical')


def evaluate_geography_address(response: str,
                               item: Dict[str, Any],
                               instruction_name: str,
                               required_parameters: str = '',
                               **kwargs) -> Dict[str, Any]:
    if not response or not str(response).strip():
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    rp = (required_parameters or '').strip()
    eq = item.get('edit_question', '') or '' if item else ''
    if _is_placeholder_required_parameters(rp):
        rp = ''
    schema_keys = _extract_address_schema_keys(rp, eq)
    if schema_keys:
        obj = _try_parse_json_object_from_response(response)
        if obj is not None:
            missing = [k for k in schema_keys if not _dict_has_schema_key(obj, k)]
            if not missing:
                return {'passed': True, 'score': 1.0, 'detail': f'JSON field:{schema_keys}'}
            return {'passed': False, 'score': 0.0, 'detail': f'JSON field{missing}(required :{schema_keys}）'}
        ok, why = _heuristic_address_match(response)
        if ok:
            return {'passed': True, 'score': 1.0, 'detail': f'{why}; JSON field{schema_keys}'}
        return {'passed': False, 'score': 0.0, 'detail': f'Not found JSON passed ; required field:{schema_keys}'}
    ok, why = _heuristic_address_match(response)
    if ok:
        return {'passed': True, 'score': 1.0, 'detail': why}
    return {'passed': False, 'score': 0.0, 'detail': 'The output does not satisfy required'}


def _heuristic_address_match(response: str) -> tuple:
    level_indicators = [
        '\\d+°\\d+[\'\\"]?\\s*[NS]\\s*\\d+°\\d+[\'\\"]?\\s*[EW]', '(?:Province|City|District|County|Street|Road|Ave)',
        '[A-Za-z\\s]+,\\s*[A-Za-z\\s]+,\\s*[A-Za-z\\s]+', '\\d+\\s*(?:km|miles?)\\s*(?:north|south|east|west)'
    ]
    for p in level_indicators:
        if re.search(p, response, re.I):
            return (True, 'The output satisfies ( )')
    parts = re.split('[,，/\\-]\\s*', response)
    if len([p for p in parts if len(p.strip()) > 2]) >= 2:
        return (True, 'The output contains ( )')
    return (False, '')


def _is_placeholder_required_parameters(rp: str) -> bool:
    if not rp:
        return True
    low = rp.lower()
    return any((s in low for s in _PLACEHOLDER_SUBSTRINGS))


def _extract_address_schema_keys(rp: str, eq: str) -> List[str]:
    for blob in (rp, eq):
        keys = _extract_keys_from_text(blob or '')
        if keys:
            return keys
    return []


def _extract_keys_from_text(text: str) -> List[str]:
    if not text.strip():
        return []
    i = text.find('{')
    j = text.rfind('}')
    if i != -1 and j > i:
        snippet = text[i:j + 1]
        try:
            d = json.loads(snippet)
            if isinstance(d, dict) and d:
                return list(d.keys())
        except json.JSONDecodeError:
            pass
        keys = re.findall('"([^"]+)"\\s*:', snippet)
        if not keys:
            keys = re.findall("'([^']+)'\\s*:", snippet)
        if keys:
            return list(dict.fromkeys(keys))
    low = text.lower()
    for prefix in ('geographical hierarchy format:', 'geographical address format:', 'address format:',
                   'hierarchy format:', 'hierarchy:', 'format:'):
        if prefix in low:
            idx = low.find(prefix)
            rest = text[idx + len(prefix):].strip()
            rest = rest.split('\n')[0]
            parts = re.split('[,，;]', rest)
            out: List[str] = []
            for p in parts:
                p = p.strip().strip('"\'')
                if p and len(p) < 100:
                    out.append(p)
            if out:
                return out
    return []


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


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = s.replace('-', '_')
    s = re.sub('[\\s/]+', '_', s)
    return s


def _dict_has_schema_key(obj: Dict[str, Any], expected: str) -> bool:
    ne = _norm_key(expected)
    for k in obj.keys():
        if _norm_key(k) == ne:
            return True
    return False
