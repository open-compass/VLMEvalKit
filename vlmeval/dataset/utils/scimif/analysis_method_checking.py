import re
from typing import Any, Dict

from .extraction_utils import extract_method


def evaluate_method_constraint(response: str,
                               item: Dict[str, Any],
                               instruction_name: str,
                               required_parameters: str = '',
                               llm_client=None,
                               **kwargs) -> Dict[str, Any]:
    if not required_parameters or not response:
        return {'passed': False, 'score': 0.0, 'detail': 'Missing required_parameters response is empty'}
    method_text = required_parameters.strip()
    for phrase in [
            'specific chemical fomulas or laws', 'specific physical fomulas or laws',
            'specific geographical fomulas or laws', 'specific biological fomulas or laws',
            'specific materials science fomulas or laws'
    ]:
        method_text = re.sub(re.escape(phrase) + '\\s*[:：]?\\s*', '', method_text, flags=re.I)
    method_text = method_text.strip()
    if not method_text:
        return {'passed': False, 'score': 0.0, 'detail': 'Unable to parse required_parameters formula/law'}
    extracted, extract_src = extract_method(response, method_text, llm_client=llm_client, item=item)
    if not extracted:
        return {'passed': False, 'score': 0.0, 'detail': f"formula/law: '{method_text}'"}
    patterns = _build_method_patterns(method_text)
    matched = any((re.search(p, extracted, re.IGNORECASE | re.DOTALL) for p in patterns))
    if not matched:
        matched = any((re.search(p, response, re.IGNORECASE | re.DOTALL) for p in patterns))
    if matched:
        return {'passed': True, 'score': 1.0, 'detail': f'method; extracted({extract_src}): {extracted}'}
    return {'passed': False, 'score': 0.0, 'detail': f'method required ; extracted({extract_src}): {extracted}'}


def _build_method_patterns(method_text: str) -> list:
    patterns = []
    patterns.append(re.escape(method_text))
    if '=' in method_text:
        parts = method_text.split('=', 1)
        if len(parts) == 2:
            patterns.append(re.escape(parts[0].strip()) + '\\s*=\\s*' + re.escape(parts[1].strip()))
    formula_aliases = {
        'F=ma': ['F\\s*=\\s*ma', 'Newton'],
        'PV=nRT': ['PV\\s*=\\s*nRT', 'ideal gas'],
        'E=mc²': ['E\\s*=\\s*mc', 'mass-energy']
    }
    for k, aliases in formula_aliases.items():
        if k in method_text or any((a.lower() in method_text.lower() for a in aliases)):
            patterns.extend(aliases)
    return patterns
