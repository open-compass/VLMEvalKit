import re
from typing import Any, Dict, List

DEFAULT_TECHNIQUES = [
    'SEM', 'TEM', 'XRD', 'AFM', 'XPS', 'FTIR', 'Raman', 'DSC', 'TGA', 'BET', 'UV-Vis', 'NMR', 'EDX', 'EDS', 'STEM',
    'HRTEM', 'SAED', 'DTA', 'TMA', 'DMA', 'ICP', 'XRF', 'SIMS', 'ESCA'
]


def evaluate_characterization_technique(response: str,
                                        item: Dict[str, Any],
                                        instruction_name: str,
                                        required_parameters: str = '',
                                        **kwargs) -> Dict[str, Any]:
    if not response:
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    techniques = _parse_technique_list(required_parameters)
    if not techniques:
        techniques = DEFAULT_TECHNIQUES
    mentioned = []
    for t in techniques:
        if re.search('\\b' + re.escape(t) + '\\b', response, re.I):
            mentioned.append(t)
    if not mentioned:
        return {'passed': False, 'score': 0.0, 'detail': 'The output does not contain a valid'}
    invalid = []
    tech_pattern = '\\b(SEM|TEM|XRD|AFM|XPS|FTIR|Raman|DSC|TGA|BET|NMR|EDX|EDS|HRTEM)\\b'
    for m in re.finditer(tech_pattern, response, re.I):
        if m.group(1).upper() not in [x.upper() for x in techniques]:
            invalid.append(m.group(1))
    if invalid and (not mentioned):
        return {'passed': False, 'score': 0.0, 'detail': f'The output contains :{invalid}'}
    return {'passed': True, 'score': 1.0, 'detail': f'The output contains a valid :{mentioned}'}


def _parse_technique_list(required_parameters: str) -> List[str]:
    if not required_parameters:
        return []
    text = required_parameters.strip()
    for sep in [',', ';', '、', '|']:
        if sep in text:
            return [t.strip() for t in text.split(sep) if t.strip()]
    return [text] if text else []
