import re
from typing import Any, Dict


def evaluate_entity_relationship(response: str,
                                 item: Dict[str, Any],
                                 instruction_name: str,
                                 required_parameters: str = '',
                                 **kwargs) -> Dict[str, Any]:
    if not response:
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    tuple_pattern = '\\([^)]+\\)'
    tuples_found = re.findall(tuple_pattern, response)
    valid_tuples = []
    for t in tuples_found:
        inner = t[1:-1]
        parts = [p.strip().strip('"\'') for p in re.split('[,，]', inner)]
        if 2 <= len(parts) <= 5 and all((len(p) > 0 for p in parts)):
            valid_tuples.append(t)
    if len(valid_tuples) >= 1:
        return {'passed': True, 'score': 1.0, 'detail': f'The output contains {len(valid_tuples)}'}
    if re.search('\\([^,]+,\\s*[^,]+,\\s*[^)]+\\)', response):
        return {'passed': True, 'score': 1.0, 'detail': 'The output satisfies (subject, relation, object)'}
    return {'passed': False, 'score': 0.0, 'detail': 'The output does not contain a valid'}
