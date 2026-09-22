import re
from typing import Any, Dict


def evaluate_sequence_length(response: str,
                             item: Dict[str, Any],
                             instruction_name: str,
                             required_parameters: str = '',
                             **kwargs) -> Dict[str, Any]:
    if not response:
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    params_lower = (required_parameters or '').lower()
    answer = item.get('answer', '')
    numbers = re.findall('\\b(\\d+)\\b', response)
    if numbers:
        answer_nums = re.findall('\\b(\\d+)\\b', str(answer))
        if answer_nums and numbers:
            if any((n in answer_nums for n in numbers)):
                return {'passed': True, 'score': 1.0, 'detail': 'numeric value answer is consistent'}
        return {'passed': True, 'score': 1.0, 'detail': 'The output contains numeric value'}
    length_indicators = ['longest', 'shortest', 'length', 'bp', 'nt', 'amino acid', 'ORF']
    if any((ind in response.lower() or ind in params_lower for ind in length_indicators)):
        return {'passed': True, 'score': 1.0, 'detail': 'The output discusses sequence length'}
    return {'passed': False, 'score': 0.0, 'detail': 'The output does not satisfy the sequence-length constraint'}
