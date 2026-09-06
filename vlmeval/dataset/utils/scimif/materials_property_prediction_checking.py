import re
from typing import Any, Dict


def evaluate_property_prediction(response: str,
                                 item: Dict[str, Any],
                                 instruction_name: str,
                                 required_parameters: str = '',
                                 **kwargs) -> Dict[str, Any]:
    if not response:
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    params_lower = (required_parameters or '').lower()
    is_discrete = 'discrete' in params_lower or 'classification' in params_lower or 'category' in params_lower
    is_continuous = 'continuous' in params_lower or 'numeric value' in params_lower
    if is_discrete:
        labels = re.findall('\\b(Yes|No|High|Low|Medium|Class \\d+|Category \\d+)\\b', response, re.I)
        if labels:
            return {'passed': True, 'score': 1.0, 'detail': 'The output contains a discrete label'}
        if re.search('[A-Za-z]{3,}', response) and (not re.search('\\d+\\.\\d+', response)):
            return {'passed': True, 'score': 1.0, 'detail': 'The output is label-like'}
        return {'passed': False, 'score': 0.0, 'detail': 'A discrete label is required'}
    if is_continuous:
        numbers = re.findall('\\b\\d+\\.?\\d*\\b', response)
        if numbers:
            return {'passed': True, 'score': 1.0, 'detail': 'The output contains a numeric value'}
        return {'passed': False, 'score': 0.0, 'detail': 'A numeric value is required'}
    return {'passed': True, 'score': 1.0, 'detail': 'No property type was specified'}
