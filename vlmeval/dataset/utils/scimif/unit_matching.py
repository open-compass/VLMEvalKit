import re
from typing import Any, Dict


def evaluate_unit_consistency(response: str,
                              item: Dict[str, Any],
                              instruction_name: str,
                              required_parameters: str = '',
                              **kwargs) -> Dict[str, Any]:
    if not required_parameters or not response:
        return {'passed': False, 'score': 0.0, 'detail': 'Missing required_parameters response is empty'}
    units_text = required_parameters.strip()
    for phrase in [
            'specific chemical stoichiometric units', 'specific physical stoichiometric units',
            'specific geographical units', 'specific biological units', 'specific materials units'
    ]:
        units_text = re.sub(re.escape(phrase) + '\\s*[:：]?\\s*', '', units_text, flags=re.I)
    units_text = units_text.strip() or required_parameters.strip()
    unit_patterns = _extract_unit_patterns(units_text)
    matched = False
    for pattern in unit_patterns:
        if re.search(pattern, response, re.IGNORECASE | re.DOTALL):
            matched = True
            break
    if matched:
        return {'passed': True, 'score': 1.0, 'detail': f'The output contains unit:{units_text}'}
    return {'passed': False, 'score': 0.0, 'detail': f"The output does not contain unit '{units_text}'"}


def _extract_unit_patterns(units_text: str) -> list:
    common_units = [
        'mol/L', 'mol·L⁻¹', 'mol\\s*/\\s*L', 'mol\\s*·\\s*L', 'kg', 'g', 'mg', 'm/s', 'm/s²', 'km', 'm', 'cm', 'mm',
        'Pa', 'kPa', 'MPa', 'GPa', 'J', 'kJ', 'eV', '°C', 'K', '°F', 'cells/mL', 'cells/\\s*mL', '%', 'percent'
    ]
    patterns = []
    clean = units_text.strip()
    if clean:
        escaped = re.escape(clean)
        patterns.append(escaped)
    for u in common_units:
        if re.search(u, units_text, re.I):
            patterns.append(u)
    return patterns if patterns else [re.escape(units_text)]
