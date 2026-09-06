import re
from typing import Any, Dict

try:
    from rdkit import Chem
    from rdkit.Chem import Lipinski
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
from .extraction_utils import extract_molecule
from .rdkit_utils import mol_from_smiles_lenient


def evaluate_bond_count(response: str,
                        item: Dict[str, Any],
                        instruction_name: str,
                        required_parameters: str = '',
                        llm_client=None,
                        **kwargs) -> Dict[str, Any]:
    if not required_parameters or not response:
        return {'passed': False, 'score': 0.0, 'detail': 'Missing required_parameters response is empty'}
    edit_question = item.get('edit_question', '') or '' if item else ''
    target = _parse_bond_requirements(f"{required_parameters or ''}\n{edit_question}")
    if not target:
        return {
            'passed': False,
            'score': 0.0,
            'detail': f'Evaluator unable to parsechemical bond :{required_parameters}',
            'skipped': True
        }
    if not HAS_RDKIT:
        return {
            'passed': False,
            'score': 0.0,
            'detail': 'RDKit not installed, cannot reliably molecule chemical bond count',
            'skipped': True
        }
    mol_str, fmt, extract_src = extract_molecule(response, llm_client=llm_client, item=item)
    if not mol_str:
        return {'passed': False, 'score': 0.0, 'detail': 'molecule'}
    actual = _count_bonds_rdkit(mol_str, fmt)
    if actual is None:
        return {'passed': False, 'score': 0.0, 'detail': f'moleculeParsing failed; extracted: {mol_str}'}
    mismatches = []
    for bond_type, count in target.items():
        got = actual.get(bond_type, 0)
        if got != count:
            mismatches.append(f'{bond_type}: required {count}, actual {got}')
    passed = len(mismatches) == 0
    base_detail = 'chemical bond count required' if passed else '; '.join(mismatches)
    detail = f'{base_detail}; extracted({extract_src}): {mol_str}'
    return {'passed': passed, 'score': 1.0 if passed else 0.0, 'detail': detail}


def _parse_bond_requirements(text: str) -> Dict[str, int]:
    result: Dict[str, int] = {}
    pattern = '(?:exactly\\s+)?(\\d+)\\s+(rotatable|single|double|triple|aromatic)\\s+bonds?\\b'
    for m in re.finditer(pattern, text or '', re.I):
        result[m.group(2).lower()] = int(m.group(1))
    total_patterns = [
        '(?:exactly\\s+)?(\\d+)\\s+(?:total\\s+)?chemical\\s+bonds?\\b',
        '(?:exactly\\s+)?(\\d+)\\s+total\\s+(?:explicit\\s+heavy-atom\\s+)?bonds?\\b',
        '(?:exactly\\s+)?(\\d+)\\s+explicit\\s+heavy-atom\\s+bonds?\\b'
    ]
    for total_pattern in total_patterns:
        m = re.search(total_pattern, text or '', re.I)
        if m:
            result['total'] = int(m.group(1))
            break
    return result


def _count_bonds_rdkit(mol_str: str, fmt: str) -> Dict[str, int]:
    try:
        if (fmt or '').lower() == 'selfies':
            try:
                import selfies
                decoded = selfies.decoder(mol_str)
                if decoded:
                    mol_str = decoded
            except Exception:
                pass
        mol = mol_from_smiles_lenient(mol_str)
        if mol is None:
            return None
        single = double = triple = aromatic = 0
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            if bt == Chem.BondType.SINGLE:
                single += 1
            elif bt == Chem.BondType.DOUBLE:
                double += 1
            elif bt == Chem.BondType.TRIPLE:
                triple += 1
            elif bt == Chem.BondType.AROMATIC:
                aromatic += 1
        return {
            'total': int(mol.GetNumBonds()),
            'single': single,
            'double': double,
            'triple': triple,
            'aromatic': aromatic,
            'rotatable': int(Lipinski.NumRotatableBonds(mol))
        }
    except Exception:
        return None
