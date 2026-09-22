import re
from typing import Any, Dict

try:
    import rdkit
    HAS_RDKIT = rdkit is not None
except ImportError:
    HAS_RDKIT = False
try:
    import selfies
    HAS_SELFIES = True
except ImportError:
    HAS_SELFIES = False
from .extraction_utils import extract_molecule
from .rdkit_utils import mol_from_smiles_lenient


def evaluate_molecular_format(response: str,
                              item: Dict[str, Any],
                              instruction_name: str,
                              required_parameters: str = '',
                              llm_client=None,
                              **kwargs) -> Dict[str, Any]:
    if not response:
        return {'passed': False, 'score': 0.0, 'detail': 'response is empty'}
    fmt = _detect_required_format(required_parameters, item)
    if not fmt:
        fmt = 'SMILES'
    if fmt.upper() == 'SELFIES':
        return _validate_selfies(response, llm_client=llm_client, item=item)
    if fmt.upper() == 'SMILES':
        return _validate_smiles(response, llm_client=llm_client, item=item)
    if fmt.upper() == 'IUPAC':
        return _validate_iupac(response, llm_client=llm_client, item=item)
    return {'passed': False, 'score': 0.0, 'detail': f':{fmt}'}


def _detect_required_format(required_parameters: str, item: Dict) -> str:
    text = (required_parameters or '') + ' ' + (item.get('edit_question', '') or '')
    text_lower = text.lower()
    if 'selfies' in text_lower:
        return 'SELFIES'
    if 'smiles' in text_lower:
        return 'SMILES'
    if 'iupac' in text_lower:
        return 'IUPAC'
    return ''


def _validate_selfies(response: str, llm_client=None, item=None) -> Dict[str, Any]:
    mol_str, _, extract_src = extract_molecule(response, llm_client=llm_client, item=item)
    if not mol_str:
        return {'passed': False, 'score': 0.0, 'detail': 'Not found SELFIES string'}
    if not _is_valid_selfies_candidate(mol_str):
        return {'passed': False, 'score': 0.0, 'detail': f'valid SELFIES ; extracted({extract_src}): {mol_str}'}
    if HAS_SELFIES:
        try:
            smiles = selfies.decoder(mol_str)
            if smiles and HAS_RDKIT:
                mol = mol_from_smiles_lenient(smiles)
                if mol is not None:
                    return {
                        'passed': True,
                        'score': 1.0,
                        'detail': f'SELFIES The format is valid; extracted({extract_src}): {mol_str}'
                    }
        except Exception:
            return {
                'passed': False,
                'score': 0.0,
                'detail': f'SELFIES Parsing failed; extracted({extract_src}): {mol_str}'
            }
    if re.match('^(\\[[^\\]]+\\])+$', mol_str):
        return {'passed': True, 'score': 1.0, 'detail': f'SELFIES ; extracted({extract_src}): {mol_str}'}
    return {
        'passed': False,
        'score': 0.0,
        'detail': f'SELFIES The format is invalid; extracted({extract_src}): {mol_str}'
    }


def _is_valid_selfies_candidate(s: str) -> bool:
    if not s or not s.strip().startswith('['):
        return False
    low = s.strip().lower()
    if low in {'molecule', 'smiles', 'selfies', 'chemical', 'answer'}:
        return False
    return bool(re.match('^(\\[[^\\]]+\\])+$', s.strip()))


def _validate_smiles(response: str, llm_client=None, item=None) -> Dict[str, Any]:
    mol_str, _, extract_src = extract_molecule(response, llm_client=llm_client, item=item)
    if not mol_str:
        return {'passed': False, 'score': 0.0, 'detail': 'Not found SMILES string'}
    if HAS_RDKIT:
        mol = mol_from_smiles_lenient(mol_str)
        if mol is not None:
            return {
                'passed': True,
                'score': 1.0,
                'detail': f'SMILES The format is valid; extracted({extract_src}): {mol_str}'
            }
        return {
            'passed': False,
            'score': 0.0,
            'detail': f'RDKit Unable to parse SMILES; extracted({extract_src}): {mol_str}'
        }
    if re.fullmatch('[A-Za-z0-9@+\\-\\[\\]\\(\\)=#$\\\\/%.:*]+', mol_str) and len(mol_str) >= 3:
        return {'passed': True, 'score': 1.0, 'detail': f'SMILES ; extracted({extract_src}): {mol_str}'}
    return {
        'passed': False,
        'score': 0.0,
        'detail': f'SMILES The format is invalid; extracted({extract_src}): {mol_str}'
    }


def _is_valid_smiles_candidate(s: str) -> bool:
    if not s or len(s) < 3:
        return False
    low = s.strip().lower()
    blocklist = {'molecule', 'smiles', 'selfies', 'chemical', 'answer', 'structure', 'compound', 'format'}
    if low in blocklist:
        return False
    if not re.fullmatch('[A-Za-z0-9@+\\-\\[\\]\\(\\)=#$\\\\/%.:*]+', s):
        return False
    if HAS_RDKIT:
        return mol_from_smiles_lenient(s) is not None
    return bool(re.search('(?:Cl|Br|Si|Se|Te|[BCNOFPSIbcnosp])', s))


def _validate_iupac(response: str, llm_client=None, item=None) -> Dict[str, Any]:
    extracted, extract_src = _extract_iupac(response, llm_client=llm_client, item=item)
    if not extracted:
        return {'passed': False, 'score': 0.0, 'detail': 'Not found IUPAC'}
    if re.search('\\b(ethane|methane|propane|butane|pentane|hexane|benzene|ethanol)\\b', extracted, re.I):
        return {'passed': True, 'score': 1.0, 'detail': f'Detected IUPAC ; extracted({extract_src}): {extracted}'}
    if re.search('[a-z]+\\-\\d\\-[a-z]+|[a-z]+\\d+[a-z]*', extracted):
        return {'passed': True, 'score': 1.0, 'detail': f'IUPAC ; extracted({extract_src}): {extracted}'}
    return {
        'passed': False,
        'score': 0.0,
        'detail': f'Did not detect a valid IUPAC ; extracted({extract_src}): {extracted}'
    }


def _extract_iupac(response: str, llm_client=None, item=None) -> tuple:
    m = re.search('\\b([a-z]+\\-\\d\\-[a-z]+|[a-z]+\\d+[a-z]*|[A-Za-z]+(?:ane|ene|ol|one|al|oic)\\b)', response, re.I)
    if m:
        return (m.group(1), 'regex')
    if llm_client:
        from .extraction_utils import _extract_via_llm
        extracted = _extract_via_llm(
            response,
            llm_client,
            prompt_addendum='Extract the IUPAC chemical name from the response. Return only the name, nothing else.')
        if extracted:
            return (extracted, 'llm')
    return ('', '')
