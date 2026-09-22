import re
from typing import Any, Dict, Optional, Tuple

try:
    import rdkit
    HAS_RDKIT = rdkit is not None
except ImportError:
    HAS_RDKIT = False
from .extraction_utils import extract_molecule
from .rdkit_utils import mol_from_smiles_lenient

_SELFIES_ELEMENT_TOKENS = frozenset(
    {'H', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'Si', 'As', 'Se', 'Sb', 'Te', 'Po'})
_SMILES_ELEMENT_RE = re.compile(
    'Ac|Ag|Al|Am|Ar|As|At|Au|Ba|Be|Bh|Bi|Bk|Br|Ca|Cd|Ce|Cf|Cl|Cm|Cn|Co|Cr|Cs|Cu|Db|Ds|Dy|Er|Es|Eu|Fe|Fl|Fm|Fr|Ga|Gd|Ge|He|Hf|Hg|Ho|Hs|In|Ir|Kr|La|Li|Lr|Lu|Lv|Mc|Md|Mg|Mn|Mo|Mt|Na|Nb|Nd|Ne|Nh|Ni|No|Np|Os|Pa|Pb|Pd|Pm|Po|Pr|Pt|Pu|Ra|Rb|Re|Rf|Rg|Rh|Rn|Ru|Sb|Sc|Se|Sg|Si|Sm|Sn|Sr|Ta|Tb|Tc|Te|Th|Ti|Tl|Tm|Ts|Xe|Yb|Zn|Zr|B|C|N|O|F|P|S|I|K|V|Y|W|H|U|b|c|n|o|p|s'  # noqa: E501
)


def _count_atoms_from_selfies_brackets(mol_str: str) -> Optional[Dict[str, int]]:
    counts: Dict[str, int] = {}
    for inner in re.findall('\\[([^\\]]+)\\]', mol_str):
        inner = inner.strip()
        if inner.startswith(('Branch', 'Ring', 'Expl', 'Pad')):
            continue
        if inner.startswith('=') and len(inner) >= 2:
            sym = inner[1:]
            if sym in _SELFIES_ELEMENT_TOKENS:
                counts[sym] = counts.get(sym, 0) + 1
            continue
        if inner in _SELFIES_ELEMENT_TOKENS:
            counts[inner] = counts.get(inner, 0) + 1
    return counts if counts else None


def evaluate_atom_count(response: str,
                        item: Dict[str, Any],
                        instruction_name: str,
                        required_parameters: str = '',
                        llm_client=None,
                        **kwargs) -> Dict[str, Any]:
    if not required_parameters or not response:
        return {'passed': False, 'score': 0.0, 'detail': 'Missing required_parameters response is empty'}
    edit_question = item.get('edit_question', '') or '' if item else ''
    requirement_text = f"{required_parameters or ''}\n{edit_question}"
    target_counts = _parse_atom_requirements(requirement_text)
    target_total = _parse_total_atom_requirement(requirement_text)
    if not target_counts and target_total is None:
        return {
            'passed': False,
            'score': 0.0,
            'detail': f'Evaluator unable to parseatom :{required_parameters}',
            'skipped': True
        }
    mol_str, fmt, extract_src = extract_molecule(response, llm_client=llm_client, item=item)
    if not mol_str:
        return {'passed': False, 'score': 0.0, 'detail': 'molecule (SMILES/SELFIES)'}
    if HAS_RDKIT:
        actual, parse_note = _count_atoms_rdkit(mol_str, fmt)
    else:
        actual, parse_note = (_count_atoms_fallback(mol_str, fmt), 'not installed RDKit, atom token')
    if actual is None:
        reason = parse_note or 'RDKit cannot molecule'
        return {'passed': False, 'score': 0.0, 'detail': f'moleculeParsing failed; {reason}; extracted: {mol_str}'}
    mismatches = []
    for elem, count in target_counts.items():
        got = actual.get(elem, 0)
        if got != count:
            mismatches.append(f'{elem}: required {count}, actual {got}')
    if target_total is not None:
        got_total = sum(actual.values())
        if got_total != target_total:
            mismatches.append(f'atom : required{target_total}, actual {got_total}')
    passed = len(mismatches) == 0
    base_detail = 'atom count required' if passed else '; '.join(mismatches)
    extra = f'; {parse_note}' if parse_note else ''
    detail = f'{base_detail}; extracted({extract_src}): {mol_str}{extra}'
    return {'passed': passed, 'score': 1.0 if passed else 0.0, 'detail': detail}


def _parse_atom_requirements(text: str) -> Dict[str, int]:
    elem_map = {
        'hydrogen': 'H',
        'boron': 'B',
        'carbon': 'C',
        'nitrogen': 'N',
        'oxygen': 'O',
        'fluorine': 'F',
        'silicon': 'Si',
        'phosphorus': 'P',
        'sulfur': 'S',
        'chlorine': 'Cl',
        'arsenic': 'As',
        'selenium': 'Se',
        'bromine': 'Br',
        'antimony': 'Sb',
        'tellurium': 'Te',
        'iodine': 'I',
        'polonium': 'Po'
    }
    valid_symbols = set(elem_map.values())
    result: Dict[str, int] = {}
    pattern = '(\\d+)\\s+([A-Za-z]{1,12})\\s+atoms?\\b'
    for m in re.finditer(pattern, text or '', re.I):
        num, raw_name = (int(m.group(1)), m.group(2))
        name = raw_name.lower()
        elem = elem_map.get(name)
        if elem is None:
            symbol = raw_name[0].upper() + raw_name[1:].lower()
            elem = symbol if symbol in valid_symbols else None
        if elem:
            result[elem] = num
    return result


def _parse_total_atom_requirement(text: str) -> Optional[int]:
    for m in re.finditer('(?:exactly\\s+)?(\\d+)\\s+atoms?\\b', text or '', re.I):
        prefix = (text or '')[max(0, m.start() - 20):m.start()]
        if re.search('[A-Za-z]+\\s*$', prefix) and (not re.search('(?:exactly|total|contains?)\\s*$', prefix, re.I)):
            continue
        return int(m.group(1))
    return None


def _decode_selfies_for_rdkit(mol_str: str) -> Tuple[Optional[str], Optional[Dict[str, int]], str]:
    try:
        import selfies
        smiles = selfies.decoder(mol_str)
        if smiles is None:
            return (None, None, 'SELFIES decoder None')
        return (smiles, None, '')
    except ImportError:
        naive = _count_atoms_from_selfies_brackets(mol_str)
        if naive:
            return (None, naive, 'not installed selfies, token ( : pip install selfies)')
        return (None, None, 'not installed selfies , cannot SELFIES. : pip install selfies')
    except Exception as e:
        return (None, None, f'SELFIES decoder error: {e}')


def _count_atoms_rdkit(mol_str: str, fmt: str) -> Tuple[Optional[Dict[str, int]], str]:
    smiles = mol_str
    if (fmt or '').lower() == 'selfies':
        sm, naive, note = _decode_selfies_for_rdkit(mol_str)
        if naive is not None:
            return (naive, note)
        if sm is None:
            if mol_from_smiles_lenient(mol_str) is None:
                return (None, note)
            smiles = mol_str
        else:
            smiles = sm
    try:
        mol = mol_from_smiles_lenient(smiles)
        if mol is None:
            sm_preview = smiles if len(smiles) <= 220 else smiles[:220] + '...'
            return (None, f'RDKit MolFromSmiles failed( sanitize=False); decoded SMILES:{sm_preview!r}')
        counts: Dict[str, int] = {}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
        return (counts, '')
    except Exception as e:
        return (None, f'error:{e}')


def _count_atoms_fallback(mol_str: str, fmt: str = '') -> Optional[Dict[str, int]]:
    if (fmt or '').lower() == 'selfies':
        return _count_atoms_from_selfies_brackets(mol_str)
    counts: Dict[str, int] = {}
    bracket_spans = []
    for match in re.finditer('\\[([^\\]]+)\\]', mol_str):
        bracket_spans.append(match.span())
        inner = re.sub('^\\d+', '', match.group(1))
        atom = re.match('([A-Z][a-z]?|[bcnops])', inner)
        if atom:
            symbol = atom.group(1)
            symbol = symbol.capitalize() if symbol in 'bcnops' else symbol
            counts[symbol] = counts.get(symbol, 0) + 1
    chars = list(mol_str)
    for start, end in bracket_spans:
        chars[start:end] = ' ' * (end - start)
    unbracketed = ''.join(chars)
    for match in _SMILES_ELEMENT_RE.finditer(unbracketed):
        symbol = match.group(0)
        symbol = symbol.capitalize() if symbol in 'bcnops' else symbol
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts if counts else None
