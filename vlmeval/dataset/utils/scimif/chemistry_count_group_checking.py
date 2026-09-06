import re
from typing import Any, Dict, Iterable

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
from .extraction_utils import extract_molecule
from .rdkit_utils import mol_from_smiles_lenient

FUNCTIONAL_GROUP_SMARTS = {
    'hydroxyl': '[OX2H;!$(O[C,S,P]=O)]',
    'carboxyl': '[CX3](=O)[OX2H1]',
    'carbonyl': '[CX3]=[OX1]',
    'amine': '[NX3;!$(N[CX3](=O))]',
    'amide': '[CX3](=O)[NX3]',
    'ester': '[CX3](=O)[OX2][#6;!$(C=O)]',
    'ether': '[OD2;!$([O][C,S,P]=O)]([#6])[#6]',
    'aldehyde': '[CX3H1](=O)[#6,#1]',
    'ketone': '[#6][CX3](=O)[#6]',
    'benzene': 'c1ccccc1',
    'sulfone': '[SX4](=[OX1])(=[OX1])',
    'sulfoxide': '[SX3](=[OX1])([#6])[#6]',
    'sulfide': '[SX2]([#6])[#6]',
    'disulfide': '[SX2]-[SX2]',
    'nitro': '[NX3+](=[OX1])[O-]',
    'nitrile': '[CX2]#[NX1]',
    'halo': '[#6]-[F,Cl,Br,I]',
    'anhydride': '[CX3](=O)[OX2][CX3](=O)',
    'borane': '[BX3;H1,H2,H3]',
    'thiol': '[SX2H]'
}
GROUP_PATTERNS = [('carboxyl', 'carboxyl(?:ic\\s+acid)?'), ('hydroxyl', 'hydroxyl|hydroxy|alcohol'),
                  ('amine', 'amine|amino'), ('amide', 'amide'), ('aldehyde', 'aldehyde'), ('ketone', 'ketone'),
                  ('benzene', 'benzene(?:\\s+ring)?|phenyl'), ('ester', 'ester'), ('ether', 'ether'),
                  ('sulfone', 'sulfone'), ('sulfoxide', 'sulfoxide'), ('disulfide', 'disulfide'),
                  ('sulfide', 'sulfide|thioether'), ('nitro', 'nitro'), ('nitrile', 'nitrile|cyano'),
                  ('halo', 'halo|halide'), ('anhydride', 'anhydride'), ('borane', 'borane'), ('carbonyl', 'carbonyl'),
                  ('thiol', 'thiol|sulfhydryl')]


def evaluate_group_count(response: str,
                         item: Dict[str, Any],
                         instruction_name: str,
                         required_parameters: str = '',
                         llm_client=None,
                         **kwargs) -> Dict[str, Any]:
    if not required_parameters or not response:
        return {'passed': False, 'score': 0.0, 'detail': 'Missing required_parameters response is empty'}
    edit_question = item.get('edit_question', '') or '' if item else ''
    target = _parse_group_requirements(f"{required_parameters or ''}\n{edit_question}")
    if not target:
        return {
            'passed': False,
            'score': 0.0,
            'detail': f'Evaluator unable to parsefunctional group :{required_parameters}',
            'skipped': True
        }
    if not HAS_RDKIT:
        return {
            'passed': False,
            'score': 0.0,
            'detail': 'RDKit not installed, cannot reliably functional group',
            'skipped': True
        }
    mol_str, fmt, extract_src = extract_molecule(response, llm_client=llm_client, item=item)
    if not mol_str:
        return {'passed': False, 'score': 0.0, 'detail': 'molecule'}
    actual = _count_groups_rdkit(mol_str, fmt, target.keys())
    if actual is None:
        return {'passed': False, 'score': 0.0, 'detail': f'moleculeParsing failed; extracted: {mol_str}'}
    mismatches = []
    for group, requirement in target.items():
        count = requirement['count']
        mode = requirement['mode']
        got = actual.get(group, 0)
        ok = mode == 'exactly' and got == count or (mode == 'at_least' and got >= count) or (mode == 'at_most'
                                                                                             and got <= count)
        if not ok:
            mode_zh = {'exactly': 'exactly', 'at_least': 'at least', 'at_most': 'at most'}[mode]
            mismatches.append(f'{group}: required{mode_zh} {count}, actual {got}')
    passed = len(mismatches) == 0
    base_detail = 'functional group count required' if passed else '; '.join(mismatches)
    detail = f'{base_detail}; extracted({extract_src}): {mol_str}'
    return {'passed': passed, 'score': 1.0 if passed else 0.0, 'detail': detail}


def _parse_group_requirements(text: str) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    blob = text or ''
    for canonical, group_pattern in GROUP_PATTERNS:
        pattern = f'(?:(exactly|at\\s+least|at\\s+most)\\s+)?(\\d+)\\s+(?:{group_pattern})s?(?:\\s+(?:functional\\s+)?groups?|\\s+rings?)?\\b'  # noqa: E501
        for m in re.finditer(pattern, blob, re.I):
            qualifier = (m.group(1) or '').lower().replace(' ', '_')
            if qualifier not in {'at_least', 'at_most'}:
                qualifier = 'exactly'
            result[canonical] = {'count': int(m.group(2)), 'mode': qualifier}
            break
    return result


def _count_groups_rdkit(mol_str: str, fmt: str, groups: Iterable[str]) -> Dict[str, int]:
    try:
        smiles = mol_str
        if (fmt or '').lower() == 'selfies':
            try:
                import selfies
                decoded = selfies.decoder(mol_str)
                if decoded:
                    smiles = decoded
            except Exception:
                smiles = mol_str
        mol = mol_from_smiles_lenient(smiles)
        if mol is None:
            return None
        result = {}
        for g in groups:
            smarts = FUNCTIONAL_GROUP_SMARTS.get(g)
            if smarts:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    result[g] = len(mol.GetSubstructMatches(pat, uniquify=True))
        return result
    except Exception:
        return None
