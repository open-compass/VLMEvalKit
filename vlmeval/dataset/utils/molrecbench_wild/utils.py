"""Shared graph and SMILES evaluation helpers."""

from __future__ import annotations
import itertools
import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Iterator, Mapping, Sequence

from rdkit import Chem
from rdkit.Geometry import Point3D

from .constants import ABBR2MOLBLOCKS

GREEK_LETTERS = list("αβγδεζηθικλμνξοπρστυφχψω")
_GREEK_CHARS = frozenset(GREEK_LETTERS)
_ALL_GREEK_CHARS = frozenset(
    "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
)


def simplify_bonds(
    bonds: Iterable[Iterable[Any]],
    *,
    strict: bool = False,
    preserve_tail: bool = False,
) -> list[list[Any]]:
    """Collapse drawing-specific bond types to chemical bond orders."""

    simplified: list[list[Any]] = []
    for raw_bond in bonds:
        bond = list(raw_bond)
        if len(bond) < 3:
            if strict:
                raise ValueError(f"Invalid bond entry: {raw_bond!r}")
            continue
        atom_1, atom_2, bond_type = map(int, bond[:3])
        tail = bond[3:] if preserve_tail else []
        if bond_type in {1, 7, 8, 11, 12, 13, 15, 16, 17, 21, 23}:
            simplified.append([atom_1, atom_2, 1, *tail])
        elif bond_type in {2, 9, 10, 14, 18, 19}:
            simplified.append([atom_1, atom_2, 2, *tail])
        elif bond_type in {3, 20, 22}:
            simplified.append([atom_1, atom_2, 3, *tail])
        elif bond_type in {4, 5, 6}:
            simplified.append([atom_1, atom_2, bond_type, *tail])
    return simplified


def check_R_atom(symbols: Iterable[str]) -> bool:
    symbols_string = "".join(str(symbol) for symbol in symbols)
    return any(letter in symbols_string for letter in GREEK_LETTERS)


def is_special_R(symbol: str) -> bool:
    return any(letter in str(symbol) for letter in GREEK_LETTERS)


def special_R_stem(symbol: str) -> str:
    for index in range(len(symbol) - 1, -1, -1):
        if symbol[index] in _GREEK_CHARS:
            return symbol[:index]
    return symbol


def iter_special_R_substitution_mappings(
    symbols_gt: Iterable[str], symbols_pred: Iterable[str]
) -> Iterator[dict[str, str]]:
    def collect(symbols: Iterable[str]) -> dict[str, set[str]]:
        groups: dict[str, set[str]] = defaultdict(set)
        for symbol in symbols:
            if is_special_R(symbol):
                groups[special_R_stem(symbol)].add(symbol)
        return groups

    gt_groups = collect(symbols_gt)
    pred_groups = collect(symbols_pred)
    if set(gt_groups) != set(pred_groups):
        return
    if any(
        len(gt_groups[stem]) != len(pred_groups[stem])
        for stem in gt_groups
    ):
        return
    stems = sorted(gt_groups)
    permutations = [
        tuple(itertools.permutations(sorted(pred_groups[stem])))
        for stem in stems
    ]
    for combination in itertools.product(*permutations):
        mapping: dict[str, str] = {}
        for stem, permutation in zip(stems, combination):
            mapping.update(
                dict(zip(permutation, sorted(gt_groups[stem])))
            )
        yield mapping


def Convert_Rx_to_R(symbols: Iterable[str]) -> list[str]:
    return [
        "R" if is_special_R(symbol) else symbol for symbol in symbols
    ]


def normalize_greek_letters(
    items: Iterable[str], sort_items: bool = False
) -> list[str]:
    pattern = re.compile("[" + "".join(GREEK_LETTERS) + "]")
    greek_index = {
        character: index
        for index, character in enumerate(GREEK_LETTERS)
    }
    values = list(items)
    used_letters = sorted(
        {
            character
            for item in values
            for character in pattern.findall(item)
        },
        key=greek_index.__getitem__,
    )
    mapping = {
        original: GREEK_LETTERS[index]
        for index, original in enumerate(used_letters)
    }
    result = [
        pattern.sub(lambda match: mapping[match.group(0)], item)
        for item in values
    ]
    if sort_items:
        result.sort(
            key=lambda text: (
                greek_index[match.group(0)]
                if (match := pattern.search(text))
                else float("inf")
            )
        )
    return result


def simplify_R_group_in_symbols(symbols: Iterable[str]) -> list[str]:
    # Graph node construction treats ``[Rα]`` and ``Rα`` as the same
    # symbol by removing one pair of outer brackets.  Apply that
    # normalization before special-R counting/simplification as well, so the
    # special-R path does not reintroduce a distinction between the two forms.
    values = [
        symbol[1:-1]
        if symbol.startswith("[") and symbol.endswith("]")
        else symbol
        for symbol in symbols
    ]
    counts = Counter(values)

    def simplify(symbol: str) -> str:
        for index in range(len(symbol) - 1, -1, -1):
            if symbol[index] in _ALL_GREEK_CHARS:
                return symbol[:index] + "**"
        return "R**"

    return [
        simplify(symbol)
        if is_special_R(symbol) and counts[symbol] == 1
        else symbol
        for symbol in values
    ]


def _compare_bracket(
    bracket_gt: Mapping[str, Any],
    bracket_pred: Mapping[str, Any],
    mapping: Mapping[int, int],
) -> bool:
    if bracket_gt.get("alias") != bracket_pred.get("alias"):
        return False
    try:
        mapped_gt_atoms = sorted(
            mapping[int(atom)] for atom in bracket_gt.get("atoms", [])
        )
        pred_atoms = sorted(
            int(atom) for atom in bracket_pred.get("atoms", [])
        )
    except (KeyError, TypeError, ValueError):
        return False
    return mapped_gt_atoms == pred_atoms


def compare_brackets(
    brackets_gt: Iterable[Mapping[str, Any]] | None,
    brackets_pred: Iterable[Mapping[str, Any]] | None,
    mapping: Mapping[int, int] | None,
) -> bool:
    gt_values = list(brackets_gt or [])
    pred_values = list(brackets_pred or [])
    if not gt_values and not pred_values:
        return True
    if mapping is None or len(gt_values) != len(pred_values):
        return False
    used = [False] * len(pred_values)
    for bracket_gt in gt_values:
        for index, bracket_pred in enumerate(pred_values):
            if not used[index] and _compare_bracket(
                bracket_gt, bracket_pred, mapping
            ):
                used[index] = True
                break
        else:
            return False
    return True


CONFLICT_SYMBOLS = {"Ar", "Ts", "Ac", "D", "Np", "Sn", "Mo", "W"}
EXP_SKIP_SYMBOLS = {
    "Rα",
    "Rβ",
    "Rδ",
    "Rγ",
    "Rε",
    "R'",
    "*C",
    "Alkyl",
    "R2X",
    "Ar",
    "X",
    "TBETOf",
    "CEB",
    "OPA",
    "E",
    "*",
    "F5",
    "_AP1",
    "?",
    "_",
    "R_",
    "DG",
    "R1",
    "R2",
    "M",
    "Ha",
    "Hb",
    "Hc",
    "Hd",
    "AF",
    "FA",
    "Y",
    "Y1",
    "L",
    "BO",
    "Z",
    "PEG",
}
REVERSE_FUNCTIONAL_GROUP = {
    "NO2": "O2N",
    "O2N": "NO2",
    "CHO": "OHC",
    "OHC": "CHO",
    "OAc": "AcO",
    "AcO": "OAc",
    "OBn": "BnO",
    "BnO": "OBn",
    "OBz": "BzO",
    "BzO": "OBz",
    "NBoc": "BocN",
    "BocN": "NBoc",
    "OMs": "MsO",
    "MsO": "OMs",
    "OTf": "TfO",
    "TfO": "OTf",
    "OMe": "MeO",
    "MeO": "OMe",
    "SMe": "MeS",
    "MeS": "SMe",
    "NMe": "MeN",
    "MeN": "NMe",
    "OEt": "EtO",
    "EtO": "OEt",
    "NCF3": "F3CN",
    "CF3N": "NCF3",
    "OCF3": "F3CO",
    "F3CO": "OCF3",
    "CO2H": "HO2C",
    "HO2C": "CO2H",
    "CN": "NC",
    "NC": "CN",
    "SO3H": "HO3S",
    "HO3S": "SO3H",
    "OCH3": "CH3O",
    "CH3O": "OCH3",
    "CO": "OC",
    "OC": "CO",
    "(H3C)3C": "C(H3C)3",
    "C(H3C)3": "(H3C)3C",
    "NCS": "SCN",
    "SCN": "NCS",
    "RO2C": "CO2R",
    "CO2R": "CO2R",
    "CF3": "F3C",
    "F3C": "CF3",
    "NaO3S": "SO3Na",
    "SO3Na": "NaO3S",
    "OP": "PO",
    "PO": "OP",
    "RO": "OR",
    "OR": "RO",
    "TBSO": "OTBS",
    "OTBS": "TBSO",
    "PG1": "G1P",
    "G1P": "PG1",
    "SO3-": "-O3S",
    "-O3S": "SO3-",
}


def get_max_isotope_in_smiles(smiles: str, debug: bool = False) -> int:
    try:
        matches = re.findall(r"\[(\d+)\*]", smiles)
        return max(int(match) for match in matches) if matches else 0
    except Exception:
        if debug:
            raise
        return 0


def atomwise_tokenizer(smiles: str) -> list[str]:
    pattern = (
        r"(\[(?:[^\[\]]+|\[[^\[\]]+])+\]|Br?|Cl?|N|O|S|P|F|I|"
        r"b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|"
        r"\$|%[0-9]{2}|[0-9])"
    )
    return re.findall(pattern, smiles)


def canonicalize_smiles_w_superatom(
    smiles: Any,
    super_atom_map: dict[str, str] | None = None,
    ignore_chiral: bool = False,
    ignore_cistrans: bool = True,
    recover_super_atom: bool = True,
    replace_H: bool = False,
    kekule: bool = False,
    debug: bool = False,
) -> tuple[str, dict[str, str], bool]:
    """Canonicalize SMILES while temporarily replacing unknown bracket labels."""

    if not isinstance(smiles, str) or not smiles:
        return "", {}, False
    if super_atom_map is None:
        super_atom_map = {}
    else:
        super_atom_map = dict(super_atom_map)

    super_index = max(
        get_max_isotope_in_smiles(smiles), len(super_atom_map)
    )
    if ignore_cistrans:
        smiles = smiles.replace("/", "").replace("\\", "")

    tokens = atomwise_tokenizer(smiles)
    for index, token in enumerate(tokens):
        if not (token.startswith("[") and token.endswith("]")):
            continue
        symbol = token[1:-1]
        replacement: str | None = None
        if token in super_atom_map:
            replacement = super_atom_map[token]
        elif symbol.startswith("R") and symbol[1:].isdigit():
            pass
        elif replace_H and symbol == "H":
            pass
        elif symbol in CONFLICT_SYMBOLS:
            pass
        elif Chem.AtomFromSmiles(token) is not None:
            continue

        if replacement is None:
            super_index += 1
            replacement = f"[{super_index}Tc]"
            super_atom_map[token] = replacement
            reverse = REVERSE_FUNCTIONAL_GROUP.get(symbol)
            if reverse is not None:
                super_atom_map[f"[{reverse}]"] = replacement
        tokens[index] = replacement

    canonical = "".join(tokens)
    succeeded = True
    try:
        canonical = Chem.CanonSmiles(
            canonical, useChiral=(not ignore_chiral)
        )
    except Exception as exc:
        succeeded = False
        if debug:
            print(f"SMILES canonicalization failed for {canonical!r}: {exc}")

    if kekule:
        try:
            mol = Chem.MolFromSmiles(canonical, sanitize=False)
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            canonical = Chem.MolToSmiles(mol)
        except Exception as exc:
            if debug:
                print(f"SMILES kekulization failed for {canonical!r}: {exc}")

    if recover_super_atom:
        for original, replacement in super_atom_map.items():
            canonical = canonical.replace(replacement, original)
    if ignore_cistrans:
        canonical = canonical.replace("/", "").replace("\\", "")
    return canonical, super_atom_map, succeeded


def extract_brackets(smiles: str) -> list[str]:
    stack: list[int] = []
    results: list[str] = []
    for index, char in enumerate(smiles):
        if char == "[":
            stack.append(index)
        elif char == "]" and stack:
            start = stack.pop()
            if not stack:
                results.append(smiles[start + 1: index])
    return results


def replace_superatoms_with_ts(
    smiles: str, tmp_atom: str = "Tc", tmp_isotope: int = 20
) -> tuple[str, dict[str, str]]:
    superatom_map: dict[str, str] = {}
    for superatom in set(extract_brackets(smiles)):
        if superatom in CONFLICT_SYMBOLS:
            is_superatom = True
        else:
            is_superatom = Chem.AtomFromSmiles(f"[{superatom}]") is None
        if not is_superatom:
            continue
        placeholder = f"{tmp_isotope}{tmp_atom}"
        superatom_map[placeholder] = superatom
        smiles = smiles.replace(f"[{superatom}]", f"[{placeholder}]")
        tmp_isotope += 1
    return smiles, superatom_map


def parse_attachpoints_v2000(molblock: str) -> list[int]:
    attach_points: list[int] = []
    for line in molblock.splitlines():
        if not line.startswith("M  APO"):
            continue
        parts = line.split()
        atom_count = int(parts[2])
        offset = 3
        for _ in range(atom_count):
            atom_index = int(parts[offset]) - 1
            ap_count = int(parts[offset + 1])
            attach_points.append(atom_index)
            if ap_count == 3:
                attach_points.append(atom_index)
            offset += 2
    return attach_points


def replace_superatom_with_mol(
    smiles_main: str,
    canonical: bool = True,
    kekuleSmiles: bool = True,
    report_missing_abbr: bool = True,
    debug: bool = False,
) -> tuple[str, list[dict[str, Any]]]:
    """Expand bracketed abbreviations without reordering stereocentre neighbors."""

    smiles_with_placeholders, superatom_map = replace_superatoms_with_ts(
        smiles_main
    )
    mol_main = Chem.MolFromSmiles(smiles_with_placeholders, sanitize=False)
    if mol_main is None:
        raise ValueError(f"Cannot parse SMILES: {smiles_main!r}")

    rw_mol = Chem.RWMol(mol_main)
    missing_abbreviations: list[dict[str, Any]] = []
    placeholders = [
        (atom.GetIdx(), f"{atom.GetIsotope()}{atom.GetSymbol()}")
        for atom in rw_mol.GetAtoms()
        if f"{atom.GetIsotope()}{atom.GetSymbol()}" in superatom_map
    ]

    for atom_index, placeholder in placeholders:
        abbreviation = superatom_map[placeholder]
        variants = ABBR2MOLBLOCKS.get(abbreviation)
        if variants is None:
            if abbreviation not in EXP_SKIP_SYMBOLS:
                missing_abbreviations.append({"abbr": abbreviation})
            continue

        atom = rw_mol.GetAtomWithIdx(atom_index)
        neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        bond_types = {
            neighbor: rw_mol.GetBondBetweenAtoms(
                atom_index, neighbor
            ).GetBondType()
            for neighbor in neighbors
        }
        attachment_count = sum(set(bond_types.values()))
        attachment_key = str(attachment_count)
        if attachment_key not in variants:
            missing_abbreviations.append(
                {
                    "abbr": abbreviation,
                    "attachment_points_num": attachment_count,
                }
            )
            continue

        molblock = variants[attachment_key]
        fragment = Chem.MolFromMolBlock(molblock, removeHs=False)
        attach_points = parse_attachpoints_v2000(molblock)
        if fragment is None or not attach_points:
            continue

        first_attach = attach_points[0]
        source_atom = fragment.GetAtomWithIdx(first_attach)
        replacement_atom = Chem.Atom(source_atom.GetAtomicNum())
        replacement_atom.SetFormalCharge(source_atom.GetFormalCharge())
        rw_mol.ReplaceAtom(atom_index, replacement_atom)

        atom_map = {first_attach: atom_index}
        for source_index in range(fragment.GetNumAtoms()):
            if source_index != first_attach:
                atom_map[source_index] = rw_mol.AddAtom(
                    fragment.GetAtomWithIdx(source_index)
                )
        for bond in fragment.GetBonds():
            begin = atom_map[bond.GetBeginAtomIdx()]
            end = atom_map[bond.GetEndAtomIdx()]
            if rw_mol.GetBondBetweenAtoms(begin, end) is None:
                rw_mol.AddBond(begin, end, bond.GetBondType())

        for neighbor_index in range(
            1, min(len(neighbors), len(attach_points))
        ):
            main_neighbor = neighbors[neighbor_index]
            if (
                rw_mol.GetBondBetweenAtoms(atom_index, main_neighbor)
                is not None
            ):
                bond_type = bond_types[main_neighbor]
                rw_mol.RemoveBond(atom_index, main_neighbor)
                target = atom_map[attach_points[neighbor_index]]
                if rw_mol.GetBondBetweenAtoms(target, main_neighbor) is None:
                    rw_mol.AddBond(target, main_neighbor, bond_type)

    final_mol = rw_mol.GetMol()
    Chem.SanitizeMol(final_mol, catchErrors=True)
    try:
        expanded = Chem.MolToSmiles(
            final_mol,
            canonical=canonical,
            kekuleSmiles=kekuleSmiles,
        )
    except Exception:
        expanded = Chem.MolToSmiles(
            final_mol, canonical=canonical, kekuleSmiles=False
        )
    for placeholder, abbreviation in superatom_map.items():
        if placeholder in expanded:
            expanded = expanded.replace(placeholder, abbreviation)
    if report_missing_abbr and missing_abbreviations:
        print(f"WARNING: Missing abbreviations: {missing_abbreviations}")
    if debug:
        print(f"Expanded {smiles_main!r} to {expanded!r}")
    return expanded, missing_abbreviations


def _value(values: Sequence[Any] | None, index: int, default: Any) -> Any:
    if values is None or index >= len(values) or values[index] is None:
        return default
    return values[index]


def convert_graph_to_mol_block(
    symbols: Sequence[str],
    coords: Sequence[Sequence[float]],
    bonds: Sequence[Sequence[Any]],
    charges: Sequence[int | None] | None = None,
    radicals: Sequence[int | None] | None = None,
    valences: Sequence[int | None] | None = None,
    isotopes: Sequence[int | None] | None = None,
) -> tuple[str, dict[str, int]]:
    """Build the mol block used for Carbon-format SMILES conversion."""

    rw_mol = Chem.RWMol()
    superatom_map: dict[str, int] = {}
    next_superatom_isotope = 41

    for index, original_symbol in enumerate(symbols):
        symbol = str(original_symbol).replace("[", "").replace("]", "")
        atom = None
        if symbol not in CONFLICT_SYMBOLS:
            parse_symbol = f"[{symbol}]" if symbol in {"Fe", "H"} else symbol
            atom = Chem.AtomFromSmiles(parse_symbol)
        if atom is None:
            atom = Chem.Atom("Tc")
            if symbol not in superatom_map:
                superatom_map[symbol] = next_superatom_isotope
                next_superatom_isotope += 1
            atom.SetIsotope(superatom_map[symbol])

        charge = int(_value(charges, index, 0))
        if charge:
            atom.SetFormalCharge(charge)
        isotope_value = _value(isotopes, index, None)
        if isotope_value is not None:
            atom.SetIsotope(int(isotope_value))
        radical = _value(radicals, index, None)
        if radical in {1, 3}:
            atom.SetNumRadicalElectrons(2)
        elif radical == 2:
            atom.SetNumRadicalElectrons(1)
        atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        rw_mol.AddAtom(atom)

    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
        5: Chem.BondType.SINGLE,
        6: Chem.BondType.SINGLE,
    }
    for bond in bonds:
        atom_1, atom_2, bond_type = int(bond[0]), int(bond[1]), int(bond[2])
        if bond_type not in bond_types:
            raise ValueError(f"Unsupported simplified bond type: {bond_type}")
        rw_mol.AddBond(atom_1, atom_2, bond_types[bond_type])
        if bond_type == 5:
            rw_mol.GetBondBetweenAtoms(atom_1, atom_2).SetBondDir(
                Chem.BondDir.BEGINWEDGE
            )
        elif bond_type == 6:
            rw_mol.GetBondBetweenAtoms(atom_1, atom_2).SetBondDir(
                Chem.BondDir.BEGINDASH
            )

    mol = rw_mol.GetMol()
    if coords:
        conformer = Chem.Conformer(len(symbols))
        conformer.Set3D(False)
        for index, point in enumerate(coords):
            x = float(point[0]) if len(point) > 0 else 0.0
            y = float(point[1]) if len(point) > 1 else 0.0
            conformer.SetAtomPosition(index, Point3D(x, y, 0.0))
        mol.AddConformer(conformer)

    return Chem.MolToMolBlock(mol, kekulize=False), superatom_map


def carbon_to_smiles(record: Mapping[str, Any]) -> str:
    """Convert one Carbon-format annotation to unexpanded canonical SMILES."""

    symbols = record.get("symbols")
    bonds = record.get("bonds")
    coords = record.get("coords")
    if not isinstance(symbols, list) or not isinstance(bonds, list):
        raise ValueError("Carbon record must contain list fields 'symbols' and 'bonds'")
    if not isinstance(coords, list):
        coords = [[0.0, 0.0] for _ in symbols]

    mol_block, superatom_map = convert_graph_to_mol_block(
        symbols=symbols,
        coords=coords,
        bonds=simplify_bonds(bonds, strict=True, preserve_tail=True),
        charges=record.get("charges"),
        radicals=record.get("radicals"),
        valences=record.get("valences"),
        isotopes=record.get("isotopes"),
    )
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False)
    if mol is None:
        raise ValueError("RDKit failed to parse generated mol block")
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=True)
    except Exception:
        smiles = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
    for symbol, isotope in superatom_map.items():
        smiles = smiles.replace(f"{isotope}Tc", symbol)
    return smiles
