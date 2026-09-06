from __future__ import annotations
from typing import Any, Optional


def suppress_rdkit_logs() -> None:
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
        return
    except Exception:
        pass
    try:
        from rdkit import rdBase
        rdBase.DisableLog('rdApp.*')
    except Exception:
        pass


def enable_rdkit_logs() -> None:
    try:
        from rdkit import RDLogger
        RDLogger.EnableLog('rdApp.*')
        return
    except Exception:
        pass
    try:
        from rdkit import rdBase
        rdBase.EnableLog('rdApp.*')
    except Exception:
        pass


def mol_from_smiles_lenient(smiles: str) -> Optional[Any]:
    from rdkit import Chem
    if not smiles or not str(smiles).strip():
        return None
    suppress_rdkit_logs()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        return mol
    finally:
        enable_rdkit_logs()
