"""Minimal Carbon-format molecular graph used by Graph and S-Graph scoring."""

from __future__ import annotations
import math
import numbers
from copy import deepcopy
from typing import Any, Mapping

import networkx as nx

from .utils import simplify_bonds

DIRECTED_BOND_TYPES = {5, 6, 11, 13, 17, 21}


class MolGraph:
    """In-memory molecular graph for Carbon-format comparison."""

    def __init__(
        self,
        id: str | None = None,
        carbon_info: Mapping[str, Any] | None = None,
        attribute: Any = None,
        **_: Any,
    ) -> None:
        self.id = id
        self.attribute = attribute
        self.symbols: list[str] = []
        self.charges: list[Any] = []
        self.radicals: list[Any] = []
        self.valences: list[Any] = []
        self.isotopes: list[Any] = []
        self.attach_points: list[Any] = []
        self.coords: list[Any] = []
        self.bonds_list: list[list[Any]] = []
        self.brackets: list[dict[str, Any]] = []
        if carbon_info is not None:
            self.load_from_carbon_info(carbon_info)

    @staticmethod
    def _attribute_values(
        carbon_info: Mapping[str, Any],
        name: str,
        count: int,
        *,
        required: bool,
    ) -> list[Any]:
        value = carbon_info.get(name)
        if value is None:
            if required and name not in carbon_info:
                raise ValueError(f"Carbon graph is missing field {name!r}")
            return [None] * count
        if not isinstance(value, list) or len(value) != count:
            raise ValueError(
                f"Carbon field {name!r} must have {count} entries"
            )
        return deepcopy(value)

    def load_from_carbon_info(
        self, carbon_info: Mapping[str, Any]
    ) -> None:
        symbols = carbon_info.get("symbols")
        bonds = carbon_info.get("bonds")
        if not isinstance(symbols, list) or not isinstance(bonds, list):
            raise ValueError(
                "Carbon graph requires list fields 'symbols' and 'bonds'"
            )
        self.id = carbon_info.get("id", self.id)
        if any(not isinstance(symbol, str) or not symbol for symbol in symbols):
            raise ValueError("Every Carbon symbol must be a non-empty string")
        self.symbols = deepcopy(symbols)
        count = len(self.symbols)
        self.charges = self._attribute_values(
            carbon_info, "charges", count, required=True
        )
        self.radicals = self._attribute_values(
            carbon_info, "radicals", count, required=True
        )
        self.valences = self._attribute_values(
            carbon_info, "valences", count, required=True
        )
        self.isotopes = self._attribute_values(
            carbon_info, "isotopes", count, required=True
        )
        self.attach_points = self._attribute_values(
            carbon_info,
            "attach_points",
            count,
            required=False,
        )
        coords = carbon_info.get("coords")
        self.coords = (
            deepcopy(coords)
            if isinstance(coords, list)
            else [[0.0, 0.0] for _ in range(count)]
        )
        validated_bonds: list[list[int]] = []
        for raw_bond in bonds:
            if not isinstance(raw_bond, list) or len(raw_bond) != 3:
                raise ValueError(
                    f"Every Carbon bond must contain three integers: "
                    f"{raw_bond!r}"
                )
            normalized: list[int] = []
            for value in raw_bond:
                if (
                    isinstance(value, bool)
                    or not isinstance(value, numbers.Real)
                    or not math.isfinite(float(value))
                    or float(value) != int(value)
                ):
                    raise ValueError(
                        f"Every Carbon bond must contain integer-valued "
                        f"numbers: {raw_bond!r}"
                    )
                normalized.append(int(value))
            atom_1, atom_2, bond_type = normalized
            if not (0 <= atom_1 < count and 0 <= atom_2 < count):
                raise ValueError(
                    f"Carbon bond endpoint is out of range: {raw_bond!r}"
                )
            if not 1 <= bond_type <= 23:
                raise ValueError(
                    f"Carbon bond type is out of range: {raw_bond!r}"
                )
            validated_bonds.append([atom_1, atom_2, bond_type])
        self.bonds_list = validated_bonds
        brackets = carbon_info.get("brackets")
        if not isinstance(brackets, list):
            raise ValueError("Carbon field 'brackets' must be a list")
        self.brackets = (
            deepcopy(brackets) if isinstance(brackets, list) else []
        )

    @staticmethod
    def _symbol(symbol: str) -> str:
        if symbol.startswith("[") and symbol.endswith("]"):
            return symbol[1:-1]
        return symbol

    def dump_to_simplify_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for index, symbol in enumerate(self.symbols):
            graph.add_node(index, symbol=self._symbol(symbol))
        for atom_1, atom_2, bond_type in simplify_bonds(
            self.bonds_list
        ):
            graph.add_edge(atom_1, atom_2, bond=bond_type)
            if bond_type not in DIRECTED_BOND_TYPES:
                graph.add_edge(atom_2, atom_1, bond=bond_type)
        return graph

    def dump_to_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for index, symbol in enumerate(self.symbols):
            graph.add_node(
                index,
                symbol=self._symbol(symbol),
                charge=self.charges[index],
                radical=self.radicals[index],
                valence=self.valences[index],
                isotope=self.isotopes[index],
                attach_point=self.attach_points[index],
            )
        for atom_1, atom_2, bond_type in self.bonds_list:
            graph.add_edge(atom_1, atom_2, bond=bond_type)
            if bond_type not in DIRECTED_BOND_TYPES:
                graph.add_edge(atom_2, atom_1, bond=bond_type)
        return graph

    def dump_to_SMILES(
        self,
        expand: bool = True,
        super_atom_map: dict[str, str] | None = None,
        **_: Any,
    ) -> tuple[str, dict[str, str], list[dict[str, Any]]]:
        """Convert this Carbon-format graph to normalized SMILES."""

        from .utils import carbon_to_smiles, replace_superatom_with_mol

        mapping = dict(super_atom_map or {})
        smiles = carbon_to_smiles(self.dump_to_carbon())
        missing: list[dict[str, Any]] = []
        if expand:
            smiles, missing = replace_superatom_with_mol(
                smiles, report_missing_abbr=False
            )
        return smiles, mapping, missing

    def dump_to_carbon(self, simplify: bool = False) -> dict[str, Any]:
        if simplify:
            return {
                "symbols": deepcopy(self.symbols),
                "coords": deepcopy(self.coords),
                "bonds": simplify_bonds(self.bonds_list),
            }
        return {
            "symbols": deepcopy(self.symbols),
            "charges": deepcopy(self.charges),
            "radicals": deepcopy(self.radicals),
            "valences": deepcopy(self.valences),
            "isotopes": deepcopy(self.isotopes),
            "attach_points": deepcopy(self.attach_points),
            "coords": deepcopy(self.coords),
            "bonds": deepcopy(self.bonds_list),
            "brackets": deepcopy(self.brackets),
        }
