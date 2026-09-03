"""Convert VLMEvalKit prediction workbooks to evaluator JSONL files.

The converter deliberately keeps workbook validation separate from model-output
validation. Broken workbook structure (missing columns, duplicate IDs, or an
ambiguous sheet selection) is always fatal. A malformed model prediction keeps
its row in the output as an empty prediction and is reported as a diagnostic;
``--strict-predictions`` makes those diagnostics fatal as well.
"""

from __future__ import annotations
import argparse
import json
import math
import numbers
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .constants import (AROMATIC_BOND_ID, BOND_TYPES, CROSSED_DOUBLE_BOND_ID,
                        DASHED_DATIVE_BOND_ID, DATIVE_BOND_ID)

TRACKS = ("smiles", "s_graph", "graph")
TRACK_ALIASES = {
    "smiles": "smiles",
    "s_graph": "s_graph",
    "s-graph": "s_graph",
    "sgraph": "s_graph",
    "graph_simple": "s_graph",
    "simple_graph": "s_graph",
    "graph": "graph",
}
ATTACHMENT_BOND_TYPES = {"attachment point", "attachment_point"}
NORMALIZED_BOND_TYPES = {
    str(name).strip().lower(): bond_type for name, bond_type in BOND_TYPES.items()
}
BOND_TYPE_ALIASES = {
    "aromatic ring": AROMATIC_BOND_ID,
    "double either": CROSSED_DOUBLE_BOND_ID,
    "double_either": CROSSED_DOUBLE_BOND_ID,
    "dipolar": DATIVE_BOND_ID,
    "dashed dipolar": DASHED_DATIVE_BOND_ID,
    "dashed_dipolar": DASHED_DATIVE_BOND_ID,
}


class ConversionError(ValueError):
    """Base class for conversion failures."""


class WorkbookConversionError(ConversionError):
    """The workbook cannot be converted without guessing its structure."""


class PredictionConversionError(ConversionError):
    """A single model prediction is malformed or internally inconsistent."""


def normalize_track(track: str) -> str:
    """Return the canonical evaluator track name.

    Dataset IDs use human-facing spellings such as ``S-Graph`` while the
    upstream converter uses ``s_graph``.  Keeping the normalization here makes
    the conversion semantics identical for every caller.
    """

    if not isinstance(track, str):
        raise WorkbookConversionError("track must be a string")
    normalized = TRACK_ALIASES.get(track.strip().lower())
    if normalized is None:
        raise WorkbookConversionError(f"unsupported track: {track}")
    return normalized


@dataclass(frozen=True)
class ConversionIssue:
    sheet: str
    row: int
    id: str
    error: str


@dataclass(frozen=True)
class ConversionSummary:
    records: int
    prediction_errors: int
    sheets: tuple[str, ...]
    issues: tuple[ConversionIssue, ...]


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, numbers.Integral)) and bool(missing)


def normalize_id(value: Any, suffix: str = "") -> str:
    """Return a stable record ID, preserving workbook text by default."""

    if isinstance(value, bool) or _is_missing_scalar(value):
        raise WorkbookConversionError("record ID is missing")

    if isinstance(value, numbers.Integral):
        record_id = str(int(value))
    elif isinstance(value, numbers.Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise WorkbookConversionError("record ID is not finite")
        record_id = (
            str(int(numeric_value)) if numeric_value.is_integer() else str(value)
        )
    else:
        record_id = str(value).strip()

    if not record_id:
        raise WorkbookConversionError("record ID is empty")
    if suffix and not record_id.endswith(suffix):
        record_id += suffix
    return record_id


def normalize_prediction_id(value: Any, suffix: str = "") -> str:
    """Validate a benchmark prediction ID without coercing its type."""

    if _is_missing_scalar(value):
        raise WorkbookConversionError("record ID is missing")
    if not isinstance(value, str):
        raise WorkbookConversionError("record ID must be a string")
    return normalize_id(value, suffix)


def _safe_json_loads(text: str) -> Any:
    """Parse JSON, with a narrow fallback for legacy unescaped backslashes."""

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    def strict_loads(candidate: str) -> Any:
        return json.loads(candidate, parse_constant=reject_nonfinite)

    try:
        parsed = strict_loads(text)
    except ValueError as original_error:
        repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)
        if repaired == text:
            raise original_error
        return strict_loads(repaired)

    # A legacy model may emit a SMILES stereobond '\\' directly before an
    # aromatic n/b (or another JSON control-escape letter). Standard JSON then
    # silently turns it into a control character. Control characters are not
    # valid SMILES, so repair only that otherwise-ambiguous legacy case. Runs
    # containing an even number of backslashes are already valid JSON escapes.
    if (
        isinstance(parsed, dict)
        and isinstance(parsed.get("smiles"), str)
        and any(ord(character) < 32 for character in parsed["smiles"])
    ):
        repaired = re.sub(
            r"(\\+)(?=[bfnrt])",
            lambda match: (
                match.group(1) + "\\" if len(match.group(1)) % 2 else match.group(1)
            ),
            text,
        )
        if repaired != text:
            return strict_loads(repaired)
    return parsed


def _json_candidates(text: str) -> Iterable[str]:
    stripped = text.strip()
    if stripped:
        yield stripped

    for match in re.finditer(r"<\|begin_of_box\|>([\s\S]*?)<\|end_of_box\|>", text):
        yield match.group(1).strip()

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I):
        yield match.group(1).strip()

    # Also isolate balanced objects embedded in prose. This scanner only finds
    # object boundaries; _safe_json_loads still performs the actual JSON parse.
    # It therefore supports legacy SMILES outputs containing an unescaped '\\'.
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        in_string = False
        escaped = False
        for index, character in enumerate(text[start:], start=start):
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    yield text[start: index + 1]
                    break


def extract_prediction_object(value: Any, track: str) -> dict[str, Any]:
    """Extract the first track-compatible JSON object from model output."""

    if isinstance(value, Mapping):
        candidates: Iterable[Any] = (value,)
    else:
        if _is_missing_scalar(value):
            raise PredictionConversionError("prediction is missing")
        candidates = _json_candidates(str(value))

    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = (
                dict(candidate)
                if isinstance(candidate, Mapping)
                else _safe_json_loads(candidate)
            )
        except (ValueError, TypeError, RecursionError) as error:
            parse_errors.append(str(error))
            continue
        if not isinstance(parsed, dict):
            continue
        if track == "smiles" and "smiles" in parsed:
            return parsed
        if track != "smiles" and "atoms" in parsed:
            return parsed

    detail = f": {parse_errors[-1]}" if parse_errors else ""
    required = "smiles" if track == "smiles" else "atoms"
    raise PredictionConversionError(
        f"could not find a JSON object containing '{required}'{detail}"
    )


def empty_prediction(record_id: str, track: str) -> dict[str, Any]:
    if track == "smiles":
        return {
            "id": record_id,
            "smiles": "",
            "symbols": None,
            "charges": None,
            "radicals": None,
            "valences": None,
            "isotopes": None,
            "attach_points": None,
            "coords": None,
            "bonds": None,
            "brackets": None,
        }
    return {
        "id": record_id,
        "smiles": None,
        "symbols": [],
        "charges": [],
        "radicals": [],
        "valences": [],
        "isotopes": [],
        "attach_points": [],
        "coords": [],
        "bonds": [],
        "brackets": [],
    }


def _validated_record(record: dict[str, Any]) -> dict[str, Any]:
    try:
        serialized = json.dumps(record, ensure_ascii=False, allow_nan=False)
        serialized.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise PredictionConversionError(
            f"prediction cannot be represented as UTF-8 JSON: {error}"
        ) from error
    return record


def convert_smiles_prediction(record_id: str, value: Any) -> dict[str, Any]:
    parsed = extract_prediction_object(value, "smiles")
    smiles = parsed.get("smiles")
    if not isinstance(smiles, str):
        raise PredictionConversionError("'smiles' must be a string")
    record = empty_prediction(record_id, "smiles")
    record["smiles"] = smiles
    return _validated_record(record)


def convert_prediction(
    index: Any,
    prediction: Any,
    track: str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Convert one VLMEvalKit prediction to an official Carbon record.

    Invalid model responses are represented by an empty prediction with the
    original ID, matching :func:`convert_dataframe`.  Set ``strict=True`` to
    surface :class:`PredictionConversionError` for diagnostic tooling.  ID
    errors remain fatal because guessing an ID would invalidate coverage.
    """

    canonical_track = normalize_track(track)
    record_id = normalize_prediction_id(index)
    try:
        if canonical_track == "smiles":
            return convert_smiles_prediction(record_id, prediction)
        return convert_graph_prediction(record_id, prediction)
    except PredictionConversionError:
        if strict:
            raise
        return empty_prediction(record_id, canonical_track)


def _integer_id(value: Any, context: str) -> int:
    if isinstance(value, bool) or _is_missing_scalar(value):
        raise PredictionConversionError(f"{context} is missing")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            return int(numeric_value)
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
    raise PredictionConversionError(f"{context} must be an integer atom ID")


def _list_field(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise PredictionConversionError(f"{context} must be a JSON array")
    return value


def _optional_integer(atom: Mapping[str, Any], field: str, atom_id: int) -> int | None:
    value = atom.get(field)
    if value is None:
        return None
    return _integer_id(value, f"atom {atom_id} field {field!r}")


def _normalize_bond_type(value: Any) -> tuple[str, int | None]:
    if not isinstance(value, str) or not value.strip():
        raise PredictionConversionError("bond_type must be a non-empty string")
    name = value.strip().lower()
    if name in ATTACHMENT_BOND_TYPES:
        return name, None
    bond_type = NORMALIZED_BOND_TYPES.get(name, BOND_TYPE_ALIASES.get(name))
    if bond_type is None or bond_type == 0:
        raise PredictionConversionError(f"unknown bond type: {value!r}")
    return name, int(bond_type)


def convert_graph_prediction(record_id: str, value: Any) -> dict[str, Any]:
    """Convert one CARBON prediction and remap attachment dummy atoms once.

    An ``attachment point`` bond follows the historical workbook contract:
    ``atom1`` is retained and ``atom2`` is the dummy atom to remove.
    """

    parsed = extract_prediction_object(value, "graph")
    atoms = _list_field(parsed.get("atoms"), "'atoms'")
    raw_bonds = _list_field(parsed.get("bonds", []), "'bonds'")
    raw_brackets = _list_field(parsed.get("brackets", []), "'brackets'")

    atom_by_id: dict[int, dict[str, Any]] = {}
    atom_order: list[int] = []
    for position, raw_atom in enumerate(atoms):
        if not isinstance(raw_atom, Mapping):
            raise PredictionConversionError(f"atom {position} must be an object")
        atom = dict(raw_atom)
        atom_id = _integer_id(atom.get("id", position), f"atom {position} id")
        if atom_id in atom_by_id:
            raise PredictionConversionError(f"duplicate atom ID: {atom_id}")
        symbol = atom.get("atom")
        if not isinstance(symbol, str) or not symbol:
            raise PredictionConversionError(
                f"atom {atom_id} field 'atom' must be a non-empty string"
            )
        atom_by_id[atom_id] = atom
        atom_order.append(atom_id)

    ordinary_bonds: list[tuple[int, int, int]] = []
    attachment_bonds: list[tuple[int, int]] = []
    for position, raw_bond in enumerate(raw_bonds):
        if not isinstance(raw_bond, Mapping):
            raise PredictionConversionError(f"bond {position} must be an object")
        atom1 = _integer_id(raw_bond.get("atom1"), f"bond {position} atom1")
        atom2 = _integer_id(raw_bond.get("atom2"), f"bond {position} atom2")
        if atom1 not in atom_by_id or atom2 not in atom_by_id:
            raise PredictionConversionError(
                f"bond {position} references an unknown atom ID"
            )
        if atom1 == atom2:
            raise PredictionConversionError(f"bond {position} is a self-bond")
        name, bond_type = _normalize_bond_type(raw_bond.get("bond_type"))
        if name in ATTACHMENT_BOND_TYPES:
            attachment_bonds.append((atom1, atom2))
        else:
            assert bond_type is not None
            ordinary_bonds.append((atom1, atom2, bond_type))

    removed_ids = [dummy_id for _, dummy_id in attachment_bonds]
    if len(removed_ids) != len(set(removed_ids)):
        raise PredictionConversionError(
            "an attachment dummy atom is referenced more than once"
        )
    removed_id_set = set(removed_ids)
    if any(real_id in removed_id_set for real_id, _ in attachment_bonds):
        raise PredictionConversionError(
            "an attachment's retained atom is also marked for removal"
        )
    if any(
        atom1 in removed_id_set or atom2 in removed_id_set
        for atom1, atom2, _ in ordinary_bonds
    ):
        raise PredictionConversionError(
            "an attachment dummy atom is used by an ordinary bond"
        )

    bracket_data: list[tuple[str, list[int]]] = []
    for position, raw_bracket in enumerate(raw_brackets):
        if not isinstance(raw_bracket, Mapping):
            raise PredictionConversionError(f"bracket {position} must be an object")
        alias_value = raw_bracket.get("mark", raw_bracket.get("alias"))
        if (
            isinstance(alias_value, bool)
            or alias_value is None
            or isinstance(alias_value, (list, dict))
        ):
            raise PredictionConversionError(
                f"bracket {position} mark must be a string or number"
            )
        if isinstance(alias_value, numbers.Integral):
            alias = str(int(alias_value))
        elif isinstance(alias_value, numbers.Real):
            numeric_alias = float(alias_value)
            if not math.isfinite(numeric_alias):
                raise PredictionConversionError(
                    f"bracket {position} mark must be finite"
                )
            alias = (
                str(int(numeric_alias))
                if numeric_alias.is_integer()
                else str(alias_value)
            )
        else:
            alias = str(alias_value)
        bracket_atoms = _list_field(
            raw_bracket.get("atoms"), f"bracket {position} atoms"
        )
        atom_ids = [
            _integer_id(atom_id, f"bracket {position} atom")
            for atom_id in bracket_atoms
        ]
        if any(atom_id not in atom_by_id for atom_id in atom_ids):
            raise PredictionConversionError(
                f"bracket {position} references an unknown atom ID"
            )
        bracket_data.append(
            (alias, [atom_id for atom_id in atom_ids if atom_id not in removed_id_set])
        )

    kept_ids = [atom_id for atom_id in atom_order if atom_id not in removed_id_set]
    old_to_new = {atom_id: position for position, atom_id in enumerate(kept_ids)}
    attachment_counts = Counter(real_id for real_id, _ in attachment_bonds)

    record = empty_prediction(record_id, "graph")
    record["symbols"] = [atom_by_id[atom_id]["atom"] for atom_id in kept_ids]
    record["charges"] = [
        _optional_integer(atom_by_id[atom_id], "charge", atom_id)
        for atom_id in kept_ids
    ]
    record["radicals"] = [
        _optional_integer(atom_by_id[atom_id], "radical", atom_id)
        for atom_id in kept_ids
    ]
    record["valences"] = [
        _optional_integer(atom_by_id[atom_id], "valence", atom_id)
        for atom_id in kept_ids
    ]
    record["isotopes"] = [
        _optional_integer(atom_by_id[atom_id], "isotope", atom_id)
        for atom_id in kept_ids
    ]
    record["attach_points"] = [
        attachment_counts.get(atom_id) or None for atom_id in kept_ids
    ]
    record["coords"] = [atom_by_id[atom_id].get("point_2d") for atom_id in kept_ids]
    record["bonds"] = [
        [old_to_new[atom1], old_to_new[atom2], bond_type]
        for atom1, atom2, bond_type in ordinary_bonds
    ]
    record["brackets"] = [
        {
            "alias": alias,
            "atoms": [old_to_new[atom_id] for atom_id in atom_ids],
        }
        for alias, atom_ids in bracket_data
    ]
    return _validated_record(record)


def convert_dataframe(
    dataframe: pd.DataFrame,
    *,
    sheet: str = "Sheet1",
    id_suffix: str = "",
) -> tuple[list[dict[str, Any]], list[ConversionIssue]]:
    required_columns = {"index", "track", "prediction"}
    missing_columns = sorted(required_columns - set(dataframe.columns))
    if missing_columns:
        raise WorkbookConversionError(
            f"sheet {sheet!r} is missing columns: {', '.join(missing_columns)}"
        )

    records: list[dict[str, Any]] = []
    issues: list[ConversionIssue] = []
    seen_ids: set[str] = set()
    for offset, (_, row) in enumerate(dataframe.iterrows(), start=2):
        try:
            record_id = normalize_prediction_id(row["index"], id_suffix)
        except WorkbookConversionError as error:
            raise WorkbookConversionError(
                f"sheet {sheet!r}, row {offset}: {error}"
            ) from error
        if record_id in seen_ids:
            raise WorkbookConversionError(
                f"sheet {sheet!r}, row {offset}: duplicate ID {record_id!r}"
            )
        seen_ids.add(record_id)

        track = normalize_track(row["track"])
        try:
            if track == "smiles":
                record = convert_smiles_prediction(record_id, row["prediction"])
            else:
                record = convert_graph_prediction(record_id, row["prediction"])
        except PredictionConversionError as error:
            record = empty_prediction(record_id, track)
            issues.append(
                ConversionIssue(
                    sheet=sheet,
                    row=offset,
                    id=record_id,
                    error=str(error),
                )
            )
        records.append(record)
    return records, issues


def write_jsonl_atomic(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, destination)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _selected_sheets(
    sheet_names: Sequence[str], sheet: str | None, all_sheets: bool
) -> tuple[str, ...]:
    if sheet and all_sheets:
        raise WorkbookConversionError(
            "--sheet and --all-sheets cannot be used together"
        )
    if sheet:
        if sheet not in sheet_names:
            available = ", ".join(sheet_names)
            raise WorkbookConversionError(
                f"sheet {sheet!r} not found; available sheets: {available}"
            )
        return (sheet,)
    if all_sheets:
        return tuple(sheet_names)
    if len(sheet_names) != 1:
        available = ", ".join(sheet_names)
        raise WorkbookConversionError(
            "workbook has multiple sheets; choose --sheet NAME or "
            f"--all-sheets (available: {available})"
        )
    return (sheet_names[0],)


def convert_workbook(
    input_path: str | Path,
    output_path: str | Path,
    *,
    track: str,
    sheet: str | None = None,
    all_sheets: bool = False,
    id_suffix: str = "",
    strict_predictions: bool = False,
    errors_output: str | Path | None = None,
) -> ConversionSummary:
    input_resolved = Path(input_path).expanduser().resolve()
    output_resolved = Path(output_path).expanduser().resolve()
    errors_resolved = (
        Path(errors_output).expanduser().resolve()
        if errors_output is not None
        else None
    )
    if input_resolved == output_resolved:
        raise WorkbookConversionError("input and output paths must be different")
    if errors_resolved is not None and errors_resolved in {
        input_resolved,
        output_resolved,
    }:
        raise WorkbookConversionError(
            "--errors-output must differ from the input and prediction output"
        )

    try:
        excel = pd.ExcelFile(input_path)
    except (OSError, ValueError) as error:
        raise WorkbookConversionError(
            f"could not open workbook {str(input_path)!r}: {error}"
        ) from error

    with excel:
        selected_sheets = _selected_sheets(excel.sheet_names, sheet, all_sheets)
        records: list[dict[str, Any]] = []
        issues: list[ConversionIssue] = []
        seen_ids: dict[str, str] = {}
        for sheet_name in selected_sheets:
            try:
                dataframe = pd.read_excel(excel, sheet_name=sheet_name)
            except Exception as error:
                raise WorkbookConversionError(
                    f"could not read sheet {sheet_name!r}: {error}"
                ) from error
            sheet_records, sheet_issues = convert_dataframe(
                dataframe,
                track=track,
                sheet=sheet_name,
                id_suffix=id_suffix,
            )
            for record in sheet_records:
                previous_sheet = seen_ids.get(record["id"])
                if previous_sheet is not None:
                    raise WorkbookConversionError(
                        f"duplicate ID {record['id']!r} across sheets "
                        f"{previous_sheet!r} and {sheet_name!r}"
                    )
                seen_ids[record["id"]] = sheet_name
            records.extend(sheet_records)
            issues.extend(sheet_issues)

    if not records:
        raise WorkbookConversionError("selected workbook sheet(s) contain no records")

    if errors_output is not None:
        try:
            write_jsonl_atomic(errors_output, (asdict(issue) for issue in issues))
        except OSError as error:
            raise ConversionError(
                f"could not write diagnostics to {str(errors_output)!r}: {error}"
            ) from error

    if strict_predictions and issues:
        first = issues[0]
        raise PredictionConversionError(
            f"{len(issues)} prediction(s) failed; first error at "
            f"sheet {first.sheet!r}, row {first.row}, ID {first.id!r}: "
            f"{first.error}"
        )

    try:
        write_jsonl_atomic(output_path, records)
    except OSError as error:
        raise ConversionError(
            f"could not write predictions to {str(output_path)!r}: {error}"
        ) from error
    return ConversionSummary(
        records=len(records),
        prediction_errors=len(issues),
        sheets=selected_sheets,
        issues=tuple(issues),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a VLMEvalKit prediction workbook to JSONL."
    )
    parser.add_argument("-i", "--input", required=True, help="Input XLSX path")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--track",
        required=True,
        choices=TRACKS,
        help="Prediction track; never inferred from a filename",
    )
    sheet_group = parser.add_mutually_exclusive_group()
    sheet_group.add_argument("--sheet", help="Convert exactly one named sheet")
    sheet_group.add_argument(
        "--all-sheets",
        action="store_true",
        help="Merge all sheets in workbook order; IDs must remain unique",
    )
    parser.add_argument(
        "--id-suffix",
        default="",
        help="Append a suffix only when absent (IDs are preserved by default)",
    )
    parser.add_argument(
        "--strict-predictions",
        action="store_true",
        help="Fail without replacing the output if any prediction is malformed",
    )
    parser.add_argument(
        "--errors-output",
        help="Optional JSONL path for row-level prediction diagnostics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = convert_workbook(
            args.input,
            args.output,
            track=args.track,
            sheet=args.sheet,
            all_sheets=args.all_sheets,
            id_suffix=args.id_suffix,
            strict_predictions=args.strict_predictions,
            errors_output=args.errors_output,
        )
    except ConversionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    sheet_text = ", ".join(summary.sheets)
    for issue in summary.issues[:5]:
        print(
            f"warning: sheet {issue.sheet!r}, row {issue.row}, "
            f"ID {issue.id!r}: {issue.error}",
            file=sys.stderr,
        )
    if len(summary.issues) > 5:
        print(
            f"warning: {len(summary.issues) - 5} additional prediction errors",
            file=sys.stderr,
        )
    print(
        f"Converted {summary.records} records from {sheet_text} "
        f"to {args.output} ({summary.prediction_errors} prediction errors)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
