"""Direct-SMILES scoring against Carbon-format graph annotations.

The scorer filters unsupported GT annotations, converts each remaining
Carbon graph to an unexpanded SMILES, expands known abbreviations, canonicalizes
GT and prediction with RDKit, and counts exact canonical matches. Evaluation
ignores double-bond cis/trans slash directions by default while still comparing
atom chirality. The unified CLI provides ``--preserve-cistrans`` to compare
both forms of stereochemistry.
"""

from __future__ import annotations
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rdkit import RDLogger

from .utils import canonicalize_smiles_w_superatom, carbon_to_smiles, replace_superatom_with_mol

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GT = REPO_ROOT / "dataset" / "annotation.jsonl"
DEFAULT_PRED = (
    REPO_ROOT
    / "results"
    / "MLLM"
    / "GPT-5.6-sol"
    / "GPT-5.6-sol_smiles.jsonl"
)
GREEK_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]")
ANNOTATION_METADATA_FIELDS = ("attach_points",)
FILTER_REASON_DESCRIPTIONS = {
    "greek_letter_in_symbol": "GT symbols contains a Greek letter.",
    "question_mark_in_symbol": (
        "GT symbols contains a half-width or full-width question mark."
    ),
    "nonempty_brackets": "GT brackets contains annotation information.",
    "bond_type_over_6": (
        "GT contains a non-common bond type whose value is greater than 6."
    ),
    "nonempty_attach_points": (
        "GT attach_points contains attachment-point information."
    ),
    "gt_carbon_to_smiles_failed": (
        "The GT Carbon graph could not be converted to a non-empty SMILES."
    ),
    "gt_unexpandable_abbreviation": (
        "GT contains an abbreviation without a usable expansion template."
    ),
    "gt_canonicalization_failed": (
        "RDKit could not canonicalize the expanded GT SMILES."
    ),
}


def _filter_reason_description(reason: str) -> str:
    if reason in FILTER_REASON_DESCRIPTIONS:
        return FILTER_REASON_DESCRIPTIONS[reason]
    if reason.startswith("nonempty_"):
        field = reason.removeprefix("nonempty_")
        return f"GT {field} contains annotation information."
    return reason


def _is_nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(_is_nonempty(item) for item in value)
    if isinstance(value, dict):
        return bool(value)
    return value not in ("", 0)


def has_bond_type_over_6(bonds: Any) -> bool:
    if not isinstance(bonds, (list, tuple)):
        return False
    for bond in bonds:
        if not isinstance(bond, (list, tuple)) or len(bond) < 3:
            continue
        try:
            if float(bond[2]) > 6:
                return True
        except (TypeError, ValueError):
            continue
    return False


def filter_reasons(
    gt_record: dict[str, Any],
    metadata_fields: tuple[str, ...] = ANNOTATION_METADATA_FIELDS,
) -> list[str]:
    """Return every reason that excludes a GT record from SMILES scoring."""

    symbols = gt_record.get("symbols") or []
    reasons: list[str] = []
    if any(GREEK_RE.search(str(symbol)) for symbol in symbols):
        reasons.append("greek_letter_in_symbol")
    if any("?" in str(symbol) or "？" in str(symbol) for symbol in symbols):
        reasons.append("question_mark_in_symbol")
    if _is_nonempty(gt_record.get("brackets")):
        reasons.append("nonempty_brackets")
    if has_bond_type_over_6(gt_record.get("bonds")):
        reasons.append("bond_type_over_6")
    for field in metadata_fields:
        if _is_nonempty(gt_record.get(field)):
            reasons.append(f"nonempty_{field}")
    return reasons


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON ({exc})")
                continue
            if not isinstance(record, dict):
                errors.append(f"line {line_number}: record is not an object")
                continue
            records.append(record)
    return records, errors


def resolve_jsonl_path(path: Path, role: str) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        if role == "gt":
            candidate = path / "annotation.jsonl"
            if candidate.is_file():
                return candidate
        patterns = (
            ("*smiles.jsonl", "*.jsonl")
            if role == "prediction"
            else ("*.jsonl",)
        )
        for pattern in patterns:
            candidates = sorted(path.glob(pattern))
            if len(candidates) == 1:
                return candidates[0]
            if len(candidates) > 1:
                raise ValueError(
                    f"Expected one {pattern} file in {path}, "
                    f"found {len(candidates)}"
                )
        raise FileNotFoundError(f"No JSONL file found in {path}")
    raise FileNotFoundError(f"Input path does not exist: {path}")


def _index_by_id(
    records: Iterable[dict[str, Any]], source_name: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row_number, record in enumerate(records, start=1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{source_name} row {row_number}: missing string id")
        elif record_id in indexed:
            errors.append(f"{source_name}: duplicate id {record_id!r}")
        else:
            indexed[record_id] = record
    return indexed, errors


def prediction_coverage_errors(
    full_gt_records: Iterable[dict[str, Any]],
    evaluated_gt_records: Iterable[dict[str, Any]],
    prediction_records: Iterable[dict[str, Any]],
    *,
    prediction_read_errors: Iterable[str] = (),
) -> list[str]:
    """Return prediction ID/read errors for the requested GT coverage.

    Predictions for GT records outside a limited evaluation are allowed, but
    IDs outside the complete GT are always considered extra.
    """

    full_gt_by_id, gt_index_errors = _index_by_id(full_gt_records, "GT")
    if gt_index_errors:
        raise ValueError("; ".join(gt_index_errors))
    evaluated_gt_by_id, evaluated_gt_errors = _index_by_id(
        evaluated_gt_records, "evaluated GT"
    )
    if evaluated_gt_errors:
        raise ValueError("; ".join(evaluated_gt_errors))
    prediction_by_id, prediction_index_errors = _index_by_id(
        prediction_records, "prediction"
    )

    errors = [
        f"prediction read error: {error}"
        for error in prediction_read_errors
    ]
    errors.extend(prediction_index_errors)
    missing = sorted(evaluated_gt_by_id.keys() - prediction_by_id.keys())
    extra = sorted(prediction_by_id.keys() - full_gt_by_id.keys())
    if missing:
        errors.append(
            f"prediction coverage: {len(missing)} evaluated GT IDs are "
            f"missing (examples: {', '.join(repr(value) for value in missing[:3])})"
        )
    if extra:
        errors.append(
            f"prediction coverage: {len(extra)} IDs are outside the full GT "
            f"(examples: {', '.join(repr(value) for value in extra[:3])})"
        )
    return errors


def _normalize_missing_abbreviations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            abbreviation = item.get("abbr")
            if abbreviation in (None, ""):
                continue
            entry: dict[str, Any] = {"abbr": str(abbreviation)}
            if item.get("attachment_points_num") is not None:
                entry["attachment_points_num"] = item[
                    "attachment_points_num"
                ]
            normalized.append(entry)
        elif item not in (None, ""):
            normalized.append({"abbr": str(item)})
    return normalized


def normalize_smiles(
    smiles: Any,
    super_atom_map: dict[str, str] | None,
    ignore_cistrans: bool,
) -> tuple[str, str, dict[str, str], bool, bool, list[dict[str, Any]]]:
    """Expand and canonicalize one SMILES string."""

    if not isinstance(smiles, str) or not smiles.strip():
        return "", "", dict(super_atom_map or {}), False, False, []
    raw = smiles.strip()
    try:
        expanded, missing = replace_superatom_with_mol(
            raw, report_missing_abbr=False
        )
    except Exception:
        return "", "", dict(super_atom_map or {}), False, False, []
    missing_abbreviations = _normalize_missing_abbreviations(missing)
    if not isinstance(expanded, str) or not expanded.strip():
        return (
            "",
            "",
            dict(super_atom_map or {}),
            False,
            False,
            missing_abbreviations,
        )
    canonical, updated_map, success = canonicalize_smiles_w_superatom(
        expanded,
        super_atom_map=dict(super_atom_map or {}),
        ignore_cistrans=ignore_cistrans,
    )
    return (
        expanded,
        canonical,
        updated_map,
        True,
        bool(success),
        missing_abbreviations,
    )


def _write_missing_abbreviations_csv(
    path: Path, occurrences: list[dict[str, Any]]
) -> None:
    aggregates: dict[str, dict[str, Any]] = {}
    for occurrence in occurrences:
        abbreviation = str(occurrence["abbr"])
        aggregate = aggregates.setdefault(
            abbreviation,
            {
                "occurrence_count": 0,
                "gt_occurrence_count": 0,
                "pred_occurrence_count": 0,
                "sample_ids": set(),
                "attachment_point_counts": Counter(),
            },
        )
        aggregate["occurrence_count"] += 1
        aggregate[f"{occurrence['source']}_occurrence_count"] += 1
        aggregate["sample_ids"].add(str(occurrence["id"]))
        attachment_points = occurrence.get("attachment_points_num")
        if attachment_points is not None:
            aggregate["attachment_point_counts"][str(attachment_points)] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "abbreviation",
        "occurrence_count",
        "sample_count",
        "gt_occurrence_count",
        "pred_occurrence_count",
        "attachment_point_counts",
        "sample_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for abbreviation, aggregate in sorted(
            aggregates.items(),
            key=lambda item: (-item[1]["occurrence_count"], item[0]),
        ):
            sample_ids = sorted(aggregate["sample_ids"])
            writer.writerow(
                {
                    "abbreviation": abbreviation,
                    "occurrence_count": aggregate["occurrence_count"],
                    "sample_count": len(sample_ids),
                    "gt_occurrence_count": aggregate[
                        "gt_occurrence_count"
                    ],
                    "pred_occurrence_count": aggregate[
                        "pred_occurrence_count"
                    ],
                    "attachment_point_counts": json.dumps(
                        dict(
                            sorted(
                                aggregate[
                                    "attachment_point_counts"
                                ].items()
                            )
                        ),
                        ensure_ascii=False,
                    ),
                    "sample_ids": "; ".join(sample_ids),
                }
            )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_details_csv(
    path: Path, details: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for detail in details:
        for key in detail:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow(
                {key: _csv_value(detail.get(key)) for key in fieldnames}
            )


def evaluate(
    gt_path: Path,
    pred_path: Path,
    output_path: Path | None = None,
    summary_path: Path | None = None,
    missing_abbreviations_path: Path | None = None,
    metadata_fields: tuple[str, ...] = ANNOTATION_METADATA_FIELDS,
    ignore_cistrans: bool = True,
    limit: int | None = None,
    output_csv_path: Path | None = None,
    allow_coverage_mismatch: bool = False,
) -> dict[str, Any]:
    """Run SMILES evaluation and return its aggregate summary."""

    gt_path = resolve_jsonl_path(Path(gt_path), "gt")
    pred_path = resolve_jsonl_path(Path(pred_path), "prediction")
    gt_records, gt_read_errors = read_jsonl(gt_path)
    pred_records, pred_read_errors = read_jsonl(pred_path)
    if gt_read_errors:
        raise ValueError("; ".join(f"GT read error: {error}" for error in gt_read_errors))
    full_gt_records = gt_records
    gt_index_errors: list[str] = []
    pred_by_id, pred_index_errors = _index_by_id(
        pred_records, "prediction"
    )
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        gt_records = gt_records[:limit]
    coverage_errors = prediction_coverage_errors(
        full_gt_records,
        gt_records,
        pred_records,
        prediction_read_errors=pred_read_errors,
    )
    if coverage_errors and not allow_coverage_mismatch:
        raise ValueError(
            "Prediction coverage validation failed: "
            + "; ".join(coverage_errors)
        )
    subset_by_id = {
        record.get("id"): record.get("evaluation_subset")
        for record in gt_records
    }

    filter_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    missing_occurrences: list[dict[str, Any]] = []
    included_ids: set[Any] = set()
    correct_count = 0
    included_count = 0
    converted_gt_count = 0
    gt_expansion_failed_count = 0
    gt_unexpandable_count = 0
    pred_expansion_failed_count = 0
    gt_canonicalization_failed_count = 0
    pred_canonicalization_failed_count = 0

    for gt_record in gt_records:
        record_id = gt_record.get("id")
        pred_record = pred_by_id.get(record_id)
        reasons = filter_reasons(gt_record, metadata_fields)
        filter_counts.update(reasons)
        detail: dict[str, Any] = {
            "id": record_id,
            "included": not reasons,
            "filter_reasons": reasons,
            "filter_reason_details": [
                _filter_reason_description(reason) for reason in reasons
            ],
            "correct": False,
            "gt_smiles_raw": None,
            "gt_smiles_expanded": None,
            "gt_smiles_canonical": None,
            "gt_missing_abbreviations": [],
            "pred_smiles_raw": (
                pred_record.get("smiles") if pred_record else None
            ),
            "pred_smiles_expanded": None,
            "pred_smiles_canonical": None,
            "pred_missing_abbreviations": [],
            "gt_conversion_success": False,
            "gt_abbreviation_expansion_success": False,
            "pred_abbreviation_expansion_success": False,
            "gt_canonicalization_success": False,
            "pred_canonicalization_success": False,
            "error": None,
        }
        if reasons:
            status_counts["filtered"] += 1
            details.append(detail)
            continue

        try:
            gt_raw = carbon_to_smiles(gt_record)
            if not isinstance(gt_raw, str) or not gt_raw.strip():
                raise ValueError("conversion returned an empty SMILES")
            detail["gt_smiles_raw"] = gt_raw
            detail["gt_conversion_success"] = True
            converted_gt_count += 1
        except Exception as exc:
            reason = "gt_carbon_to_smiles_failed"
            detail["included"] = False
            detail["filter_reasons"].append(reason)
            detail["filter_reason_details"].append(
                _filter_reason_description(reason)
            )
            detail["error"] = f"gt_conversion_error: {exc}"
            filter_counts[reason] += 1
            status_counts["filtered"] += 1
            details.append(detail)
            continue

        (
            gt_expanded,
            gt_canonical,
            super_atom_map,
            gt_expansion_ok,
            gt_ok,
            gt_missing,
        ) = normalize_smiles(gt_raw, {}, ignore_cistrans)
        detail["gt_smiles_expanded"] = gt_expanded
        detail["gt_smiles_canonical"] = gt_canonical
        detail["gt_missing_abbreviations"] = gt_missing
        detail["gt_abbreviation_expansion_success"] = gt_expansion_ok
        detail["gt_canonicalization_success"] = gt_ok
        missing_occurrences.extend(
            {"id": record_id, "source": "gt", **missing}
            for missing in gt_missing
        )

        if gt_missing:
            reason = "gt_unexpandable_abbreviation"
            gt_unexpandable_count += 1
            detail["included"] = False
            detail["filter_reasons"].append(reason)
            detail["filter_reason_details"].append(
                _filter_reason_description(reason)
            )
            detail["error"] = reason
            filter_counts[reason] += 1
            status_counts["filtered"] += 1
            details.append(detail)
            continue
        if not gt_expansion_ok:
            included_count += 1
            included_ids.add(record_id)
            gt_expansion_failed_count += 1
            status_counts["gt_abbreviation_expansion_failed"] += 1
            detail["error"] = "gt_abbreviation_expansion_failed"
            details.append(detail)
            continue
        if not gt_ok:
            reason = "gt_canonicalization_failed"
            gt_canonicalization_failed_count += 1
            detail["included"] = False
            detail["filter_reasons"].append(reason)
            detail["filter_reason_details"].append(
                _filter_reason_description(reason)
            )
            detail["error"] = reason
            filter_counts[reason] += 1
            status_counts["filtered"] += 1
            details.append(detail)
            continue

        included_count += 1
        included_ids.add(record_id)
        if pred_record is None:
            detail["error"] = "prediction_id_not_found"
            status_counts["prediction_missing"] += 1
            details.append(detail)
            continue

        (
            pred_expanded,
            pred_canonical,
            _,
            pred_expansion_ok,
            pred_ok,
            pred_missing,
        ) = normalize_smiles(
            detail["pred_smiles_raw"], super_atom_map, ignore_cistrans
        )
        detail["pred_smiles_expanded"] = pred_expanded
        detail["pred_smiles_canonical"] = pred_canonical
        detail["pred_missing_abbreviations"] = pred_missing
        detail["pred_abbreviation_expansion_success"] = pred_expansion_ok
        detail["pred_canonicalization_success"] = pred_ok
        missing_occurrences.extend(
            {"id": record_id, "source": "pred", **missing}
            for missing in pred_missing
        )

        pred_raw = detail["pred_smiles_raw"]
        if (
            not pred_expansion_ok
            and isinstance(pred_raw, str)
            and pred_raw.strip()
        ):
            pred_expansion_failed_count += 1
            status_counts["pred_abbreviation_expansion_failed"] += 1
            detail["error"] = "pred_abbreviation_expansion_failed"
        elif not pred_ok:
            pred_canonicalization_failed_count += 1
            status_counts["pred_canonicalization_failed"] += 1
            detail["error"] = (
                "prediction_missing_or_canonicalization_failed"
            )
        elif gt_canonical == pred_canonical:
            detail["correct"] = True
            correct_count += 1
            status_counts["correct"] += 1
        else:
            detail["error"] = "canonical_smiles_mismatch"
            status_counts["incorrect"] += 1
        details.append(detail)

    total_gt = len(gt_records)
    missing_occurrences = [
        occurrence
        for occurrence in missing_occurrences
        if occurrence["id"] in included_ids
    ]
    missing_types = {
        occurrence["abbr"] for occurrence in missing_occurrences
    }
    missing_sample_ids = {
        occurrence["id"] for occurrence in missing_occurrences
    }
    summary: dict[str, Any] = {
        "gt_path": str(gt_path),
        "prediction_path": str(pred_path),
        "total_gt_records": total_gt,
        "prediction_records": len(pred_records),
        "included_records": included_count,
        "filtered_records": total_gt - included_count,
        "gt_conversion_success_records": converted_gt_count,
        "gt_abbreviation_expansion_failed_records": (
            gt_expansion_failed_count
        ),
        "gt_unexpandable_abbreviation_records": gt_unexpandable_count,
        "pred_abbreviation_expansion_failed_records": (
            pred_expansion_failed_count
        ),
        "abbreviation_expansion_failed_records": (
            gt_expansion_failed_count + pred_expansion_failed_count
        ),
        "gt_canonicalization_failed_records": (
            gt_canonicalization_failed_count
        ),
        "pred_canonicalization_failed_records": (
            pred_canonicalization_failed_count
        ),
        "missing_abbreviation_types": len(missing_types),
        "missing_abbreviation_occurrences": len(missing_occurrences),
        "missing_abbreviation_sample_records": len(missing_sample_ids),
        "missing_abbreviation_scope": "included_records_only",
        "missing_abbreviations_path": (
            str(missing_abbreviations_path)
            if missing_abbreviations_path is not None
            else None
        ),
        "correct_records": correct_count,
        "accuracy_on_included_records": (
            correct_count / included_count if included_count else 0.0
        ),
        "accuracy_over_all_gt_records": (
            correct_count / total_gt if total_gt else 0.0
        ),
        "ignore_cistrans": ignore_cistrans,
        "filtered_metadata_fields": list(metadata_fields),
        "filter_reason_counts": dict(sorted(filter_counts.items())),
        "filter_reason_descriptions": dict(
            sorted(FILTER_REASON_DESCRIPTIONS.items())
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "gt_read_errors": gt_read_errors,
        "prediction_read_errors": pred_read_errors,
        "index_errors": gt_index_errors + pred_index_errors,
        "coverage_errors": coverage_errors,
    }
    subset_names = {
        "A": "A",
        "B": "B",
        "C": "C",
    }
    subset_metrics: dict[str, dict[str, Any]] = {
        "Full": {
            "total_gt_records": total_gt,
            "scored_records": included_count,
            "correct_records": correct_count,
            "accuracy": (
                correct_count / included_count if included_count else 0.0
            ),
        }
    }
    for display_name, subset_name in subset_names.items():
        subset_details = [
            detail
            for detail in details
            if subset_by_id.get(detail["id"]) == subset_name
        ]
        scored = sum(bool(detail["included"]) for detail in subset_details)
        correct = sum(
            bool(detail["included"] and detail["correct"])
            for detail in subset_details
        )
        subset_metrics[display_name] = {
            "subset": subset_name,
            "total_gt_records": len(subset_details),
            "scored_records": scored,
            "correct_records": correct,
            "accuracy": correct / scored if scored else 0.0,
        }
    summary["subset_metrics"] = subset_metrics

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for detail in details:
                handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
        stem = output_path.stem
        suffix = output_path.suffix or ".jsonl"
        correct_path = output_path.with_name(f"{stem}_correct{suffix}")
        incorrect_path = output_path.with_name(
            f"{stem}_incorrect{suffix}"
        )
        row_groups = (
            (
                correct_path,
                [
                    row
                    for row in details
                    if row["included"] and row["correct"]
                ],
            ),
            (
                incorrect_path,
                [
                    row
                    for row in details
                    if row["included"] and not row["correct"]
                ],
            ),
        )
        for path, rows in row_groups:
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["result_path"] = str(output_path)
        summary["correct_path"] = str(correct_path)
        summary["incorrect_path"] = str(incorrect_path)

    if output_csv_path is not None:
        _write_details_csv(Path(output_csv_path), details)
        summary["result_csv_path"] = str(output_csv_path)
    if missing_abbreviations_path is not None:
        _write_missing_abbreviations_csv(
            Path(missing_abbreviations_path), missing_occurrences
        )
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return summary
