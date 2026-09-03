"""Exact-match evaluators for Carbon-format molecular annotations."""

from __future__ import annotations
import argparse
import copy
import json
import math
import re
import signal
import threading
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable, Mapping, Sequence

from networkx.algorithms import isomorphism

from ._types import EvaluationResult
from .mol_graph import MolGraph
from .smiles_metric import ANNOTATION_METADATA_FIELDS, DEFAULT_GT
from .smiles_metric import DEFAULT_PRED as DEFAULT_SMILES_PRED
from .smiles_metric import evaluate as run_smiles_evaluation
from .smiles_metric import prediction_coverage_errors
from .utils import (Convert_Rx_to_R, canonicalize_smiles_w_superatom, check_R_atom,
                    compare_brackets, is_special_R, iter_special_R_substitution_mappings,
                    normalize_greek_letters, simplify_R_group_in_symbols, special_R_stem)

DEFAULT_GRAPH_PRED = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "MLLM"
    / "GPT-5.6-sol"
    / "GPT-5.6-sol_graph.jsonl"
)
DEFAULT_S_GRAPH_PRED = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "MLLM"
    / "GPT-5.6-sol"
    / "GPT-5.6-sol_graph_simple.jsonl"
)


class TimeoutException(Exception):
    """Raised when one graph-isomorphism comparison takes too long."""


@contextmanager
def time_limit(seconds: int | float | None):
    """Limit a comparison on platforms that provide SIGALRM."""

    if (
        not seconds
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
        or threading.current_thread() is not threading.main_thread()
    ):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def signal_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutException()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


class Evaluator:
    """Evaluate Graph and S-Graph exact matches for Carbon records."""

    def __init__(
        self,
        gt_list: Iterable[dict[str, Any]],
        pred_list: Iterable[dict[str, Any]],
        infer_version: str = "v2",
        print_error: bool = False,
        timeout_seconds: int | float | None = 5,
    ) -> None:
        gt_records = list(gt_list)
        pred_records = list(pred_list)
        self.ids: list[str] = []
        self.eval_info: dict[str, dict[str, Any]] = {}
        self.mol_graph_gts: dict[str, MolGraph] = {}
        self.mol_graph_preds: dict[str, MolGraph] = {}
        self.gt_success_count = 0
        self.pred_success_count = 0
        self.infer_version = infer_version
        self.total_gt_count = len(gt_records)
        self.print_error = print_error
        self.timeout_seconds = timeout_seconds
        self.attribute: dict[str, dict[str, list[int]]] = {
            "smiles": {},
            "simplified_graph": {},
            "graph": {},
        }
        self.index_errors: list[str] = []

        pred_by_id: dict[str, dict[str, Any]] = {}
        for index, record in enumerate(pred_records, start=1):
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                self.index_errors.append(
                    f"prediction row {index}: missing string id"
                )
            elif record_id in pred_by_id:
                self.index_errors.append(
                    f"prediction: duplicate id {record_id!r}"
                )
            else:
                pred_by_id[record_id] = record

        seen_gt_ids: set[str] = set()
        for index, record in enumerate(gt_records, start=1):
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                record_id = f"__invalid_gt_row_{index}"
                self.index_errors.append(
                    f"GT row {index}: missing string id"
                )
            elif record_id in seen_gt_ids:
                raise ValueError(f"GT: duplicate id {record_id!r}")
            seen_gt_ids.add(record_id)
            self.ids.append(record_id)
            self.eval_info[record_id] = {
                "id": record_id,
                "hardcase_label": record.get("hardcase_label"),
                "gt_load_success": False,
                "pred_load_success": False,
            }

            try:
                gt_graph = MolGraph(
                    id=record_id,
                    carbon_info=record,
                    attribute=record.get("hardcase_label"),
                )
                self.gt_success_count += 1
                self.eval_info[record_id]["gt_load_success"] = True
            except Exception as exc:
                gt_graph = MolGraph(id=record_id)
                self.eval_info[record_id]["gt_load_error"] = str(exc)
                if self.print_error:
                    print(f"GT {record_id!r} load failed: {exc}")
            self.mol_graph_gts[record_id] = gt_graph

            pred_record = pred_by_id.get(record_id)
            if pred_record is None:
                pred_graph = MolGraph(id=record_id)
                self.eval_info[record_id][
                    "pred_load_error"
                ] = "prediction_id_not_found"
            else:
                try:
                    pred_graph = MolGraph(
                        id=record_id, carbon_info=pred_record
                    )
                    self.pred_success_count += 1
                    self.eval_info[record_id]["pred_load_success"] = True
                except Exception as exc:
                    pred_graph = MolGraph(id=record_id)
                    self.eval_info[record_id]["pred_load_error"] = str(exc)
                    if self.print_error:
                        print(
                            f"Prediction {record_id!r} load failed: {exc}"
                        )
            self.mol_graph_preds[record_id] = pred_graph

    def evaluate_simplified_graph(self) -> tuple[int, int]:
        node_match = isomorphism.categorical_node_match("symbol", None)
        edge_match = isomorphism.categorical_edge_match("bond", None)
        correct_count = 0
        success_count = 0
        for record_id in self.ids:
            gt_graph = self.mol_graph_gts[record_id]
            pred_graph = self.mol_graph_preds[record_id]
            info = self.eval_info[record_id]
            info["simplified_graph_eval"] = False
            info["simplified_graph_gt"] = gt_graph.dump_to_carbon(
                simplify=True
            )
            info["simplified_graph_pred"] = pred_graph.dump_to_carbon(
                simplify=True
            )
            if not gt_graph.symbols:
                continue
            success_count += 1
            if not pred_graph.symbols:
                continue
            graph_correct, _ = self._compare_graph(
                gt_graph,
                pred_graph,
                node_match,
                edge_match,
                simplify=True,
            )
            if graph_correct:
                correct_count += 1
                info["simplified_graph_eval"] = True
        return success_count, correct_count

    def evaluate_graph(self) -> tuple[int, int]:
        node_match = isomorphism.categorical_node_match(
            [
                "symbol",
                "charge",
                "radical",
                "valence",
                "isotope",
                "attach_point",
            ],
            [None, None, None, None, None, None],
        )
        edge_match = isomorphism.categorical_edge_match("bond", None)
        correct_count = 0
        success_count = 0
        for record_id in self.ids:
            gt_graph = self.mol_graph_gts[record_id]
            pred_graph = self.mol_graph_preds[record_id]
            info = self.eval_info[record_id]
            info["graph_eval"] = False
            info["graph_gt"] = gt_graph.dump_to_carbon()
            info["graph_pred"] = pred_graph.dump_to_carbon()
            if not gt_graph.symbols:
                continue
            success_count += 1
            if not pred_graph.symbols:
                continue
            graph_correct, mapping = self._compare_graph(
                gt_graph,
                pred_graph,
                node_match,
                edge_match,
                simplify=False,
            )
            correct = graph_correct and compare_brackets(
                gt_graph.brackets, pred_graph.brackets, mapping
            )
            if correct:
                correct_count += 1
                info["graph_eval"] = True
            attributes = gt_graph.attribute
            if attributes:
                for attribute in attributes:
                    self.attribute["graph"].setdefault(
                        str(attribute), []
                    ).append(int(bool(correct)))
        return success_count, correct_count

    @staticmethod
    def _simplify_R_group_in_smiles(smiles: str) -> str:
        """Apply special-R simplification for canonical SMILES comparison."""

        pattern = re.compile(r"\[([^\[\]]+)\]")
        symbols = [
            match.group(1)
            for match in pattern.finditer(smiles)
            if is_special_R(match.group(1))
        ]
        counts = Counter(symbols)

        def replace(match: re.Match[str]) -> str:
            symbol = match.group(1)
            if not is_special_R(symbol) or counts[symbol] != 1:
                return match.group(0)
            stem = special_R_stem(symbol)
            return f"[{stem}**]"

        return pattern.sub(replace, smiles)

    @staticmethod
    def _canonical_smiles_pair(
        smiles_gt: str,
        smiles_pred: str,
        *,
        kekule: bool,
        ignore_cistrans: bool,
    ) -> tuple[bool, str, str]:
        gt_canonical, super_atom_map, gt_ok = (
            canonicalize_smiles_w_superatom(
                smiles_gt,
                super_atom_map={},
                recover_super_atom=True,
                ignore_cistrans=ignore_cistrans,
                kekule=kekule,
            )
        )
        pred_canonical, _, pred_ok = canonicalize_smiles_w_superatom(
            smiles_pred,
            super_atom_map=super_atom_map,
            recover_super_atom=True,
            ignore_cistrans=ignore_cistrans,
            kekule=kekule,
        )
        return (
            bool(gt_ok and pred_ok and gt_canonical == pred_canonical),
            gt_canonical,
            pred_canonical,
        )

    @classmethod
    def _eval_smiles_impl(
        cls,
        smiles_gt: str,
        smiles_pred: str,
        *,
        kekule: bool,
        ignore_cistrans: bool = True,
    ) -> tuple[bool, str, str]:
        """Compare SMILES using special-R substitution matching."""

        gt_simplified = cls._simplify_R_group_in_smiles(smiles_gt)
        pred_simplified = cls._simplify_R_group_in_smiles(smiles_pred)
        matched, gt_canonical, pred_canonical = cls._canonical_smiles_pair(
            gt_simplified,
            pred_simplified,
            kekule=kekule,
            ignore_cistrans=ignore_cistrans,
        )
        if matched:
            return matched, gt_canonical, pred_canonical

        gt_symbols = [
            match.group(1)
            for match in re.finditer(r"\[([^\[\]]+)\]", gt_simplified)
            if is_special_R(match.group(1))
        ]
        pred_symbols = [
            match.group(1)
            for match in re.finditer(r"\[([^\[\]]+)\]", pred_simplified)
            if is_special_R(match.group(1))
        ]
        for mapping in iter_special_R_substitution_mappings(
            gt_symbols, pred_symbols
        ):
            mapped_pred = re.sub(
                r"\[([^\[\]]+)\]",
                lambda match: f"[{mapping.get(match.group(1), match.group(1))}]",
                pred_simplified,
            )
            matched, gt_canonical, pred_canonical = (
                cls._canonical_smiles_pair(
                    gt_simplified,
                    mapped_pred,
                    kekule=kekule,
                    ignore_cistrans=ignore_cistrans,
                )
            )
            if matched:
                return matched, gt_canonical, pred_canonical
        return False, gt_canonical, pred_canonical

    def evaluate_smiles(
        self,
        expand: bool = False,
        kekule: bool = False,
        ignore_cistrans: bool = True,
    ) -> tuple[int, int]:
        """Evaluate exact matches after Carbon-to-SMILES normalization."""

        correct_count = 0
        success_count = 0
        for record_id in self.ids:
            info = self.eval_info[record_id]
            info.update(
                {
                    "smiles_eval": False,
                    "smiles_gt": None,
                    "smiles_pred": None,
                }
            )
            try:
                smiles_gt, super_atom_map, _ = self.mol_graph_gts[
                    record_id
                ].dump_to_SMILES(super_atom_map={}, expand=expand)
                if not smiles_gt.strip():
                    continue
                info["smiles_gt"] = smiles_gt
                success_count += 1
            except Exception as exc:
                info["smiles_gt_error"] = str(exc)
                continue
            try:
                smiles_pred, _, _ = self.mol_graph_preds[
                    record_id
                ].dump_to_SMILES(
                    super_atom_map=super_atom_map, expand=expand
                )
                info["smiles_pred"] = smiles_pred
            except Exception as exc:
                info["smiles_pred_error"] = str(exc)
                continue
            correct, gt_canonical, pred_canonical = self._eval_smiles_impl(
                smiles_gt,
                smiles_pred,
                kekule=kekule,
                ignore_cistrans=ignore_cistrans,
            )
            info["smiles_gt_canonical"] = gt_canonical
            info["smiles_pred_canonical"] = pred_canonical
            if correct:
                correct_count += 1
                info["smiles_eval"] = True
        return success_count, correct_count

    def _compare_graph(
        self,
        mol_graph_gt: MolGraph,
        mol_graph_pred: MolGraph,
        node_match: Any,
        edge_match: Any,
        simplify: bool,
    ) -> tuple[bool, dict[int, int] | None]:
        try:
            with time_limit(self.timeout_seconds):
                return self._compare_graph_impl(
                    copy.deepcopy(mol_graph_gt),
                    copy.deepcopy(mol_graph_pred),
                    node_match,
                    edge_match,
                    simplify,
                )
        except TimeoutException:
            return False, None
        except (IndexError, KeyError, TypeError, ValueError):
            return False, None

    @staticmethod
    def _matcher(
        mol_graph_gt: MolGraph,
        mol_graph_pred: MolGraph,
        node_match: Any,
        edge_match: Any,
        simplify: bool,
    ) -> isomorphism.DiGraphMatcher:
        graph_gt = (
            mol_graph_gt.dump_to_simplify_graph()
            if simplify
            else mol_graph_gt.dump_to_graph()
        )
        graph_pred = (
            mol_graph_pred.dump_to_simplify_graph()
            if simplify
            else mol_graph_pred.dump_to_graph()
        )
        return isomorphism.DiGraphMatcher(
            graph_gt,
            graph_pred,
            node_match=node_match,
            edge_match=edge_match,
        )

    def _compare_graph_impl(
        self,
        mol_graph_gt: MolGraph,
        mol_graph_pred: MolGraph,
        node_match: Any,
        edge_match: Any,
        simplify: bool,
    ) -> tuple[bool, dict[int, int] | None]:
        if check_R_atom(mol_graph_gt.symbols):
            mol_graph_gt.symbols = simplify_R_group_in_symbols(
                mol_graph_gt.symbols
            )
            mol_graph_pred.symbols = simplify_R_group_in_symbols(
                mol_graph_pred.symbols
            )
            gt_copy = copy.deepcopy(mol_graph_gt)
            pred_copy = copy.deepcopy(mol_graph_pred)
            gt_copy.symbols = Convert_Rx_to_R(gt_copy.symbols)
            pred_copy.symbols = Convert_Rx_to_R(pred_copy.symbols)
            matcher = self._matcher(
                gt_copy,
                pred_copy,
                node_match,
                edge_match,
                simplify,
            )
            if matcher.is_isomorphic():
                mol_graph_gt.symbols = normalize_greek_letters(
                    mol_graph_gt.symbols
                )
                mol_graph_pred.symbols = normalize_greek_letters(
                    mol_graph_pred.symbols
                )
                gt_special = {
                    symbol
                    for symbol in mol_graph_gt.symbols
                    if is_special_R(symbol)
                }
                pred_special = {
                    symbol
                    for symbol in mol_graph_pred.symbols
                    if is_special_R(symbol)
                }
                if len(gt_special) != len(pred_special):
                    return False, None
                for symbol_mapping in (
                    iter_special_R_substitution_mappings(
                        mol_graph_gt.symbols, mol_graph_pred.symbols
                    )
                ):
                    mapped_pred = copy.deepcopy(mol_graph_pred)
                    mapped_pred.symbols = [
                        symbol_mapping.get(symbol, symbol)
                        for symbol in mapped_pred.symbols
                    ]
                    matcher = self._matcher(
                        mol_graph_gt,
                        mapped_pred,
                        node_match,
                        edge_match,
                        simplify,
                    )
                    if matcher.is_isomorphic():
                        return True, dict(matcher.mapping)
        else:
            matcher = self._matcher(
                mol_graph_gt,
                mol_graph_pred,
                node_match,
                edge_match,
                simplify,
            )
            if matcher.is_isomorphic():
                return True, dict(matcher.mapping)
        return False, None

    def save_eval_info(self, save_path: str | Path) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.eval_info, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def save_attribute_result(self, save_path: str | Path) -> None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.attribute, handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and require every non-empty row to be an object."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: record is not an object"
                )
            records.append(value)
    return records


def read_split_ids(path: Path) -> set[str]:
    """Read non-empty sample IDs from a text file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def run_graph_evaluation(
    *,
    gt_path: Path,
    pred_path: Path,
    metric: str,
    split_path: Path | None = None,
    output_path: Path | None = None,
    summary_path: Path | None = None,
    limit: int | None = None,
    timeout_seconds: float | None = 5,
    allow_coverage_mismatch: bool = False,
) -> dict[str, Any]:
    """Run Graph or simplified-Graph exact-match evaluation."""

    gt_records = read_jsonl(Path(gt_path))
    full_gt_records = gt_records
    pred_records = read_jsonl(Path(pred_path))
    if split_path is not None:
        split_ids = read_split_ids(Path(split_path))
        full_gt_ids = {
            record.get("id")
            for record in full_gt_records
            if isinstance(record.get("id"), str) and record.get("id")
        }
        unknown_split_ids = sorted(split_ids - full_gt_ids)
        if unknown_split_ids:
            raise ValueError(
                f"split contains {len(unknown_split_ids)} IDs outside the full GT "
                f"(examples: {', '.join(repr(value) for value in unknown_split_ids[:3])})"
            )
        gt_records = [
            record
            for record in gt_records
            if record.get("id") in split_ids
        ]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        gt_records = gt_records[:limit]
    coverage_errors = prediction_coverage_errors(
        full_gt_records,
        gt_records,
        pred_records,
    )
    if coverage_errors and not allow_coverage_mismatch:
        raise ValueError(
            "Prediction coverage validation failed: "
            + "; ".join(coverage_errors)
        )

    evaluator = Evaluator(
        gt_records,
        pred_records,
        timeout_seconds=timeout_seconds,
    )
    if metric == "graph":
        valid, correct = evaluator.evaluate_graph()
        result_key = "graph_eval"
        metric_label = "Graph"
    elif metric == "s_graph":
        valid, correct = evaluator.evaluate_simplified_graph()
        result_key = "simplified_graph_eval"
        metric_label = "S-Graph"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    summary: dict[str, Any] = {
        "metric": metric_label,
        "gt_path": str(gt_path),
        "prediction_path": str(pred_path),
        "split_path": str(split_path) if split_path is not None else None,
        "total_gt_records": len(gt_records),
        "prediction_records": len(pred_records),
        "valid_gt_records": valid,
        "correct_records": correct,
        "accuracy_on_valid_gt_records": (
            correct / valid if valid else 0.0
        ),
        "accuracy_over_all_gt_records": (
            correct / len(gt_records) if gt_records else 0.0
        ),
        "gt_load_success_records": evaluator.gt_success_count,
        "prediction_load_success_records": evaluator.pred_success_count,
        "index_errors": evaluator.index_errors,
        "coverage_errors": coverage_errors,
    }
    subset_names = {
        "A": "A",
        "B": "B",
        "C": "C",
    }
    subset_metrics: dict[str, dict[str, Any]] = {
        "Full": {
            "total_gt_records": len(gt_records),
            "scored_records": valid,
            "correct_records": correct,
            "accuracy": correct / valid if valid else 0.0,
        }
    }
    for display_name, subset_name in subset_names.items():
        subset_records = [
            record
            for record in gt_records
            if record.get("evaluation_subset") == subset_name
        ]
        scored = sum(
            bool(evaluator.mol_graph_gts[record["id"]].symbols)
            for record in subset_records
        )
        subset_correct = sum(
            bool(evaluator.eval_info[record["id"]].get(result_key, False))
            for record in subset_records
        )
        subset_metrics[display_name] = {
            "subset": subset_name,
            "total_gt_records": len(subset_records),
            "scored_records": scored,
            "correct_records": subset_correct,
            "accuracy": subset_correct / scored if scored else 0.0,
        }
    summary["subset_metrics"] = subset_metrics

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for record_id in evaluator.ids:
                info = evaluator.eval_info[record_id]
                row = {
                    "id": record_id,
                    "correct": bool(info.get(result_key, False)),
                    "gt_load_success": info.get(
                        "gt_load_success", False
                    ),
                    "pred_load_success": info.get(
                        "pred_load_success", False
                    ),
                    "error": info.get("pred_load_error")
                    or info.get("gt_load_error"),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["result_path"] = str(output_path)

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return summary


def _materialize_records(
    records: Iterable[Mapping[str, Any]], source_name: str
) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for row_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"{source_name} row {row_number}: record is not an object"
            )
        materialized.append(dict(record))
    return materialized


def _write_records_jsonl(
    path: Path, records: Iterable[Mapping[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_details_jsonl(path: Path) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"evaluation detail row {line_number} is not an object"
                )
            details.append(value)
    return details


def _validated_timeout(value: int | float | None) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a non-negative number or None")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("timeout_seconds must be non-negative")
    return value


def score_records(
    gt_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    track: str,
    *,
    full_gt_records: Iterable[Mapping[str, Any]] | None = None,
    timeout_seconds: int | float | None = 5,
    ignore_cistrans: bool = True,
) -> EvaluationResult:
    """Score in-memory records with the fixed upstream evaluator.

    ``gt_records`` is the selected evaluation set.  ``full_gt_records`` may be
    supplied for a limited run so that a full prediction file is accepted
    while IDs outside the complete benchmark remain fatal.  Graph and S-Graph
    use a five-second per-sample timeout by default; ``0`` or ``None`` disables
    it exactly as in the upstream evaluator.
    """

    from .converter import normalize_track

    canonical_track = normalize_track(track)
    evaluated_gt = _materialize_records(gt_records, "evaluated GT")
    full_gt = (
        evaluated_gt
        if full_gt_records is None
        else _materialize_records(full_gt_records, "full GT")
    )
    predictions = _materialize_records(prediction_records, "prediction")

    coverage_errors = prediction_coverage_errors(
        full_gt, evaluated_gt, predictions
    )
    if coverage_errors:
        raise ValueError(
            "Prediction coverage validation failed: "
            + "; ".join(coverage_errors)
        )

    selected_ids = {record["id"] for record in evaluated_gt}
    selected_predictions = [
        record for record in predictions if record.get("id") in selected_ids
    ]
    timeout = _validated_timeout(timeout_seconds)

    with TemporaryDirectory(prefix="vlmeval-molrecbench-wild-") as temp_dir:
        temp_root = Path(temp_dir)
        gt_path = temp_root / "annotation.jsonl"
        prediction_path = temp_root / "prediction.jsonl"
        details_path = temp_root / "details.jsonl"
        _write_records_jsonl(gt_path, evaluated_gt)
        _write_records_jsonl(prediction_path, selected_predictions)

        if canonical_track == "smiles":
            summary = run_smiles_evaluation(
                gt_path=gt_path,
                pred_path=prediction_path,
                output_path=details_path,
                metadata_fields=ANNOTATION_METADATA_FIELDS,
                ignore_cistrans=ignore_cistrans,
            )
        else:
            summary = run_graph_evaluation(
                gt_path=gt_path,
                pred_path=prediction_path,
                metric=canonical_track,
                output_path=details_path,
                timeout_seconds=timeout,
            )
        details = _read_details_jsonl(details_path)

    gt_by_id = {record["id"]: record for record in evaluated_gt}
    for detail in details:
        gt_record = gt_by_id.get(detail.get("id"), {})
        detail.setdefault("evaluation_subset", gt_record.get("evaluation_subset"))
        detail.setdefault("hardcase_label", gt_record.get("hardcase_label"))

    # Temporary implementation paths are deliberately not exposed after the
    # directory is gone.  The dataset layer writes its own durable artifacts.
    for key in (
        "gt_path",
        "prediction_path",
        "result_path",
        "correct_path",
        "incorrect_path",
    ):
        summary.pop(key, None)
    summary["track"] = canonical_track
    summary["input_prediction_records"] = len(predictions)
    summary["evaluated_prediction_records"] = len(selected_predictions)
    summary["graph_timeout_seconds"] = (
        timeout if canonical_track in {"graph", "s_graph"} else None
    )
    summary["ignore_cistrans"] = (
        bool(ignore_cistrans) if canonical_track == "smiles" else None
    )
    return EvaluationResult(summary=summary, details=details)


evaluate_records = score_records


def _add_common_arguments(
    parser: argparse.ArgumentParser, *, default_prediction: Path
) -> None:
    parser.add_argument(
        "--gt", "--gt-path", "--gt_path", dest="gt", type=Path, default=DEFAULT_GT
    )
    parser.add_argument(
        "--pred",
        "--pred-path",
        "--pred_path",
        dest="pred",
        type=Path,
        default=default_prediction,
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-coverage-mismatch",
        action="store_true",
        help=(
            "Allow missing, extra, duplicate, or ID-less prediction rows. "
            "By default these coverage errors stop evaluation."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the unified SMILES, Graph, and S-Graph CLI parser."""

    parser = argparse.ArgumentParser(
        prog="python -m evaluate",
        description="Evaluate MolRecBench-Wild predictions."
    )
    subparsers = parser.add_subparsers(dest="metric", required=True)

    smiles_parser = subparsers.add_parser(
        "smiles", help="Evaluate direct SMILES predictions."
    )
    _add_common_arguments(
        smiles_parser, default_prediction=DEFAULT_SMILES_PRED
    )
    smiles_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optionally write per-record evaluation details as JSONL.",
    )
    smiles_parser.add_argument(
        "--output-csv",
        "--output_csv",
        dest="output_csv",
        type=Path,
        default=None,
        help="Optionally write per-record evaluation details as CSV.",
    )
    smiles_parser.add_argument(
        "--missing-abbreviations-output",
        "--miss_abbr_csv",
        dest="missing_abbreviations_output",
        type=Path,
        default=None,
    )
    cistrans_group = smiles_parser.add_mutually_exclusive_group()
    cistrans_group.add_argument(
        "--ignore-cistrans",
        dest="ignore_cistrans",
        action="store_true",
        default=True,
        help=(
            "Ignore double-bond cis/trans slash directions while still "
            "comparing atom chirality (benchmark default)."
        ),
    )
    cistrans_group.add_argument(
        "--preserve-cistrans",
        dest="ignore_cistrans",
        action="store_false",
        help=(
            "Compare double-bond cis/trans slash directions as well as "
            "atom chirality."
        ),
    )

    for metric, default_prediction, help_text in (
        ("graph", DEFAULT_GRAPH_PRED, "Evaluate full Graph predictions."),
        (
            "s_graph",
            DEFAULT_S_GRAPH_PRED,
            "Evaluate simplified Graph predictions.",
        ),
    ):
        graph_parser = subparsers.add_parser(metric, help=help_text)
        _add_common_arguments(
            graph_parser, default_prediction=default_prediction
        )
        graph_parser.add_argument(
            "--split",
            "--split-path",
            "--split_path",
            dest="split",
            type=Path,
            default=None,
            help="Optional text file containing one GT ID per line.",
        )
        graph_parser.add_argument("--output", type=Path, default=None)
        graph_parser.add_argument(
            "--timeout",
            type=float,
            default=5,
            help="Per-record isomorphism timeout; use 0 to disable.",
        )
    return parser


def _run_cli(args: argparse.Namespace) -> int:
    """Run an already parsed unified evaluation command."""

    if args.metric == "smiles":
        summary = run_smiles_evaluation(
            gt_path=args.gt,
            pred_path=args.pred,
            output_path=args.output,
            output_csv_path=args.output_csv,
            summary_path=args.summary,
            missing_abbreviations_path=args.missing_abbreviations_output,
            metadata_fields=ANNOTATION_METADATA_FIELDS,
            ignore_cistrans=args.ignore_cistrans,
            limit=args.limit,
            allow_coverage_mismatch=args.allow_coverage_mismatch,
        )
        print(f"评估样本数: {summary['included_records']}")
        print(f"正确样本数: {summary['correct_records']}")
        print(f"模型准确率: {summary['accuracy_on_included_records']:.4%}")
        print(
            "SMILES Precision: "
            f"{summary['accuracy_on_included_records']:.10f}"
        )
        return 0

    summary = run_graph_evaluation(
        gt_path=args.gt,
        pred_path=args.pred,
        metric=args.metric,
        split_path=args.split,
        output_path=args.output,
        summary_path=args.summary,
        limit=args.limit,
        timeout_seconds=args.timeout or None,
        allow_coverage_mismatch=args.allow_coverage_mismatch,
    )
    print(f"指标: {summary['metric']}")
    print(f"有效 GT: {summary['valid_gt_records']}")
    print(f"正确样本数: {summary['correct_records']}")
    print(f"准确率: {summary['accuracy_on_valid_gt_records']:.4%}")
    precision_label = (
        "Graph Precision"
        if args.metric == "graph"
        else "Simplified Graph Precision"
    )
    print(
        f"{precision_label}: "
        f"{summary['accuracy_on_valid_gt_records']:.10f}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the unified evaluation command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _run_cli(args)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
