"""Lazy public API for the MolRecBench-Wild official evaluator.

Importing this package never imports RDKit or NetworkX.  The chemistry stack is
loaded only when :func:`score_records` is called; prediction conversion remains
available in inference-only environments.
"""

from __future__ import annotations
from importlib.util import find_spec
from typing import Any, Iterable, Mapping

from ._types import EvaluationResult


class MolRecBenchDependencyError(ImportError):
    """The optional dependencies required for official scoring are absent."""


def conversion_dependencies_available() -> bool:
    """Return whether the DataFrame converter's dependency is installed."""

    return find_spec("pandas") is not None


def evaluation_dependencies_available() -> bool:
    """Return whether both official evaluator dependencies are installed."""

    return find_spec("networkx") is not None and find_spec("rdkit") is not None


def convert_prediction(
    index: Any,
    prediction: Any,
    track: str,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Lazily convert one prediction; see :mod:`.converter`."""

    from .converter import convert_prediction as implementation

    return implementation(index, prediction, track, strict=strict)


def convert_dataframe(
    dataframe: Any,
    *,
    sheet: str = "Sheet1",
    id_suffix: str = "",
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Lazily convert a prediction DataFrame; see :mod:`.converter`."""

    from .converter import convert_dataframe as implementation

    return implementation(
        dataframe,
        sheet=sheet,
        id_suffix=id_suffix,
    )


def score_records(
    gt_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    track: str,
    *,
    full_gt_records: Iterable[Mapping[str, Any]] | None = None,
    timeout_seconds: int | float | None = 5,
    ignore_cistrans: bool = True,
) -> EvaluationResult:
    """Lazily run official SMILES, S-Graph, or Graph record scoring."""

    try:
        from .evaluator import score_records as implementation
    except ModuleNotFoundError as error:
        missing = (error.name or "").split(".", 1)[0]
        if missing in {"networkx", "rdkit"}:
            raise MolRecBenchDependencyError(
                "MolRecBench-Wild scoring requires the optional 'networkx' "
                "and 'rdkit' packages. Install the molrecbench-wild extra."
            ) from error
        raise
    return implementation(
        gt_records,
        prediction_records,
        track,
        full_gt_records=full_gt_records,
        timeout_seconds=timeout_seconds,
        ignore_cistrans=ignore_cistrans,
    )


def evaluate_records(
    gt_records: Iterable[Mapping[str, Any]],
    prediction_records: Iterable[Mapping[str, Any]],
    track: str,
    *,
    full_gt_records: Iterable[Mapping[str, Any]] | None = None,
    timeout_seconds: int | float | None = 5,
    ignore_cistrans: bool = True,
) -> EvaluationResult:
    """Alias for :func:`score_records` with the same explicit contract."""

    return score_records(
        gt_records,
        prediction_records,
        track,
        full_gt_records=full_gt_records,
        timeout_seconds=timeout_seconds,
        ignore_cistrans=ignore_cistrans,
    )


def __getattr__(name: str):
    if name in {
        "ConversionError",
        "ConversionIssue",
        "ConversionSummary",
        "PredictionConversionError",
        "WorkbookConversionError",
    }:
        from . import converter

        return getattr(converter, name)
    raise AttributeError(name)


__all__ = [
    "ConversionError",
    "ConversionIssue",
    "ConversionSummary",
    "EvaluationResult",
    "MolRecBenchDependencyError",
    "PredictionConversionError",
    "WorkbookConversionError",
    "conversion_dependencies_available",
    "convert_dataframe",
    "convert_prediction",
    "evaluate_records",
    "evaluation_dependencies_available",
    "score_records",
]
