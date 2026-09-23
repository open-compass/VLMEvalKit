"""Dependency-free public result types for MolRecBench-Wild."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregate and per-sample results returned by the official scorer."""

    summary: dict[str, Any]
    details: list[dict[str, Any]]

    def __iter__(self):
        """Allow ``summary, details = score_records(...)`` unpacking."""

        yield self.summary
        yield self.details
