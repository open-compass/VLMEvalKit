"""CDM (Character Detection Matching) integration.

This package vendors the CDM implementation used for equation evaluation.
"""

from .cdm import cdm, cdm_metrics, calc

__all__ = [
    "cdm",
    "cdm_metrics",
    "calc",
]
