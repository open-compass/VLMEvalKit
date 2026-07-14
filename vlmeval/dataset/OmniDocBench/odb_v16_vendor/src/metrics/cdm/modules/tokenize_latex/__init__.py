"""
Pure-Python LaTeX tokenizer/normalizer used by CDM.

Public API:
  - tokenize_latex(): main entrypoint used by the rest of the project.
"""

from .tokenize_latex import tokenize_latex

__all__ = ["tokenize_latex"]

