from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class KaTeXNode:
    """
    Minimal KaTeX-like AST node (compatible with the original JS renderer).

    The renderer expects:
      - node.type: str
      - node.value: varies by node.type
    """

    type: str
    value: Any = None


@dataclass
class SupSubValue:
    base: KaTeXNode
    sup: Optional[KaTeXNode] = None
    sub: Optional[KaTeXNode] = None


@dataclass
class GenFracValue:
    hasBarLine: bool
    numer: KaTeXNode
    denom: KaTeXNode


@dataclass
class SqrtValue:
    body: KaTeXNode
    index: Optional[KaTeXNode] = None


@dataclass
class AccentValue:
    accent: str
    base: KaTeXNode


@dataclass
class FontValue:
    font: str
    body: KaTeXNode


@dataclass
class TextValue:
    body: list[KaTeXNode]


@dataclass
class PhantomValue:
    value: list[KaTeXNode]


@dataclass
class StylingValue:
    original: str
    value: list[KaTeXNode]


@dataclass
class SizingValue:
    original: str
    value: list[KaTeXNode]


@dataclass
class RuleDim:
    number: float
    unit: str


@dataclass
class RuleValue:
    width: RuleDim
    height: RuleDim


@dataclass
class LapValue:
    body: KaTeXNode


@dataclass
class OverUnderValue:
    body: KaTeXNode


@dataclass
class ColAlign:
    align: str
    type: str = "align"


@dataclass
class ColSeparator:
    separator: str
    type: str = "separator"


@dataclass
class ArrayValue:
    style: str
    cols: Optional[list[ColAlign | ColSeparator]]
    body: list[list[KaTeXNode]]  # rows -> cells (each cell is usually an ordgroup)


@dataclass
class LeftRightValue:
    left: str
    right: str
    body: list[KaTeXNode]


@dataclass
class OpValue:
    symbol: bool
    limits: bool
    body: str
