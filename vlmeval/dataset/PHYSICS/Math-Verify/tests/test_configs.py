import sympy
from latex2sympy2_extended import NormalizationConfig

from math_verify.parser import LatexExtractionConfig, parse


def test_boxed_match_priority():
    x = parse(
        "\\boxed{1}",
        (LatexExtractionConfig(),),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )
    assert len(x) == 1

    # No extraction
    x = parse(
        "\\boxed{1}",
        (LatexExtractionConfig(boxed_match_priority=-1),),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )
    assert len(x) == 0

    # final answer is match first
    x = parse(
        "final answer is $9999$, \\boxed{1}",
        (LatexExtractionConfig(boxed_match_priority=100),),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )
    assert x[0] == sympy.Integer(9999)


def test_normalization_config():
    x = parse(
        "$\\frac12$",
        (
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=False,
                    nits=True,
                    boxed=True,
                    equations=True,
                )
            ),
        ),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )
    assert len(x) == 0

    x = parse(
        "$\\frac12$",
        (
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    basic_latex=True,
                    units=True,
                    malformed_operators=True,
                    nits=True,
                    boxed=True,
                    equations=True,
                )
            ),
        ),
        fallback_mode="no_fallback",
        extraction_mode="first_match",
    )
    assert x[0] == sympy.Rational(1, 2)
