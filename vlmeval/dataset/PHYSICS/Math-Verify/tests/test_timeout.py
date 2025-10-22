import time
from unittest.mock import patch

from math_verify.grader import verify
from math_verify.parser import parse


@patch("math_verify.parser.parse_expr")
def test_timeout_expr(mock_parse_expr):
    # Mock the parsing function to simulate a delay
    def delayed_parse(*args, **kwargs):
        time.sleep(5)  # Simulate a delay longer than the timeout
        return "parsed_expr"

    mock_parse_expr.side_effect = delayed_parse

    # Test that the timeout is triggered
    x = parse(
        "1+1",
        parsing_timeout=1,
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )
    assert x == []


@patch("math_verify.parser.latex2sympy")
def test_timeout_latex(mock_parse_latex):
    # Mock the parsing function to simulate a delay
    def delayed_parse(*args, **kwargs):
        time.sleep(5)  # Simulate a delay longer than the timeout
        return "parsed_expr"

    mock_parse_latex.side_effect = delayed_parse

    # Test that the timeout is triggered
    x = parse(
        "$1+1$",
        parsing_timeout=1,
        extraction_mode="first_match",
        fallback_mode="no_fallback",
    )
    assert x == []


@patch("math_verify.grader.sympy_expr_eq")
def test_timeout_verify(mock_verify):
    # Mock the verify function to simulate a delay
    def delayed_sympy_expr_eq(*args, **kwargs):
        time.sleep(5)  # Simulate a delay longer than the timeout
        return True

    mock_verify.side_effect = delayed_sympy_expr_eq

    gold = [parse("1+1")[0]]
    assert not verify(gold, gold, timeout_seconds=1)
