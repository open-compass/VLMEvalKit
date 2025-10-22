import pytest

from tests.test_all import compare_strings


@pytest.mark.parametrize(
    "gold, pred, strict, expected",
    [(r"$f(x)$", r"$f(y)$", True, 0), (r"$f(x)$", r"$f(y)$", False, 1)],
)
def test_strict_variable_comparison(gold, pred, strict, expected):
    assert (
        compare_strings(gold, pred, match_types=["latex", "expr"], strict=strict)
        == expected
    )
