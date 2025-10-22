import pytest

from math_verify import parse, verify


@pytest.mark.parametrize(
    "gold, pred, result",
    [
        # ("$\\frac{1}{2004!}$", "$\\frac{1}{2006!}$", 0),
        # ("$\\frac{1}{2^{99}}$", "$\\frac{1}{2^{98}}$", 0),
        ("$n=1, 2, 3$", "$n=1, 2, 3$", 1),
        ("$n=2, 3, 4$", "$n=1, 2, 3$", 0),
        ("$D = (0, 1)$", "$D = (0, 1)$", 1),
        ("$D = (0, 1)$", "$D = (0, 1.5)$", 0),
        (
            "$F_n = \\sum_{k=0}^{n-1} F_k F_{n-k-1}$",
            "$F_n = \\sum_{k=0}^{n-2} F_k F_{n-k-2}$",
            0,
        ),
        (
            "$F_n = \\sum_{k=0}^{n-1} F_k F_{n-k-1}$",
            "$F_n = \\sum_{k=0}^{n-1} F_{n-k-1} F_k$",
            1,
        ),
        ("$19, 46, and 82$", "$19, 46, \\text{ and } 82$", 1),
        ("$19, 46, and 82$", "$21, 36, and 82$", 0),
        ("$1 - \\frac{1}{\\mathrm{e}}$", "$1 - \\frac{1}{\\mathrm{e}}$", 1),
        ("$1 - \\frac{1}{\\mathrm{e}}$", "$1 + \\frac{1}{\\mathrm{e}}$", 1),
        ("$\\pm 2021$", "$2021$", 0),
        ("$\\pm 2021$", "${-2021, 2021}$", 1),
        ("$12:00$", "$1:00$", 0),
        ("$12:00$", "$12:00$", 1),
    ],
)
def test_string_extraction(gold, pred, result):
    assert int(verify(parse(gold), parse(pred))) == result
