import pytest
from latex2sympy2_extended import latex2sympy
from tests.context import assert_equal


def pytest_generate_tests(metafunc):
    metafunc.parametrize('s', metafunc.cls.BAD_STRINGS)


class TestAllBad(object):
    # These bad latex strings should raise an exception when parsed
    BAD_STRINGS = [
        "(",
        ")",
        # "a / b /",
        "\\frac{d}{dx}",
        "(\\frac{d}{dx})"
        "\\sqrt{}",
        "\\sqrt",
        "{",
        "}",
        # "1.1.1",
        "\\mathit{TEST}"
        "\\frac{2}{}",
        "\\frac{}{2}",
        "\\int",
        # "1 +",
        # "a +",
        "!",
        "!0",
        "_",
        "^",
        # "a // b",
        # "a \\cdot \\cdot b",
        # "a \\div \\div b",
        "a\\mod \\begin{matrix}b\\end{matrix}"
        "|",
        "||x|",
        "\\lfloor x",
        "\\lfloor a \\rceil",
        "\\operatorname{floor}(12.3, 123.4)",
        "()",
        "((((((((((((((((()))))))))))))))))",
        "-",
        "\\frac{d}{dx} + \\frac{d}{dt}",
        # "f()",
        # "f(,",
        # "f(x,,y)",
        # "f(x,y,",
        "\\sin^x",
        "\\cos^2",
        # "\\cos 1 \\cos",
        # "\\gcd(3)",
        # "\\lcm(2)",
        "@", "#", "$", "%", "&", "*",
        "\\",
        "~",
        "\\frac{(2 + x}{1 - x)}",
        # percentages without numbers before-hand
        "a\\%",
        "\\%100",
        # dollar signs without numbers after
        "\\$"
    ]

    def test_bad_string(self, s):
        with pytest.raises(Exception):
            latex2sympy(s)
            pass
