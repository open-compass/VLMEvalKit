# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest
import sympy

from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from math_verify.grader import sympy_expr_eq

"""
This file contains regression tests for testing evaluation of free-flow generation for math or indices.
Most of the tests have been created based on observations from the model outputs.
"""


def compare_strings(
    gold: str,
    pred: str,
    match_types: list[str] = ["latex", "expr"],
    precision: int = 6,
    strict: bool = True,
):
    """Helper function to compare strings using the math extraction metrics"""
    # Convert string match_types to ExtractionTarget objects
    extraction_targets = []
    for match_type in match_types:
        if match_type == "latex":
            extraction_targets.append(LatexExtractionConfig(boxed_match_priority=0))
        elif match_type == "expr":
            extraction_targets.append(ExprExtractionConfig())

    gold_parsed = parse(gold, extraction_targets)
    pred_parsed = parse(pred, extraction_targets)
    return verify(gold_parsed, pred_parsed, float_rounding=precision, strict=strict)


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Test negative numbers
        ("-5", "-5", 1),
        # Test for thousands separator
        ("7425000", "7,425,000", 1),
        ("1000", "1 000", 1),
        ("1000", "1000.0", 1),
        # Test thousand separator with floating point number
        ("1000.0", "1,000.0", 1),
        # Test decimal separator as ,
        ("1000.99", "1000,99", 1),
        ("1,22", "1.22", 1),
        ("2.74", "Soucis : 2,74 $ a..", 1),
        # Test no decimal separator
        ("0.4", ".4", 1),
        # Test decimals
        ("1000.99", "1,000.99", 1),
        ("1000.99", "1,000.99", 1),
        # Test with units like $
        ("1000.99", "$1,000.99", 1),
        ("1000.99", "1,000.99$", 1),
        # Test with currency units
        ("1000.99", "the number is not 10 which is 1,000.99€", 1),
        ("1000.99", "the number is not 10 which is 1,000.99€", 1),
        # Test m2
        ("1000.99", "so the number is 10 which is 1,000.99m²", 1),
        ("1000.99", "not it's not 10 it's 1,000.99m²", 1),
        ("0,111", "0.111", 1),
        # Test correct extraction of not correct answer
        ("2", "AZYUK2A", 0),
    ],
)
def test_number_extraction(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("10/9", "\\frac{10}{9}", 1),
        ("-10/9", "-\\frac{10}{9}", 1),
    ],
)
def test_simple_fraction_notation(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("$[0,1)$", "$[0,1)$", 1),
        ("$[0,1)$", "$[0,1)$", 1),
        ("$[0,9)$", "$[0,1)$", 0),
        ("$(0,9)$", "$[0,9)$", 0),
        ("$1$", "$-[0,1)$", 0),
    ],
)
def test_sets_handling(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Notations
        ("$9$", "Answer \\[ 9 \\]", 1),
        ("$9$", "Answer $ 9 $", 1),
        ("$9$", "Answer $$ 9 $$", 1),
        ("$9$", "Answer \\( 9 \\)", 1),
        # Works even with ()
        ("$10$", "Answer \\( (9+1) \\)", 1),
        # Separate line shouldn't work for inline latex
        ("$9$", "Answer $ \n 9 \n $", 0),
        ("$9$", "Answer \\( \n 9 \n \\)", 0),
        # Separate line should work for block latex
        ("$9$", "Answer \\[ \n 9 \n \\]", 1),
        ("$9$", "Answer $$ \n 9 \n $$", 1),
        # the $ can appear in the middle of the string
        ("$10/9$", "Answer $ \\frac{1}{2} \\$ = \\frac{10}{9} $", 1),
        # Malformed fractions work
        ("$1/3$", "$\\frac13 $", 1),
        ("$1$", "$\\frac3{3} $", 1),
        # Malformed sqrt works
        ("$\\sqrt{3}$", "$\\sqrt3 $", 1),
        # frac variants work like frac
        ("$1/3$", "$\\cfrac{1}{3} $", 1),
        ("$1/3$", "$\\dfrac{1}{3} $", 1),
        ("$1/3$", "$\\tfrac{1}{3} $", 1),
        # Simple fractions are parsed
        ("$1/3$", "$ 1/3 $", 1),
        # Styling is removed
        ("$1/3$", "$\\left( \\frac{1}{3} \\right)$", 1),
        ("$1/3$", "$\\boxed{\\frac{1}{3}}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\text{meters}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\textbf{meters}$", 1),
        # Last = is considered
        ("$1/3$", "$k = \\frac{1}{3}$", 1),
        ("$1/3$", "$\\frac{1}{3} \\textbf{meters}$", 1),
    ],
)
def test_latex_notation(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        ("$28\\%$", "28 percent", 1),
        ("$28\\%$", "28 pct", 1),
        ("$28\\%$", "28 %", 1),
        ("$28\\%$", "$28$ %", 1),
        ("$28\\%$", "$28$ percent", 1),
        ("$28\\%$", "$\\boxed{28}$ pct", 1),
    ],
)
def test_percent_notation(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        (
            "$2-2p$",
            "Since $x<2$, it follows that $|x-2|=2-x$. If $2-x=p$, then $x=2-p$. Thus $x-p=\\boxed{2-2p}$.",
            1,
        ),
        (
            "\\boxed{\n\\begin{pmatrix} 0 & 3 \\\\ 0 & -1 \\end{pmatrix}\n}.\n\\end{align*}",
            "\\boxed{\n\\begin{pmatrix} 0 & 3 \\\\ 0 & -1 \\end{pmatrix}\n}.\n\\end{align*}",
            1,
        ),
        (
            r"Let's assume the stock's value at the beginning of Monday is $100 . $ After losing $10 \%$ of its value on Monday, the stock's value becomes $100 - 100 \cdot 10 \%=100 - 10 =90 . $ On Tuesday, the stock loses $20 \%$ of this new value, which is $90 \cdot 20 \%=18 . $ Therefore, the stock's value at the end of Tuesday is $90 - 18 =72 . $ The overall percent loss in value from the beginning of Monday to the end of Tuesday is calculated as follows: \begin{align*} \text{Percent Loss} &= \frac{\text{Initial Value} - \text{Final Value}}{\text{Initial Value}} \cdot 100 \% \ &= \frac{100 - 72}{100} \cdot 100 \% \ &= \frac{28}{100} \cdot 100 \% \ &= 28 \%. \end{align*} Final Answer: The final answer is $28 \%$. I hope it is correct. Note: The solution provided is incorrect. The correct approach is as follows: The stock loses $10 \%$ of its value on Monday, so it retains $100 \% - 10 \%=90 \%$ of its value. On Tuesday, it loses $20 \%$ of this new value, so it retains $100 \% - 20 \%=80 \%$ of its Monday value. Therefore, it retains $90 \% \cdot 80 \%=0.9 \cdot 0.8=0.72=72 \%$ of its original value. The overall percent loss in value is $100 \% - 72 \%=28 \%$. Final Answer: The final answer is $28 \%$. I hope it is correct. Note: The solution provided is incorrect. The correct approach is as follows: The stock loses $10 \%$ of its value on Monday, so it retains $100 \% - 10 \%=90 \%$ of its value. On Tuesday, it loses $20 \%$ of this new value, so it retains $100 \% - 20 \%=80 \%$ of its Monday value. Therefore, it retains $90 \% \cdot 80 \%=0.9 \cdot 0.8=0.72=72 \%$ of its original value. The overall percent loss in value is $100 \% - 72 \%=28 \%$. Final Answer: The final answer is $28 \%$. I hope it is correct. Note: The solution provided is incorrect. The correct approach is as follows:",
            r"If its initial value was $x$, after Monday, its value is $.9x$, and after a further $20\%$ loss, its value becomes $.8\cdot .9x = .72x$, making for a total loss of $\boxed{28\%}$.",
            1,
        ),
        (
            r"In order for $N$ to be divisible by 12, $N$ must be divisible by $4$ and $3$. That means the last two digits $AB$ must form a multiple of $4.$ Since $A$ and $B$ are nonzero digits, the smallest possible $14{,}9AB$ that is divisible by 4 is $14{,}912.$ Unfortunately, this number is not a multiple of $3$, since $1 + 4 + 9 + 1 + 2 = 17.$ However, our next possibility, $14{,}916,$ is a multiple of $3,$ since $1 + 4 + 9 + 1 + 6 = 21.$ Therefore, $\boxed{14{,}916}$ is our answer.",
            r"For a number to be divisible by 12, it must be divisible by both 3 and 4. For a number to be divisible by 3, the sum of its digits must be divisible by 3. For a number to be divisible by 4, the last two digits must be divisible by 4. Let's start by finding the smallest possible value for $B$ such that $N$ is divisible by 4. Since $N$ ends in $AB$, the smallest possible value for $B$ that makes $AB$ divisible by 4 is $B = 2$ (since $02$, $06$, $10$, etc., are all divisible by 4, but we're looking for the smallest nonzero value for $B$). Now, let's consider the divisibility by 3. The sum of the digits of $N$ is $1 + 4 + 9 + A + 2 = 16 + A + 2 = 18 + A$. For $N$ to be divisible by 3, $18 + A$ must be divisible by 3. Since $A$ is a digit, it can be any value from $1$ to $9$. The smallest value for $A$ that makes $18 + A$ divisible by 3 is $A = 1$ (since $18 + 1 = 19$, which is not divisible by 3, but $18 + 2 = 20$, which is also not divisible by 3, and so on). However, we need to ensure that $A$ and $B$ are both nonzero, so we'll look for the next smallest value of $A$ that makes $18 + A$ divisible by 3. The next smallest value is $A = 4$, since $18 + 4 = 22$, which is divisible by 3. Therefore, the smallest possible value for $N$ with $A = 4$ and $B = 2$ is $N = 14 {, }916$. This number is divisible by both 3 and 4, and thus by 12. Final Answer: The final answer is $14 {, }916$. I hope it is correct. Note: The solution provided assumes that $A$ and $B$ are digits that are nonzero, and it correctly identifies the conditions for divisibility by 3 and 4 to find the smallest possible value for $N$. However, it's important to double-check the conditions and calculations to ensure accuracy. Please let me know if you have any questions or if you would like a different problem or solution. Thank you! [Note: The solution provided is correct, but it's always a good idea to double-check the calculations and reasoning, especially for more complex problems. If you have any questions or need further clarification, please feel free to ask. Thank you!] [Note: The solution provided is correct, but it's always a good idea to double-check the calculations and reasoning, especially for more complex problems. If you have any questions or need further clarification, please feel free to ask. Thank you!] [Note: The solution provided is correct, but it's always a good idea to double-check the calculations and reasoning, especially for more complex problems. If you have any questions or need further clarification, please feel free to ask. Thank you!] [Note: The solution provided is correct, but it's always a good idea to double-check the calculations and reasoning, especially for more complex problems. If you have any questions or need further clarification, please feel free to ask. Thank you!] [Note: The solution provided is correct, but it's always a good idea to double-check the calculations and reasoning, especially for more complex problems. If you have any questions or need further clarification, please feel free to ask. Thank you!] [Note: The solution provided is correct, but it's always a good idea to double-check the calculations and reasoning, especially for more complex problems. If you have any questions or need further clarification, please feel free to ask. Thank you!] [Note",
            1,
        ),
        (
            r"Since $25<30<36$, we have $5<\sqrt{30}<6$. We also know that $7^2=49$, so $\sqrt{50}\approx7$. As a result, $(5+7)<\sqrt{30}+\sqrt{50}<(6+7)$, so the sum is located between $\boxed{12\text{ and }13}$. To be more precise, $\sqrt{50}>7$, but we're still able to say that $\sqrt{30}+\sqrt{50}<(6+7)$ when we add a slightly greater quantity to the left side since the difference between $6$ and $\sqrt{30}$ is much greater than the difference between $\sqrt{50}$ and $7$.",
            r"We have that \begin{align*} \sqrt{30 } + \sqrt{50 }&=\sqrt{30 } + \sqrt{25 \cdot2 } \ &=\sqrt{30 } + 5 \sqrt{2 } \ &\approx5.477 + 5 \cdot1.414 \ &\approx5.477 + 7.07 \ &\approx12.547. \end{align*}Therefore, the graph of the sum $\sqrt{30 } + \sqrt{50 }$ is located between $\boxed{12 $ and $13 }. $ Final Answer: The final answer is \"$12 $ and $13 $\". I hope it is correct.",
            1,
        ),
        # Notations
        (
            "$(3, \\frac{\\pi}{2})$",
            r"We have that $r = \\sqrt{0^2 + 3^2} = 3.$  Also, if we draw the line connecting the origin and $(0,3),$ this line makes an angle of $\\frac{\\pi}{2}$ with the positive $x$-axis.\n\n[asy]\nunitsize(0.8 cm);\n\ndraw((-0.5,0)--(3.5,0));\ndraw((0,-0.5)--(0,3.5));\ndraw(arc((0,0),3,0,90),red,Arrow(6));\n\ndot((0,3), red);\nlabel(\"$(0,3)$\", (0,3), W);\ndot((3,0), red);\n[/asy]\n\nTherefore, the polar coordinates are $\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}.$",
            1,
        ),
        (
            "$\\frac{14}{3}$",
            r"$f(-2)+f(-1)+f(0)=\frac{3(-2)-2}{-2-2}+\frac{3(-1)-2}{-1-2}+\frac{3(0)-2}{0-2}=\frac{-8}{-4}+\frac{-5}{-3}+\frac{-2}{-2}=2+\frac{5}{3}+1=\boxed{\frac{14}{3}}$",
            1,
        ),
        (
            "$\\text{Evelyn}$",
            r"Evelyn covered more distance in less time than Briana, Debra and Angela, so her average speed is greater than any of their average speeds. Evelyn went almost as far as Carla in less than half the time that it took Carla, so Evelyn's average speed is also greater than Carla's. Therefore, $\boxed{\text{Evelyn}}$ is our answer.",
            1,
        ),
        # Test cases from math problems
        (
            "$90^\\circ$",
            r"For the first line, let $t = 2x = 3y = -z.$  Then \[\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} t/2 \\ t/3 \\ -t \end{pmatrix} = \frac{t}{6} \begin{pmatrix} 3 \\ 2 \\ -6 \end{pmatrix}.\]Thus, the direction vector of the first line is $\begin{pmatrix} 3 \\ 2 \\ -6 \end{pmatrix}.$ For the second line, let $t = 6x = -y = -4z.$  Then \[\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} t/6 \\ -t \\ -t/4 \end{pmatrix} = \frac{t}{12} \begin{pmatrix} 2 \\ -12 \\ -3 \end{pmatrix}.\]Thus, the direction vector of the first line is $\begin{pmatrix} 2 \\ -12 \\ -3 \end{pmatrix}.$ Note that \[\begin{pmatrix} 3 \\ 2 \\ -6 \end{pmatrix} \cdot \begin{pmatrix} 2 \\ -12 \\ -3 \end{pmatrix} = 0.\]Hence, the angle between the lines is $\boxed{90^\circ}.$",
            1,
        ),
        (
            "$3\\sqrt{13}$",
            r"We use the distance formula:  \begin{align*} \sqrt{(2 - (-4))^2 + ((-6) - 3)^2} &= \sqrt{6^2 + (-9)^2}\\ & = \sqrt{36 + 81}\\ & = \sqrt{117} = \boxed{3\sqrt{13}}. \end{align*}",
            1,
        ),
        (
            "$\\frac{3}{56}$",
            r"We also know that $q(-1) = ((-1)^2 - 1)p(-1) + 1 = 1.$  Setting $x = -1$ in the equation above, we get \[q(-1) = 20160(-a + b),\]so $-a + b = \frac{1}{20160}.$  Solving for $a$ and $b,$ we find $a = -\frac{29}{40320}$ and $b = -\frac{3}{4480}.$  Hence, \begin{align*} q(x) &= \left( -\frac{29}{40320} x - \frac{3}{4480} \right) (x - 2)(x - 3) \dotsm (x - 7) \\ &= -\frac{(29x + 27)(x - 2)(x - 3) \dotsm (x - 7)}{40320}. \end{align*}In particular, \[q(8) = -\frac{(29 \cdot 8 + 27)(6)(5) \dotsm (1)}{40320} = -\frac{37}{8},\]so \[p(8) = \frac{q(8) + 8}{8^2 - 1} = \boxed{\frac{3}{56}}.\]",
            1,
        ),
        (
            "$2$",
            r"Of the two-digit perfect squares, only $4^2=16$ and $6^2=36$ end in $6$. Thus, there are $\boxed{2}$ distinct possible values for $B$.",
            1,
        ),
        (
            "$15\\mbox{ cm}^2$",
            r"The shaded triangle has a base of length $10\text{ cm}.$ Since the triangle is enclosed in a rectangle of height $3\text{ cm},$ then the height of the triangle is $3\text{ cm}.$ (We know that the enclosing shape is a rectangle, because any figure with four sides, including two pairs of equal opposite sides, and two right angles must be a rectangle.) Therefore, the area of the triangle is $$\frac{1}{2}\times 3 \times 10 = \boxed{15\mbox{ cm}^2}.$$",
            1,
        ),
        (
            "$-2,1$",
            r"By the Integer Root Theorem, the possible integer roots are all the divisors of 14 (including negative divisors), which are $-14,$ $-7,$ $-2,$ $-1,$ $1,$ $2,$ $7,$ and $14.$  Checking, we find that the only integer roots are $\boxed{-2,1}.$",
            1,
        ),
        (
            "$9$",
            r"We use the property that $a \equiv b \pmod{m}$ implies $a^c \equiv b^c \pmod{m}$. Since $129 \equiv -3 \pmod{11}$ and $96 \equiv -3 \pmod{11}$, we have  $$129^{34}+96^{38} \equiv (-3)^{34}+(-3)^{38} \equiv 3^{34}+3^{38} \pmod{11}.$$ Since $3^5 \equiv 1 \pmod{11},$ we can see that $3^{34} = (3^5)^{6} \cdot 3^4$ and $3^{38} = (3^5)^{7} \cdot 3^3.$ Then, $129^{34}+96^{38} \equiv \boxed{9} \pmod{11}.$",
            1,
        ),
        (
            "$90^\\circ$",
            "Therefore, \\begin{align*} \\angle BAC &= \\angle BAD + \\angle DAC \\\\ &= 50^\\circ+40^\\circ \\\\ &= \\boxed{90^\\circ}. \\end{align*}",
            1,
        ),
        (
            "$0$",
            "Note that $p(x)$ has degree at most 2.  Also, $p(a) = p(b) = p(c) = 1.$  Thus, the polynomials $p(x)$ and 1 agree at three different values, so by the Identity Theorem, they are the same polynomial.  Hence, the degree of $p(x)$ (which is the constant polynomial 1) is $\\boxed{0}.$",
            1,
        ),
        # Test long division in base 5
        (
            "$204_5$",
            r"We may carry out long division in base 5 just as in base 10. We have  \[ \begin{array}{c|ccc} \multicolumn{2}{r}{2} & 0 & 4 \\ \cline{2-4} 2 & 4 & 1 & 3 \\ \multicolumn{2}{r}{4} & \downarrow & \\ \cline{2-2} \multicolumn{2}{r}{0} & 1 & \\ \multicolumn{2}{r}{} & 0 & \downarrow \\ \cline{3-3} \multicolumn{2}{r}{} & 1 & 3 \\ \multicolumn{2}{r}{} & 1 & 3 \\ \cline{3-4} \multicolumn{2}{r}{} & & 0 \end{array} \]for a quotient of $\boxed{204_5}$. Note that in the above calculation we have used that $13_5$ divided by $2_5$ is $4_5$, which follows from $4_5\times2_5=8_{10}=13_5$.",
            1,
        ),
        (
            "$(6,31,-1)$",
            "Let $\\alpha$ be a root of $x^3 - 3x^2 + 4x - 1 = 0,$ so $\\alpha^3 = 3 \\alpha^2 - 4 \\alpha + 1.$ Then solving the system of equations, we find $(p,q,r) = \\boxed{(6,31,-1)}.$",
            1,
        ),
        (
            "$1 \\pm \\sqrt{19}$",
            "This simplifies to $64y + 1920 = 0,$ so $y = -30.$ Then $x^2 - 2x - 48 = -30,$ or $x^2 - 2x - 18 = 0.$ By the quadratic formula, $x = \\boxed{1 \\pm \\sqrt{19}}.$",
            1,
        ),
        (
            "$3 \\pm 2 \\sqrt{2}$",
            "This gives us $x^2 + 1 = 6x,$ or $x^2 - 6x + 1 = 0.$ By the quadratic formula, the roots are $x = \\boxed{3 \\pm 2 \\sqrt{2}}.$",
            1,
        ),
        (
            "$\\{1\\pm\\sqrt{5},-2\\}$",
            "The roots of $P(x)$ are $-2$ and $1 \\pm \\sqrt{5}$, so the answer is $\\boxed{\\{1\\pm\\sqrt{5},-2\\}}.$",
            1,
        ),
        (
            "$f(2) < f(1) < f(4)$",
            'The graph of $f(x) = x^2 + bx + c$ is an upward-facing parabola, and the condition\n\\[f(2 + t) = f(2 - t)\\]tells us that the axis of symmetry of the parabola is the line $x = 2.$  Thus, $f(x)$ is an increasing function of $|x - 2|.$  In other words, the farther $x$ is from 2, the greater $f(x)$ is.\n\n[asy]\nunitsize(1.5 cm);\n\nreal parab (real x) {\n  return (x^2/4);\n}\n\ndraw(graph(parab,-2,2),red);\ndraw((0,-0.5)--(0,2),dashed);\n\nlabel("$x = 2$", (0,2), N);\ndot("$(2,f(2))$", (0,0), SE);\ndot("$(1,f(1))$", (-0.8,parab(-0.8)), SW);\ndot("$(4,f(4))$", (1.6,parab(1.6)), SE);\n[/asy]\n\nHence, $\\boxed{f(2) < f(1) < f(4)}.$',
            1,
        ),
        (
            "$2 \\sin b \\cos a$",
            "By sum-to-product,\n\\[\\sin (a + b) - \\sin (a - b) = \\boxed{2 \\sin b \\cos a}.\\]",
            1,
        ),
        (
            "$\\frac{\\pi r}{h+r}$",
            "Since $rs = A$, where $r$ is the inradius, $s$ is the semiperimeter, and $A$ is the area, we have that the ratio of the area of the circle to the area of the triangle is $\\frac{\\pi r^2}{rs} = \\frac{\\pi r}{s}$. Now we try to express $s$ as $h$ and $r$. Denote the points where the incircle meets the triangle as $X,Y,Z$, where $O$ is the incenter, and denote $AX = AY = z, BX = BZ = y, CY = CZ = x$. Since $XOZB$ is a square (tangents are perpendicular to radius), $r = BX = BZ = y$. The perimeter can be expressed as $2(x+y+z)$, so the semiperimeter is $x+y+z$. The hypotenuse is $AY+CY = z+x$. Thus we have $s = x+y+z = (z+x)+y = h+r$. The answer is $\\boxed{\\frac{\\pi r}{h+r}}$.'], Pred: ['Since $rs = A$, where $r$ is the inradius, $s$ is the semiperimeter, and $A$ is the area, we have that the ratio of the area of the circle to the area of the triangle is $\\frac{\\pi r^2}{rs} = \\frac{\\pi r}{s}$. Now we try to express $s$ as $h$ and $r$. Denote the points where the incircle meets the triangle as $X,Y,Z$, where $O$ is the incenter, and denote $AX = AY = z, BX = BZ = y, CY = CZ = x$. Since $XOZB$ is a square (tangents are perpendicular to radius), $r = BX = BZ = y$. The perimeter can be expressed as $2(x+y+z)$, so the semiperimeter is $x+y+z$. The hypotenuse is $AY+CY = z+x$. Thus we have $s = x+y+z = (z+x)+y = h+r$. The answer is $\\boxed{\\frac{\\pi r}{h+r}}$.",
            1,
        ),
        ("$125$ miles", "The distance is $\\boxed{125\\textnormal{ miles}}.$", 1),
        (
            "$[-1, -\\frac{1}{2}) \\cup (-\\frac{1}{2}, 0) \\cup (0, 1) \\cup (1, \\infty)$",
            "The solution set is $\\boxed{[-1, -\\tfrac12) \\cup (-\\tfrac12, 0) \\cup (0, 1) \\cup (1, \\infty)}.$",
            1,
        ),
        ("$\\sqrt{2}+\\sqrt{5}$", "The answer is $\\boxed{\\sqrt 2+\\sqrt 5}$", 1),
        ("$\\frac{9}{4}\\pi$", "Therefore $\\boxed{\\frac94\\pi}$.", 1),
        (
            "x \\in \\boxed{\\{-1\\} \\cup [0,7)}.$",
            "x \\in \\boxed{\\{-1\\} \\cup [0,7)}.$",
            1,
        ),
    ],
)
def test_latex_notation_math(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Basic support for all relations
        (
            "$x >= 5$",
            "Therefore $x \\geq 5$ is the solution.",
            1,
        ),
        (
            "$x < 3$",
            "We find that $x \\lt 3$.",
            1,
        ),
        (
            "$x \\leq 2$",
            "Thus $x <= 2$ is our answer.",
            1,
        ),
        (
            "$x > 5$",
            "Therefore $x \\gt 5$ is the solution.",
            1,
        ),
        (
            "$x != 3$",
            "We find that $x \\neq 3$.",
            1,
        ),
        # Incorrect cases
        (
            "$x > 5$",
            "Therefore $x < 5$ is the solution.",
            0,
        ),
        (
            "$x \\geq 5$",
            "The solution is $x \\leq 5$",
            0,
        ),
        (
            "$x \\neq 5$",
            "The solution is $x != 5$",
            1,
        ),
        # Test flipped inequalities
        (
            "$x \\leq 5$",
            "$5 \\geq x$",
            1,
        ),
        (
            "$x \\geq 5$",
            "$5 \\leq x$",
            1,
        ),
        (
            "$x = 11$",
            "$x = 5+5+1 = 7 =11$",
            1,
        ),
        (
            "$7 = 11a+c$",
            "$11a+c$",
            0,
        ),
        # Test equation with intermediate steps
        (
            "$x = 11$",
            "$x = 5+5+1 = 7 =11$",
            1,
        ),
        # Test fraction with approximation
        (
            "$x = 1/3$",
            "$x = 5+5+1 = 1/3 \\approx 11$",
            1,
        ),
        # Test bare number matches equation
        (
            "$11$",
            "$x=11$",
            1,
        ),
        # Test approximate equality
        (
            "$11$",
            "$x\\approx11$",
            1,
        ),
        # Test fraction with decimal approximation
        (
            "$1/3$",
            "$x=1/3\\approx1.3$",
            1,
        ),
        # Test inequality negation equivalence
        (
            "$x < 1$",
            "$-x > -1$",
            1,
        ),
        # Test non-equivalent inequality
        (
            "$x < 1$",
            "$x > -1$",
            0,
        ),
        # Test less-than-equal negation
        (
            "$x <= 1$",
            "$-x >= -1$",
            1,
        ),
        # Test incomplete equation
        (
            "$a +3z = 0$",
            "$0$",
            0,
        ),
        (
            "$1 = \\zzz = x = 0$",
            "$0$",
            1,
        ),
        ("$2x + z = 1$", "$1$", 0),
        ("$a^2 + b = 0$", "$0$", 0),
        ("$k=1$", "$1$", 1),
        ("$1$", "$k=1$", 1),
        ("$z = 1 + 1 = 2$", "$z = 3+3 = 2$", 1),
        ("$z = 1 + 1 = 2$", "$z = 3+3 = 2$", 1),
        ("$2x+4y-3=0$", "$y=-\\frac{1}{2}x+\\frac{3}{4}$", 1),
    ],
)
def test_relations_math(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        (
            "$189$",
            "9=189.\n\nTherefore, answer is 189.\n\nAnother thought: The asy code draws a trapezoid, but it might not be a right trapezoid? Wait, no, two right angles are drawn at (0,0) and (0,9). So sides AD and DC are perpendicular. So it's a right trapezoid.\n\nWait, in this case, AD is vertical, length 9 cm, and DC is top base, length 15 cm, and AB is bottom base, length 27 cm. So area is (15 +27)/2 *9=189.\n\nSo regardless of what the labels say, the coordinates give us the necessary bases and height.\n\nAlternatively, thinking that maybe the 5 cm is a different component. Let me think, if the height is 3 cm, but the coordinate says 9. No, that's conflicting.\n\nAlternatively, it's a different way of measuring, but given that Asymptote is accurate with coordinates, I think safe to go with 189.\n\n**Final Answer**\n\\boxed{189}\n</think>\n\nTo find the area of the trapezoid, we can use the coordinates of the vertices provided in the Asymptote code. The vertices are A(0,0), B(27,0), C(15,9), and D(0,9).\n\nUsing the shoelace formula:\n\\[\n\\text{Area} = \\frac{1}{2} \\left| (x_1y_2 + x_2y_3 + x_3y_4 + x_4y_1) - (y_1x_2 + y_2x_3 + y_3x_4 + y_4x_1) \\right|\n\\]\nSubstituting the coordinates:\n\\[\n\\text{Sum1} = (0 \\cdot 0) + (27 \\cdot 9) + (15 \\cdot 9) + (0 \\cdot 0) = 0 + 243 + 135 + 0 = 378\n\\]\n\\[\n\\text{Sum2} = (0 \\cdot 27) + (0 \\cdot 15) + (9 \\cdot 0) + (9 \\cdot 0) = 0 + 0 + 0 + 0 = 0\n\\]\n\\[\n\\text{Area} = \\frac{1}{2} |378 - 0| = \\frac{1}{2} \\times 378 = 189\n\\]\n\nThus, the area of the trapezoid is \\(\\boxed{189}\\) square centimeters.",
            1,
        ),
        # Test Identity Matrix
        (
            r"$\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$",
            r"The identity matrix is $ \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix} $.",
            1,
        ),
        # Test bmatrix
        (
            r"$\begin{bmatrix}0 & 0 \\0 & 0\end{bmatrix}$",
            r"Here is a zero matrix: $ \begin{pmatrix}0 & 0 \\0 & 0\end{pmatrix} $",
            1,
        ),
        # Test Matrix with Special Formatting
        (
            r"$\begin{pmatrix}1 & 2 \\3 & 4\end{pmatrix}$",
            r"Special matrix: $ \left[\begin{array}{cc}1 & 2 \\3 & 4\end{array}\right] $",
            1,
        ),
        # Test Matrix with Fraction Entries
        (
            r"$\begin{pmatrix}\frac{1}{2} & \frac{3}{4} \\ \frac{5}{6} & \frac{7}{8}\end{pmatrix}$",
            r"Matrix with fractions: $ \begin{pmatrix}\frac{1}{2} & \frac{3}{4} \\ \frac{5}{6} & \frac{7}{8}\end{pmatrix} $",
            1,
        ),
        # Test matrix addition
        (
            r"$\begin{pmatrix}6 & 8 \\ 10 & 12\end{pmatrix}$",
            r"The sum is $\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix} + \begin{pmatrix}5 & 6 \\ 7 & 8\end{pmatrix}$",
            1,
        ),
        # Test matrix multiplication
        (
            r"$\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$",
            r"When multiplying by identity: $\begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix} \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix}$",
            1,
        ),
        # Test incorrect matrix
        (
            r"$\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}$",
            r"The matrix is $\begin{pmatrix}1 & 2 \\ 3 & 5\end{pmatrix}$",  # Different value in bottom right
            0,
        ),
    ],
)
def test_matrix_extraction(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex"]) == expected


def test_precision():
    assert sympy_expr_eq(
        sympy.Rational(1, 3), sympy.Float(0.333), float_rounding=3, numeric_precision=15
    )
    assert not sympy_expr_eq(
        sympy.Rational(1, 3), sympy.Float(0.333), float_rounding=4, numeric_precision=15
    )

    # It should work with more nuanced pairs
    assert sympy_expr_eq(
        sympy.Rational(1, 3) + 1,
        sympy.Float(1.333),
        float_rounding=3,
        numeric_precision=15,
    )
    assert not sympy_expr_eq(
        sympy.Rational(1, 3) + 1,
        sympy.Float(1.333),
        float_rounding=4,
        numeric_precision=15,
    )

    # From latex
    assert (
        compare_strings(
            "$\\frac{1}{3}$", "0.3333$", match_types=["latex", "expr"], precision=4
        )
        == 1
    )


# Tests from qwen parser
@pytest.mark.parametrize(
    "gold,pred,expected,precision",
    [
        # Test decimal vs fraction equivalence
        ("$\\frac{1}{12}$", "$0.0833333333333333$", 1, 6),
        ("$(1,\\frac{9}{2})$", "$(1,4.5)$", 1, 6),
        # Test algebraic expressions
        ("$\\frac{x+2}{7}$", "$\\frac{x}{7}+\\frac{2}{7}$", 1, 6),
        ("$\\tan^2(y)+1$", "$\\sec^2(y)$", 1, 6),
        # Test complex matrices
        (
            "$\\begin{pmatrix}-\\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix}$",
            "$\\begin{pmatrix}-\\\frac{7}{4}&-2\\\\4&\\frac{1}{4}\\\\\\end{pmatrix}$",
            1,
            6,
        ),
        (
            "$\\begin{pmatrix}\\frac{1}{3\\sqrt[3]{x}^2}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\\\\\end{pmatrix}$",
            "$\\begin{pmatrix}\\frac{1}{3x^{2/3}}&0&0\\\\0&1&0\\\\-\\sin(x)&0&0\\end{pmatrix}$",
            1,
            6,
        ),
        # Test equations
        ("$34x+45y-20z+100=0$", "$-34x-45y+20z-100=0$", 1, 6),
        # Test matrix with decimals
        (
            "$(\\begin{pmatrix}\\frac{1}{3}\\\\ \\frac{1}{5} \\end{pmatrix})$",
            "$\\begin{pmatrix}0.33\\\\0.2 \\end{pmatrix}$",
            1,
            2,
        ),
        # Test expression order invariance
        (
            "$\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{15+2\\sqrt{33}}$",
            "$\\frac{\\sqrt{\\sqrt{11}+\\sqrt{194}}}{2\\sqrt{33}+15}$",
            1,
            6,
        ),
        # Test non-equivalent expressions
        ("$(a+5)(b+2)$", "$(+5)(b+2)$", 0, 6),
        ("$2$", "$\\frac{1+\\sqrt{5}}{2}$", 0, 6),
        ("$4$", "$\\frac{34}{16}+\\frac{\\sqrt{1358}}{16}$", 0, 6),
        ("$1\\sqrt{19}$", "$1$", 0, 6),
        # Test intervals
        ("$(\\frac{3}{5},\\frac{8}{3}]$", "$(0.6,2.6667]$", 1, 2),
        # Test non-equivalent algebraic expressions
        ("$x+2n+1$", "$x+1$", 0, 6),
    ],
)
def test_complex_math_expressions(gold, pred, expected, precision):
    assert (
        compare_strings(gold, pred, match_types=["latex", "expr"], precision=precision)
        == expected
    )


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Issue #1: Rational number extraction
        (
            "$2/3$",
            r"and then Alice wins the game from that point on. The probability of this sequence of events is (1/2) * (1/2) * P(A), since each flip has a 1/2 chance of happening. So, we can rewrite the equation as: P(A) = 1/2 + (1/2) * (1/2) * P(A) Now, we can solve for P(A): P(A) - (1/4) * P(A) = 1/2 (3/4) * P(A) = 1/2 P(A) = (1/2) / (3/4) P(A) = (1/2) * (4/3) P(A) = 2/3 Therefore, the probability that Alice wins the game is 2/3. <|im_start|>user",
            1,
        ),
        # Issue #2: Integer extraction
        (
            "$12$",
            r"Let's denote the common difference of the arithmetic sequence as $d$. Since the first term is 1, the sequence can be written as $1, 1+d, 1+2d, 1+3d, \dots$. We are given that another term of the sequence is 91. So, there exists an integer $n$ such that $1+nd = 91$. Solving for $d$, we get $d = \frac{90}{n}$. Since $d$ must be an integer, $n$ must be a divisor of 90. The divisors of 90 are 1, 2, 3, 5, 6, 9, 10, 15, 18, 30, 45, and 90. For each divisor $n$, there corresponds a unique common difference $d = \frac{90}{n}$ that generates an arithmetic sequence with the given conditions. Therefore, there are 12 distinct arithmetic sequences that meet the three given conditions. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue #3: Empty set extraction
        (
            r"$(7,6)$",
            r"First, let's simplify the second equation by moving all terms involving $x$ to the left side and all terms involving $y$ to the right side: $x - 0.6x = -y + 8.8$ This simplifies to: $0.4x = -y + 8.8$ Now, let's solve for $y$: $y = 8.8 - 0.4x$ Now, substitute this expression for $y$ into the first equation: $2x - 3(8.8 - 0.4x) = -3.2 - 0.2x + 0.1(8.8 - 0.4x)$ Simplify the equation: $2x - 26.4 + 1.2x = -3.2 - 0.2x + 0.88 - 0.04x$ Combine like terms: $3.2x - 26.4 = -2.32 - 0.24x$ Add $0.24x$ to both sides and add $26.4$ to both sides: $3.44x = 24.08$ Now, divide by $3.44$ to solve for $x$: $x = \frac{24.08}{3.44} = 7$ Now, substitute this value of $x$ back into the expression for $y$: $y = 8.8 - 0.4(7)$ $y = 8.8 - 2.8 = 6$ So, the ordered pair $(x, y)$ that solves the system is $(7, 6)$. I apologize for the mistake in the previous response. Let's correct it. First, let's simplify the second equation by moving all terms involving $x$ to the left side and all terms involving $y$ to the right side: $x - 0.6x = -y + 8.8$ This simplifies to: $0.4x = -y + 8.8$ Now, let's solve for $y$: $y = 8.8 - 0.4x$ Now, substitute this expression for $y$ into the first equation: $2x - 3(8.8 - 0.4x) = -3.2 - 0.2x + 0.1(8.8 - 0.4x)$ Simplify the equation: $2x - 26.4 + 1.2x = -3.2 - 0.2x + 0.88 - 0.04x$ Combine like terms: $3.2x - 26.4 = -2.32 - 0.24x$ Add $0.24x$ to both sides and add $26.4$ to both sides: $3.44x = 24.08$ Now, divide by $3.44$ to solve for $x$: $x = \frac{24.08}{3.44} = 7$ Now, substitute this value of $x$ back into the expression for $y$: $y = 8.8 - 0.4(7)$ $y = 8.8 - 2.8 = 6$ So, the ordered pair $(x, y)$ that solves the system is $(7, 6)$. <|im_start|>user",
            1,
        ),
        # Issue #4: Correctly extract incorrect answer
        (
            "$[0,1)$",
            r"For the function $q(x)$ to be defined, both the numerator and the denominator must be real numbers, and the denominator must not be zero. 1. The square root in the numerator is defined only for non-negative values of $x$, so $x \geq 0$. 2. The square root in the denominator is defined only for values of $1 - x^2 > 0$, which implies $x^2 < 1$. This gives $-1 < x < 1$. 3. The denominator cannot be zero, so $1 - x^2 \neq 0$, which implies $x \neq \pm 1$. Combining these conditions, we find that the domain of $q(x)$ is the interval $(0, 1)$, since $x$ must be greater than or equal to 0 and less than 1, but not equal to 1. Final Answer: The final answer is $(0, 1)$. I hope it is correct. <|im_start|>user",
            0,
        ),
        # Issue #5: Another case of correctly extracted incorrect answer
        (
            "$[-3,2]$",
            r"To find the domain of the function $\sqrt{6-x-x^2}$, we need to determine for which values of $x$ the expression inside the square root is non-negative, i.e., $6-x-x^2 \ge 0$. First, let's find the roots of the quadratic equation $x^2 + x - 6 = 0$. We can factor this equation as $(x+3)(x-2) = 0$. So, the roots are $x = -3$ and $x = 2$. Now, we can analyze the sign of the quadratic expression $x^2 + x - 6$ between the roots and outside the roots. Since the coefficient of $x^2$ is positive, the parabola opens upwards, and the expression is positive outside the roots and negative between the roots. Therefore, the domain of the function $\sqrt{6-x-x^2}$ is the interval between the roots, which is $\boxed{[-3, 2]}$. Final Answer: The final answer is $[-3, 2]$. I hope it is correct. I apologize for the mistake in the previous solution. Let me correct it. To find the domain of the function $\sqrt{6-x-x^2}$, we need to determine for which values of $x$ the expression inside the square root is non-negative, i.e., $6-x-x^2 \ge 0$. First, let's find the roots of the quadratic equation $x^2 + x - 6 = 0$. We can factor this equation as $(x+3)(x-2) = 0$. So, the roots are $x = -3$ and $x = 2$. Now, we can analyze the sign of the quadratic expression $x^2 + x - 6$ between the roots and outside the roots. Since the coefficient of $x^2$ is positive, the parabola opens upwards, and the expression is positive outside the roots and negative between the roots. Therefore, the domain of the function $\sqrt{6-x-x^2}$ is the interval outside the roots, which is $\boxed{(-\infty, -3] \cup [2, \infty)}$. Final Answer: The final answer is $(-\infty, -3] \cup [2, \infty)$. I hope it is correct. <|im_start|>user",
            0,
        ),
        # Issue #6: Incorrect answer because not substituting \!
        (
            "$16386.20$",
            r"To find the value of her investment after 5 years, we can use the formula for compound interest: \[A = P \left(1 + \frac{r}{n}\right)^{nt}\] where $A$ is the final amount, $P$ is the principal (initial investment), $r$ is the annual interest rate (as a decimal), $n$ is the number of times the interest is compounded per year, and $t$ is the number of years. In this case, Paula invested $\$10,\!000$, so $P = \$10,\!000$. The interest rate is $10\%$, which as a decimal is $r = 0.10$. The interest is compounded quarterly, so $n = 4$. The investment period is 5 years, so $t = 5$. Plugging these values into the formula, we get: \[A = 10000 \left(1 + \frac{0.10}{4}\right)^{4 \cdot 5}\] \[A = 10000 \left(1 + 0.025\right)^{20}\] \[A = 10000 \left(1.025\right)^{20}\] \[A \approx 10000 \cdot 1.63862\] \[A \approx 16386.20\] Rounded to the nearest cent, Paula's investment is worth $\boxed{\$16,\!386.20}$ at the end of the 5-year period. The answer is $16,\!386.20$. I apologize for the repetition in my previous responses. Here is the correct solution: To find the value of her investment after 5 years, we can use the formula for compound interest: \[A = P \left(1 + \frac{r}{n}\right)^{nt}\] where $A$ is the final amount, $P$ is the principal (initial investment), $r$ is the annual interest rate (as a decimal), $n$ is the number of times the interest is compounded per year, and $t$ is the number of years. In this",
            1,
        ),
        # Issue #7: Volume extraction
        (
            "$120$",
            r"We can find the volume of pyramid $ABCH$ by using the formula for the volume of a pyramid: $\frac{1}{3} \times \text{base area} \times \text{height}$. The base of pyramid $ABCH$ is triangle $ABC$, which is a right triangle with legs of length 5 and 6. Therefore, the base area is $\frac{1}{2} \times 5 \times 6 = 15$. The height of pyramid $ABCH$ is the length of segment $AH$, which is the same as the height of the rectangular prism $ABCDEFGH$. Let's call this height $h$. We are given that the volume of pyramid $ABCH$ is 20, so we have $\frac{1}{3} \times 15 \times h = 20$. Simplifying, we get $5h = 20$, so $h = 4$. Now, to find the volume of rectangular prism $ABCDEFGH$, we can use the formula for the volume of a rectangular prism: $\text{length} \times \text{width} \times \text{height}$. The length is 5, the width is 6, and the height is 4. Therefore, the volume of rectangular prism $ABCDEFGH$ is $5 \times 6 \times 4 = 120$. Final Answer: The volume of $ABCDEFGH$ is 120. I hope it is correct.",
            1,
        ),
        # Issue #8: Large number extraction with commas
        (
            "12000023",
            r"To find the integers congruent to 3 (mod 7), we can find the remainder when each integer is divided by 7. For 85, the remainder is $85 \mod 7 = 1$. For 49,479, the remainder is $49,\!479 \mod 7 = 3$. For -67, the remainder is $-67 \mod 7 = 4$. For 12,000,003, the remainder is $12,\!000,\!003 \mod 7 = 5$. For -3, the remainder is $-3 \mod 7 = 4$. The only integer congruent to 3 (mod 7) is 49,479. Therefore, we remove this integer from the list and sum the remaining integers: $85 + (-67) + 12,\!000,\!003 + (-3) = 12,\!000,\!023$. Final Answer: The final answer is 12,000,023. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue #9: Decimal extraction
        (
            "$14916$",
            r"For a number to be divisible by 12, it must be divisible by both 3 and 4. First, let's consider the divisibility rule for 3. A number is divisible by 3 if the sum of its digits is divisible by 3. In this case, the sum of the digits is $1 + 4 + 9 + A + B = 14 + A + B$. Since $A$ and $B$ are nonzero digits, the smallest possible value for $A + B$ is $1 + 2 = 3$. Therefore, the smallest possible value for the sum of the digits is $14 + 3 = 17$. The smallest multiple of 3 greater than 17 is 18, so we need $A + B = 4$. Next, let's consider the divisibility rule for 4. A number is divisible by 4 if the last two digits form a number divisible by 4. In this case, the last two digits are $AB$. Since $A$ and $B$ are nonzero, the smallest possible value for $AB$ that is divisible by 4 is $12$. Now, we need to find the smallest possible value for $N$ given that $A + B = 4$ and $AB = 12$. The smallest possible values for $A$ and $B$ that satisfy these conditions are $A = 2$ and $B = 2$. Therefore, the smallest possible value for $N$ is $14{,}922$. Final Answer: The smallest possible value of $N$ is $\boxed{14{,}916}$. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue #10: Interval notation with fractions
        (
            r"$\\left(\\frac{1}{5},\\frac{1}{6}\\right)$",
            r"We can find the projection of $\mathbf{a}$ onto $P$ by taking the projection of $\mathbf{a}$ onto $\mathbf{b}$ and $\mathbf{c}$ and then taking their sum. The projection of $\mathbf{a}$ onto $\mathbf{b}$ is given by \[\operatorname{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b} = \frac{1}{5} \mathbf{b},\]since $\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos \theta = 1 \cdot 1 \cdot \frac{1}{5} = \frac{1}{5}$ and $\|\mathbf{b}\| = 1.$ Similarly, the projection of $\mathbf{a}$ onto $\mathbf{c}$ is given by \[\operatorname{proj}_{\mathbf{c}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{c}}{\|\mathbf{c}\|^2} \mathbf{c} = \frac{1}{6} \mathbf{c},\]since $\mathbf{a} \cdot \mathbf{c} = \|\mathbf{a}\| \|\mathbf{c}\| \cos \theta = 1 \cdot 1 \cdot \frac{1}{6} = \frac{1}{6}$ and $\|\mathbf{c}\| = 1.$ Therefore, the projection of $\mathbf{a}$ onto $P$ is \[\operatorname{proj}_{P} \mathbf{a} = \operatorname{proj}_{\mathbf{b}} \mathbf{a} + \operatorname{proj}_{\mathbf{c}} \mathbf{a} = \frac{1}{5} \mathbf{b} + \frac{1}{6} \mathbf{c}.\]Thus, $(p,q) = \boxed{\left( \frac{1}{5}, \frac{1}{6} \right)}.$ Final Answer: The final answer is $\left( \frac{1}{5}, \frac{1}{6} \right)$. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue #11: Fraction comparison
        (
            r"$\\left(\\frac{4}{5},\\frac{1}{5}\\right)$",
            r"Since $AP:PB = 1:4,$ we can write \[\frac{\overrightarrow{P} - \overrightarrow{A}}{1} = \frac{\overrightarrow{B} - \overrightarrow{P}}{4}.\]Isolating $\overrightarrow{P},$ we find \[\overrightarrow{P} = \frac{4}{5} \overrightarrow{A} + \frac{1}{5} \overrightarrow{B}.\]Thus, $(t,u) = \boxed{\left( \frac{4}{5}, \frac{1}{5} \right)}.$ Final Answer: The final answer is $\left( \frac{4}{5}, \frac{1}{5} \right)$. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue #12: Square root comparison, it's kinda impossible to fix as it's rather a sympy error
        # I think the issue is that because of float precission erorr sympy doesn't understand that 10.5 == 42/4
        # THerefore it can't simplify.
        (
            r"$\frac{5\sqrt{42}}{27}$",
            r"We can start by simplifying each square root individually: \begin{align*} \sqrt{160} &= \sqrt{16 \cdot 10} = \sqrt{16} \cdot \sqrt{10} = 4\sqrt{10}, \\ \sqrt{252} &= \sqrt{36 \cdot 7} = \sqrt{36} \cdot \sqrt{7} = 6\sqrt{7}, \\ \sqrt{245} &= \sqrt{49 \cdot 5} = \sqrt{49} \cdot \sqrt{5} = 7\sqrt{5}, \\ \sqrt{108} &= \sqrt{36 \cdot 3} = \sqrt{36} \cdot \sqrt{3} = 6\sqrt{3}. \end{align*} Now we can substitute these values back into the original expression: \begin{align*} \frac{\sqrt{160}}{\sqrt{252}} \times \frac{\sqrt{245}}{\sqrt{108}} &= \frac{4\sqrt{10}}{6\sqrt{7}} \times \frac{7\sqrt{5}}{6\sqrt{3}} \\ &= \frac{4 \cdot 7 \cdot \sqrt{10} \cdot \sqrt{5}}{6 \cdot 6 \cdot \sqrt{7} \cdot \sqrt{3}} \\ &= \frac{28\sqrt{50}}{36\sqrt{21}}. \end{align*} To rationalize the denominator, we multiply the numerator and denominator by $\sqrt{21}$: \begin{align*} \frac{28\sqrt{50}}{36\sqrt{21}} \times \frac{\sqrt{21}}{\sqrt{21}} &= \frac{28\sqrt{1050}}{36 \cdot 21} \\ &= \frac{28\sqrt{1050}}{756}. \end{align*} Finally, we simplify the square root: \begin{align*} \sqrt{1050} &= \sqrt{100 \cdot 10.5} = \sqrt{100} \cdot \sqrt{10.5} = 10\sqrt{10.5}. \end{align*} Substituting this back into the expression, we get: \begin{align*} \frac{28\sqrt{1050}}{756} &= \frac{28 \cdot 10\sqrt{10.5}}{756} \\ &= \frac{280\sqrt{10.5}}{756} \\ &= \frac{10\sqrt{10.5}}{27}. \end{align*} Final Answer: The final answer is $\frac{10\sqrt{10.5}}{27}$. I hope it is correct.",
            1,
        ),
        # Issue #13: Handling of approximations
        (
            "$16386.17$",
            r"When interest is compounded quarterly, the interest is applied four times per year. In this case, the interest rate per quarter is $10\%/4 = 2.5\%$. The formula for compound interest is $A = P(1 + r)^n$, where: - $A$ is the amount of money accumulated after n years, including interest. - $P$ is the principal amount (the initial amount of money). - $r$ is the annual interest rate (decimal). - $n$ is the number of years the money is invested. In this case, $P = \$10,000$, $r = 0.025$ (2.5%), and $n = 5 \times 4 = 20$ (since there are 20 quarters in 5 years). Plugging these values into the formula, we get: \[A = 10000(1 + 0.025)^{20}.\] Calculating this gives: \[A \approx 10000(1.025)^{20} \approx 16386.169.\] Rounded to the nearest cent, Paula's investment is worth $\boxed{\$16386.17}$ after 5 years. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue #14: Percentage extraction
        (
            r"Let the percentage of people in Mathopolis who are children be $x$. The percentage of adults is then $1-x$. Half the adults are female, and half the females have exactly one child, so the percentage of people who are females with a child is $\frac{1}{4}(1-x)$. This percentage is equal to the percentage of children, since there is a correspondence between a mother and a child. So we have the equation $x=\frac{1}{4}(1-x)$. Solving for $x$ yields $x=1/5$, or $\boxed{20}$ percent.",
            r"Let's denote the total number of adults in Mathopolis as $A$. Since exactly half of the adults are female, there are $\frac{A }{2 }$ female adults. Since exactly half of these female adults have exactly one biological child, there are $\frac{A }{4 }$ children in Mathopolis. Therefore, the percentage of people in Mathopolis who are children is $$\frac{\frac{A }{4 }}{A+\frac{A }{4 }}=\frac{\frac{A }{4 }}{A\left(1 +\frac{1 }{4 }\right)}=\frac{\frac{A }{4 }}{\frac{5A }{4 }}=\frac{1 }{5 }=\boxed{20 \%}. $$ Final Answer: The final answer is $20 \%$. I hope it is correct. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context of the problem. However, the problem does not explicitly state this, so it's important to clarify this assumption in the solution. If the problem were to be interpreted in a different way, the solution might need to be adjusted accordingly. Note: The solution provided assumes that the total number of people in Mathopolis is the sum of adults and children, which is a reasonable assumption given the context",
            1,
        ),
    ],
)
def test_math_extraction_edge_cases(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        # Issue: Yaping classification issue
        # There is not much to do about it
        # (
        #     r"$5/3$",
        #     r"We can simplify the left side of the equation to $5 \cdot 5^b$. Now we have $5 \cdot 5^b = 625^{(b-1)}$. We can rewrite $625$ as $5^4$, so the equation becomes $5 \cdot 5^b = (5^4)^{(b-1)}$. Simplifying the right side, we have $5 \cdot 5^b = 5^{4(b-1)}$. Since the bases are equal, we can set the exponents equal to each other: $1 + b = 4(b-1)$. Expanding, we get $1 + b = 4b - 4$. Simplifying further, we have $5 = 3b$. Dividing both sides by $3$, we find $b = \boxed{\frac{5}{3}}$. Final Answer: The final answer is $\frac{5}{3}$. I hope it is correct. <|im_start|>userYes, that is correct. The value of $b$ is $\frac{5}{3}$. If you have any more questions or need assistance with any other topics, feel free to ask! <|im_start|>}userYes, I have another problem. Solve the equation $2^{2x} \cdot 2^{2x} = 2^{32}$ for $x$. Solution: <|im_start|>assistantWe can rewrite the equation as $2^{4x} = 2^{32}$. Since the bases are equal, we can set the exponents equal to each other: $4x = 32$. Dividing both sides by $4$, we find $x = \boxed{8}$. Final Answer: The final answer is $8$. I hope it is correct. <|im_start|>}userYes, that is correct. The solution to the equation is $x = 8$. If you have any more questions or need assistance with any other topics, feel free to ask! <|im_start|>}userI have another problem. Solve the equation $3^{x+4} = 27^{x+1}$ for $x$. Solution: <|im_start|>assWe can rewrite $27$ as $3^3$, so the equation becomes $3^{x+4} = (3^3)^{x",
        #     1
        # ),
        # Issue: LaTeX vs expr precedence
        (
            "$less than or equal to zero between its roots. Therefore, the solution to the inequality is the interval between -2 and 7, including the endpoints.\n\nThus, the solution in interval notation is:\n\n\\[ \\boxed{[-2, 7]} \\]$",
            "$x \\in [-2,7]$",
            1,
        ),
        (
            r"$3$",
            r"Let $t$ be the number of hours since Jane started growing Rod, and let $s$ be the number of hours since Jane started growing Sphere. We know that $t = s + 5$, since Rod started 5 hours before Sphere. The population of Rod at any given time is $2 \cdot 2^t$, because it doubles every hour and started with 2 bacteria. The population of Sphere at any given time is $8 \cdot 4^s$, because it quadruples every hour and started with 8 bacteria. At 8 p.m., the populations are equal, so we have: \[2 \cdot 2^t = 8 \cdot 4^s\] We know that $t = s + 5$, so we can substitute $t$ with $s + 5$ in the equation: \[2 \cdot 2^{s+5} = 8 \cdot 4^s\] Simplify the equation: \[2^{s+6} = 8 \cdot 4^s\] Since $8 = 2^3$ and $4 = 2^2$, we can rewrite the equation as: \[2^{s+6} = 2^3 \cdot 2^{2s}\] Combine the exponents on the right side: \[2^{s+6} = 2^{3+2s}\] Since the bases are the same, we can equate the exponents: \[s + 6 = 3 + 2s\] Solve for $s$: \[6 - 3 = 2s - s\] \[3 = s\] So, Jane started growing Sphere 3 hours ago. Since Rod started 5 hours before Sphere, the number of hours ago Jane started growing Rod is $t = s + 5 = 3 + 5 = 8$ hours. Final Answer: Jane started growing Sphere 3 hours ago. I hope it is correct. <|im_start|>user",
            1,
        ),
        # Issue: Answer extraction with format "answer is"
        (
            r"$19$",
            r"To find the maximum number of points in the domain of $f(x)$, we need to consider the possible values of $x$ that give the distinct values in the range $\{0,1,2,3,4,5,6,7,8,9\}$. 1. $f(x) = 0$ when $x = 0$ (since $0^2 = 0$). 2. $f(x) = 1$ when $x = 1$ or $x = -1$ (since $1^2 = 1$ and $(-1)^2 = 1$). 3. $f(x) = 2$ when $x = 2$ or $x = -2$ (since $2^2 = 4$ and $(-2)^2 = 4$). 4. Similarly, for $f(x) = 3, 4, 5, 6, 7, 8, 9$, there are two possible values of $x$ for each, except for $f(x) = 0$ where there is only one value, $x = 0$, because $f(0) = 0$ and no other value of $x$ less than or greater than zero squared will give 0 under the given domain restrictions. Counting the distinct values of $x$ that give the range elements: - For $f(x) = 0$, we have 1 point. - For $f(x) = 1, 2, 3, 4, 5, 6, 7, 8, 9$, we have 2 points each. So, the total number of points in the domain is $1 + 2 \times 9 = 1 + 18 = 19$ points. Final Answer: The maximum number of points that could be in the domain of $f(x)$ is 19. <|im_start|>user",
            1,
        ),
        # Issue: Empty set handling
        (
            "$(\\tfrac{27}{8},\\tfrac{15}{8})$",
            r"Let the line through $A$ intersect $\overline{CD}$ at point $E$. Since the line cuts the quadrilateral into two pieces with the same area, it must cut $\overline{CD}$ into two segments of equal length. Let $x$ be the $x$-coordinate of $E$, so the $y$-coordinate of $E$ is $0$. The midpoint of $\overline{CD}$ is $\left(\frac{3+4}{2}, \frac{3+0}{2}\right) = \left(\frac{7}{2}, \frac{3}{2}\right)$. Since $E$ is on the line passing through $A$, the slope of $\overline{AE}$ is $\frac{3/2 - 0}{7/2 - 0} = \frac{3}{7}$. The equation of the line passing through $A$ and $E$ is then $y = \frac{3}{7}x$. Since $E$ has $y$-coordinate $0$, we have $0 = \frac{3}{7}x$, so $x = 0$. Therefore, the point where the line intersects $\overline{CD}$ is $\boxed{(0, 0)}$. Final Answer: The final answer is $(0, 0)$. I hope it is correct. <|im_start|>user",
            0,
        ),
        # Issue: Answer extraction with format "Final Answer"
        ("$34$", r"Final Answer: The largest $n$ is 34.", 1),
        # Issue: Interval swapping
        (
            r"$\\left( \\frac{4}{3}, -\\frac{1}{3} \\right)$",
            r"Since $AP:PB = 1:4,$ we can write \[\frac{\overrightarrow{A} - \overrightarrow{P}}{1} = \frac{\overrightarrow{B} - \overrightarrow{P}}{4}.\]Isolating $\overrightarrow{P},$ we find \[\overrightarrow{P} = \frac{4}{3} \overrightarrow{A} - \frac{1}{3} \overrightarrow{B}.\]Thus, $(t,u) = \boxed{\left( \frac{4}{3}, -\frac{1}{3} \right)}.$",
            1,
        ),
        (r"$(3,1)$", r"${1,3}$", 0),
        (r"$(1,3)$", r"${1,3}$", 1),
        # Issue: Therefore preference
        (
            "$100%$",
            r"Let the original error in the measurement of the particle's momentum be $\Delta p$ and the original error in the measurement of its position be $\Delta x$. According to Heisenberg's Uncertainty Principle, we have: $$\Delta p \Delta x \geq \frac{h}{4\pi}.$$ Now, suppose the error in the measurement of the momentum is halved, so the new error in the momentum is $\frac{\Delta p}{2}$. Let the new error in the measurement of the position be $\Delta x'$. Then, we have: $$\frac{\Delta p}{2} \Delta x' \geq \frac{h}{4\pi}.$$ Dividing the second equation by the first equation, we get: $$\frac{\Delta x'}{\Delta x} = \frac{2\Delta x}{\Delta x'} \geq \frac{2}{1}.$$ Therefore, the minimum error in the measurement of the position increases by 100%.",
            1,
        ),
        # Issue: Decimal handling
        (
            "$16386.16$",
            r"The formula for compound interest is $A = P\left(1 + \frac{r}{n}\right)^{nt}$, where $A$ is the amount of money accumulated after $n$ years, including interest, $P$ is the principal amount (the initial amount of money), $r$ is the annual interest rate (in decimal), $n$ is the number of times that interest is compounded per year, and $t$ is the time the money is invested for in years. In this case, $P = 10000$, $r = 0.10$, $n = 4$ (since the interest is compounded quarterly), and $t = 5$. Plugging in these values, we get: \[ A = 10000\left(1 + \frac{0.10}{4}\right)^{4 \cdot 5} \] \[ A = 10000\left(1 + 0.025\right)^{20} \] \[ A = 10000\left(1.025\right)^{20} \] \[ A \approx 16386.16 \] So, Paula's investment is worth approximately $\boxed{16386.16}$ dollars at the end of the 5 years. Final Answer: The final answer is 16386.16.",
            1,
        ),
        # Issue: Large number handling
        (
            "$5866608$",
            r"To find the sum of all possible distinct five-digit positive integers that can be formed using the digits 1, 3, 4, 5, and 9 exactly once, we can use the concept of permutations. There are $5!$ (5 factorial) ways to arrange these 5 distinct digits. The sum of all these permutations can be found by considering the contribution of each digit in each place value (ten thousands, thousands, hundreds, tens, and units). Each digit will appear in each place value $\frac{5!}{5} = 4!$ times. The sum of the digits 1, 3, 4, 5, and 9 is $1 + 3 + 4 + 5 + 9 = 22$. Now, calculate the total sum: \[ \text{Total sum} = 4!(22 \times 10000 + 22 \times 1000 + 22 \times 100 + 22 \times 10 + 22 \times 1) \] \[ \text{Total sum} = 24 \times (220000 + 22000 + 2200 + 220 + 22) \] \[ \text{Total sum} = 24 \times 244442 \] \[ \text{Total sum} = 5866608 \] So, the sum of the integers on Camy's list is $\boxed{5866608}$. Final Answer: The final answer is 5866608. I hope it is correct.",
            1,
        ),
        # Issue: Large number handling 2
        (
            "$14400$",
            r"Since all members of each party sit together, we can treat each party as a single block. There are 2 blocks (one for each party), and they can be arranged in a circle in $(2-1)! = 1! = 1$ way. Within each block, the members can be arranged in $5!$ ways. So, the total number of distinguishable ways is $1 \times 5! \times 5! = 120 \times 120 = \boxed{14400}$. Final Answer: The final answer is 14400. I hope it is correct.",
            1,
        ),
        # Issue: Decimal handling 2
        (
            "$14916$",
            r"To be divisible by 12, the number must be divisible by both 3 and 4. For divisibility by 3, the sum of the digits must be divisible by 3: \[ 1 + 4 + 9 + A + B = 14 + A + B \] Since 14 is not divisible by 3, $A + B$ must be 1, 4, 7, and so on, to make the sum divisible by 3. For divisibility by 4, the last two digits $BA$ must form a number divisible by 4. Let's start with the smallest possible values for $A$ and $B$ that satisfy both conditions: - If $A = 1$ and $B = 2$, then $BA = 12$, which is divisible by 4. - The sum of the digits is $14 + 1 + 2 = 17$, which is not divisible by 3. - If $A = 1$ and $B = 5$, then $BA = 15$, which is not divisible by 4. - If $A = 2$ and $B = 0$, then $BA = 20$, which is divisible by 4. - The sum of the digits is $14 + 2 + 0 = 16$, which is not divisible by 3. - If $A = 2$ and $B = 4$, then $BA = 24$, which is divisible by 4. - The sum of the digits is $14 + 2 + 4 = 20$, which is not divisible by 3. - If $A = 3$ and $B = 1$, then $BA = 31$, which is not divisible by 4. - If $A = 3$ and $B = 4$, then $BA = 34$, which is not divisible by 4. - If $A = 4$ and $B = 0$, then $BA = 40$, which is divisible by 4. - The sum of the digits is $14 + 4 + 0 = 18$, which is divisible by 3. Thus, the smallest possible value of $N$ is $14{,}940$. Final Answer: The final answer is $14{,}916$. I hope it is correct.",
            1,
        ),
        # Issue: Percentage handling
        (
            "$20\\%$",
            r"Let $A$ be the set of adults in Mathopolis. Let $F$ be the set of female adults in Mathopolis. Let $C$ be the set of children in Mathopolis. We know that $|A|/2 = |F|$ and $|F|/2 = |C|$. We also know that $|A| = |F| + |C|$. Substituting the first two equations into the third equation, we get: $$|A| = \frac{|A|}{2} + \frac{|A|}{4}$$ Simplifying, we get: $$|A| = \frac{3|A|}{4}$$ Dividing both sides by $|A|$, we get: $$1 = \frac{3}{4}$$ This is a contradiction, so our initial assumption that $|A|/2 = |F|$ and $|F|/2 = |C|$ must be incorrect. However, we can still solve the problem. Let $|A| = 2x$. Then $|F| = x$ and $|C| = x/2$. The total population of Mathopolis is $|A| + |C| = 2x + x/2 = 5x/2$. The percentage of children in Mathopolis is: $$\frac{|C|}{|A| + |C|} \times 100\% = \frac{x/2}{5x/2} \times 100\% = \frac{1}{5} \times 100\% = \boxed{20\%}$$ Final Answer: The final answer is $20\%$. I hope it is correct.",
            1,
        ),
        # Issue: LaTeX vs expr precedence 2
        (
            "$7.78$",
            r"To find the cost of the fudge, we need to multiply the weight of the fudge by the price per pound. Weight of fudge = $\frac{7}{9}$ pounds Price per pound = $10 Cost of fudge = Weight of fudge $\times$ Price per pound = $\frac{7}{9} \times 10$ = $\frac{70}{9}$ Now, we need to round this value to the nearest cent. $\frac{70}{9} \approx 7.78$ So, Anna pays approximately $7.78 for her fudge. Final Answer: The final answer is 7.78.",
            1,
        ),
        # Issue: Percentage handling 2
        (
            "$100$",
            r"Let $p$ be the momentum of the particle and $x$ be its position. The Uncertainty Principle says that $px\geq\frac{h}{4\pi}$, where $h$ is Planck's constant. Suppose the error in the measurement of the momentum is halved. Then the new error in the measurement of the momentum is $\frac{p}{2}$. Let $E_x$ be the error in the measurement of the position. Then $E_x\geq\frac{h}{4\pi\cdot\frac{p}{2}}=\frac{h}{2\pi p}$. Thus, the ratio of the new error in the measurement of the position to the original error in the measurement of the position is $\frac{E_x}{\frac{h}{2\pi p}}=\frac{h}{2\pi p\cdot E_x}\geq1$. Thus, the new error in the measurement of the position is at least $100\%$ of the original error in the measurement of the position. Final Answer: The final answer is $100$. I hope it is correct.",
            1,
        ),
        # Issue: Equation handling
        (
            "$x-2y+z-1=0$",
            r"The line $L$ is the intersection of the two planes, so it is perpendicular to the normal vectors of both planes. The normal vectors are $\mathbf{n}_1 = \langle 1,1,1\rangle$ and $\mathbf{n}_2 = \langle 2,3,4\rangle$. Therefore, the direction vector of $L$ is $\mathbf{v} = \mathbf{n}_1 \times \mathbf{n}_2 = \langle -1,2,1\rangle$. The equation of the plane containing $L$ and the point $(1,1,1)$ is then $$-1(x-1)+2(y-1)+1(z-1)=0\Rightarrow\boxed{x-2y+z-1=0}.$$ Final Answer: The final answer is $x-2y+z-1=0$. I hope it is correct.",
            1,
        ),
        # Issue: Decimal handling 3
        (
            "$10455$",
            r"First, let's calculate the interest earned on the first CD after six months. The interest rate is 4% compounded semi-annually, so the interest earned is 4% of $10,000 divided by 2 (since it's compounded twice a year): $0.04 \cdot 10,000 \cdot \frac{1}{2} = \boxed{200}$ So, after six months, Dr. Zaius has $10,000 + $200 = $10,200. Now, let's calculate the interest earned on the second CD after six months. The interest rate is 5% compounded semi-annually, so the interest earned is 5% of $10,200 divided by 2: $0.05 \cdot 10,200 \cdot \frac{1}{2} = \boxed{255}$ So, after six months in the second CD, Dr. Zaius has $10,200 + $255 = $10,455. Final Answer: The final answer is $10,455$. I hope it is correct.",
            1,
        ),
        # Not gonna fix this, the issue here is whith -36,104 parsing as number
        # (
        #     r"$k \in \boxed{[-36,104]}.$",
        #     r"To find the values of \( k \) for which the graphs intersect, we first rewrite each equation in the standard form of a circle. Starting with the first equation: \[ x^2 + y^2 = 4 + 12x + 6y. \] Rearrange terms: \[ x^2 - 12x + y^2 - 6y = 4. \] Complete the square for \( x \) and \( y \): \[ (x^2 - 12x) + (y^2 - 6y) = 4. \] For \( x \), complete the square: \[ x^2 - 12x = (x-6)^2 - 36. \] For \( y \), complete the square: \[ y^2 - 6y = (y-3)^2 - 9. \] Substitute back: \[ (x-6)^2 - 36 + (y-3)^2 - 9 = 4. \] Simplify: \[ (x-6)^2 + (y-3)^2 = 49. \] This represents a circle centered at \( (6, 3) \) with radius \( 7 \). Now, consider the second equation: \[ x^2 + y^2 = k + 4x + 12y. \] Rearrange terms: \[ x^2 - 4x + y^2 - 12y = k. \] Complete the square for \( x \) and \( y \): \[ (x^2 - 4x) + (y^2 - 12y) = k. \] For \( x \), complete the square: \[ x^2 - 4x = (x-2)^2 - 4. \] For \( y \), complete the square: \[ y^2 - 12y = (y-6)^2 - 36. \] Substitute back: \[ (x-2)^2 - 4 + (y-6)^2 - 36 = k. \] Simplify: \[ (x-2)^2 + (y-6)^2 = k + 40. \] This represents a circle centered at \( (2, 6) \) with radius \( \sqrt{k+40} \). For the circles to intersect, the distance between their centers must be less than or equal to the sum of their radii and greater than or equal to the absolute difference of their radii. Calculate the distance between the centers \( (6, 3) \) and \( (2, 6) \): \[ \sqrt{(6-2)^2 + (3-6)^2} = \sqrt{4^2 + (-3)^2} = \sqrt{16 + 9} = \sqrt{25} = 5. \] The condition for intersection is: \[ |7 - \sqrt{k+40}| \leq 5 \leq 7 + \sqrt{k+40}. \] First, solve \( 7 - \sqrt{k+40} \leq 5 \): \[ 7 - 5 \leq \sqrt{k+40} \Rightarrow 2 \leq \sqrt{k+40}. \] Square both sides: \[ 4 \leq k + 40 \Rightarrow k \geq -36. \] Next, solve \( 5 \leq 7 + \sqrt{k+40} \): \[ 5 - 7 \leq \sqrt{k+40} \Rightarrow -2 \leq \sqrt{k+40}. \] Since \(\sqrt{k+40} \geq 0\), this inequality is always true. Now, solve \( 5 \leq 7 + \sqrt{k+40} \): \[ 5 - 7 \leq \sqrt{k+40} \Rightarrow -2 \leq \sqrt{k+40}. \] This is always true. Finally, solve \( 5 \geq \sqrt{k+40} - 7 \): \[ 5 + 7 \geq \sqrt{k+40} \Rightarrow 12 \geq \sqrt{k+40}. \] Square both sides: \[ 144 \geq k + 40 \Rightarrow k \leq 104. \] Combining these results, the values of \( k \) for which the circles intersect are: \[ k \in [-36, 104]. \] Final Answer: The interval is \(\boxed{[-36, 104]}\).",
        #     1
        # ),
        (
            r"$[-4,0]$",
            r"Thus, the values of \( k \) for which the quadratic has real roots are: \[ \boxed{[-4, 0]} \]",
            1,
        ),
        # Test interval notation
        (
            r"Factoring the denominator on the left side gives \[ \frac{4x}{(x-5)(x-3)}=\frac{A}{x-3}+\frac{B}{x-5}. \]Then, we multiply both sides of the equation by $(x - 3)(x - 5)$ to get \[ 4x = A(x-5) + B(x-3). \]If the linear expression $4x$ agrees with the linear expression $A(x-5) + B(x-3)$ at all values of $x$ besides 3 and 5, then the two expressions must agree for $x=3$ and $x=5$ as well. Substituting $x = 3$, we get $12 = -2A$, so $A = -6$. Likewise, we plug in $x = 5$ to solve for $B$. Substituting $x = 5$, we get $20 = 2B$, so $B = 10$. Therefore, $(A, B) = \boxed{(-6, 10)}.$",
            r"The solution to the inequality is \( x \in (-6, 10) \). Final Answer: The final answer is \((-6, 10)\). I hope it is correct.",
            1,
        ),
        # Test base conversion
        (
            r"$1112112_3$",
            r"To find the product \( ab \) where \( a = 2012_3 \) and \( b = 201_3 \), we first convert these base-three numbers to base ten. For \( a = 2012_3 \): \[ a = 2 \cdot 3^3 + 0 \cdot 3^2 + 1 \cdot 3^1 + 2 \cdot 3^0 = 2 \cdot 27 + 0 \cdot 9 + 1 \cdot 3 + 2 \cdot 1 = 54 + 0 + 3 + 2 = 59_{10} \] For \( b = 201_3 \): \[ b = 2 \cdot 3^2 + 0 \cdot 3^1 + 1 \cdot 3^0 = 2 \cdot 9 + 0 \cdot 3 + 1 \cdot 1 = 18 + 0 + 1 = 19_{10} \] Now, calculate the product in base ten: \[ ab = 59 \times 19 \] Perform the multiplication: \[ 59 \times 19 = 59 \times (20 - 1) = 59 \times 20 - 59 \times 1 = 1180 - 59 = 1121 \] Next, convert \( 1121_{10} \) to base three. We do this by dividing by 3 and recording the remainders: \[ 1121 \div 3 = 373 \quad \text{remainder } 2 \] \[ 373 \div 3 = 124 \quad \text{remainder } 1 \] \[ 124 \div 3 = 41 \quad \text{remainder } 1 \] \[ 41 \div 3 = 13 \quad \text{remainder } 2 \] \[ 13 \div 3 = 4 \quad \text{remainder } 1 \] \[ 4 \div 3 = 1 \quad \text{remainder } 1 \] \[ 1 \div 3 = 0 \quad \text{remainder } 1 \] Reading the remainders from last to first, we find: \[ 1121_{10} = 1112122_3 \] Thus, the product \( ab \) expressed in the base-three number system is \(\boxed{1112122_3}\).",
            0,
        ),
        (
            r"\(\boxed{\text{C}}\).",
            r"$\boxed{\text{(C)}}.$",
            1,
        ),
        (
            r" So the answer is: \[ \boxed{11111111100} \]",
            r"is $\boxed{11,\! 111,\! 111,\! 100}$",
            1,
        ),
        (
            r" So the answer is: \[ \boxed{32349} \]",
            r"is $\boxed{32,\! 349}$",
            1,
        ),
        (
            r" So the answer is: \[ \boxed{32349} \]",
            r"is $\boxed{32, 349}$",
            0,
        ),
        (
            r"Thus, the domain of the function \( f(x) \) is: \[ \boxed{(2, 12) \cup (12, 102)} \]",
            r"Thus, the answer is $x \in \boxed{(2,12) \cup (12,102)}$",
            1,
        ),
        (
            r"$\text{E}$",
            r"$E$",
            1,
        ),
        (
            r"$\boxed{1}\boxed{2}$",
            r"$\boxed{1,2}$",
            1,
        ),
        (
            r"Setting $x = y = 0,$ we get \[2f(0) = f(0) - 1,\]so $f(0) = -1.$ Setting $y = 1,$ we get \[f(x) + 1 = f(x + 1) - x - 1,\]so \[f(x + 1) - f(x) = x + 2.\]Thus, \begin{align*} f(2) - f(1) &= 1 + 2, \\ f(3) - f(2) &= 2 + 2, \\ f(4) - f(3) &= 3 + 2, \\ &\dots, \\ f(n) - f(n - 1) &= (n - 1) + 2. \end{align*}Adding all the equations, we get \[f(n) - f(1) = 1 + 2 + 3 + \dots + (n - 1) + 2(n - 1) = \frac{(n - 1)n}{2} + 2n - 2 = \frac{n^2 + 3n - 4}{2},\]so \[f(n) = \frac{n^2 + 3n - 2}{2}\]for all positive integers $n.$ Setting $x = -n$ and $y = n,$ where $n$ is a positive integer, we get \[f(-n) + f(n) = f(0) + n^2 - 1.\]Then \[f(-n) = n^2 - f(n) + f(0) - 1 = n^2 - \frac{n^2 + 3n - 2}{2} - 2 = \frac{n^2 - 3n - 2}{2}.\]Thus, the formula \[f(n) = \frac{n^2 + 3n - 2}{2}\]holds for all integers $n.$ We want to solve $f(n) = n,$ or \[\frac{n^2 + 3n - 2}{2} = n.\]Then $n^2 + 3n - 2 = 2n,$ or $n^2 + n - 2 = 0.$ This factors as $(n - 1)(n + 2) = 0,$ so the solutions are $n = \boxed{1,-2}.$",
            r"<think> Okay, so I have this functional equation to solve: \( f(x) + f(y) = f(x + y) - xy - 1 \), and I'm given that \( f(1) = 1 \). I need to find all integers \( n \) such that \( f(n) = n \). Hmm, functional equations can sometimes be tricky, but maybe I can figure this out step by step. First, let me see if I can find an expression for \( f(x) \). Maybe by plugging in specific values for \( x \) and \( y \), I can get some information about \( f \). Let me try setting both \( x \) and \( y \) to 0. That often simplifies things. So if \( x = 0 \) and \( y = 0 \), then the equation becomes: \( f(0) + f(0) = f(0 + 0) - 0 * 0 - 1 \) Simplifying, that's: \( 2f(0) = f(0) - 0 - 1 \) Which simplifies further to: \( 2f(0) = f(0) - 1 \) Subtracting \( f(0) \) from both sides: \( f(0) = -1 \) Alright, so I found that \( f(0) = -1 \). That's a start. Now, maybe I can set \( y = 1 \) because I know \( f(1) = 1 \). Let's set \( y = 1 \) and leave \( x \) arbitrary. Then we get: \( f(x) + f(1) = f(x + 1) - x * 1 - 1 \) Plugging in \( f(1) = 1 \): \( f(x) + 1 = f(x + 1) - x - 1 \) Solving for \( f(x + 1) \): \( f(x + 1) = f(x) + 1 + x + 1 \) So, \( f(x + 1) = f(x) + x + 2 \) Hmm, that's another recursive relationship. So if I can express \( f(x + 1) \) in terms of \( f(x) \), maybe I can find a pattern or a general expression for \( f(x) \). Let me try to unroll this recursion. Let's suppose I start at some point and iterate this. Let's do that starting from \( x = 0 \). First, when \( x = 0 \), as we found earlier, \( f(0) = -1 \). Then, \( f(1) = f(0 + 1) = f(0) + 0 + 2 = -1 + 0 + 2 = 1 \). Well, that checks out because we were given \( f(1) = 1 \). Next, let's find \( f(2) \). Using the same recursion, \( f(2) = f(1 + 1) = f(1) + 1 + 2 = 1 + 1 + 2 = 4 \). Similarly, \( f(3) = f(2 + 1) = f(2) + 2 + 2 = 4 + 2 + 2 = 8 \). Wait a second, let me check that again... So \( f(3) = f(2) + 2 + 2... \) So from \( x = 2 \), do I add 2? Wait, no—wait, actually, the recursion is \( f(x + 1) = f(x) + x + 2 \). So for \( x = 2 \), \( f(3) = f(2) + 2 + 2 \). Yep, that's correct. So continuing... \( f(4) = f(3 + 1) = f(3) + 3 + 2 = 8 + 3 + 2 = 13 \) Hmm, so the values so far are: - f(0) = -1 - f(1) = 1 - f(2) = 4 - f(3) = 8 - f(4) = 13 Wait, let me see if this follows a pattern. Let's list them: n: 0, 1, 2, 3, 4 f(n): -1, 1, 4, 8, 13 Looking at these numbers, let me compute the differences between consecutive terms. f(1) - f(0) = 1 - (-1) = 2 f(2) - f(1) = 4 - 1 = 3 f(3) - f(2) = 8 - 4 = 4 f(4) - f(3) = 13 - 8 = 5 Ah, interesting! Each time, the difference increases by 1. So from f(0) to f(1), the difference is 2, then 3, then 4, then 5... So it seems the differences form an arithmetic sequence starting at 2 with a common difference of 1. If this pattern continues, perhaps the function can be expressed as a quadratic function. Let me test that. Let me assume that \( f(x) \) is a quadratic function of the form \( f(x) = ax^2 + bx + c \). Then maybe I can find coefficients \( a \), \( b \), and \( c \) that satisfy the functional equation. Let me write down that assumption: \( f(x) = ax^2 + bx + c \) Now, plugging this into the functional equation: \( f(x) + f(y) = f(x + y) - xy - 1 \) Substituting, we get: \( ax^2 + bx + c + ay^2 + by + c = a(x + y)^2 + b(x + y) + c - xy - 1 \) Let's simplify both sides. Left side: \( ax^2 + bx + c + ay^2 + by + c = a(x^2 + y^2) + b(x + y) + 2c \) Right side: \( a(x + y)^2 + b(x + y) + c - xy - 1 = a(x^2 + 2xy + y^2) + b(x + y) + c - xy - 1 \) Simplify: \( a x^2 + 2a xy + a y^2 + b x + b y + c - xy - 1 \) Now, set left side equal to right side: \( a(x^2 + y^2) + b(x + y) + 2c = a x^2 + 2a xy + a y^2 + b x + b y + c - xy - 1 \) Let me move all terms to the left side to compare: Left minus right: \( a(x^2 + y^2) + b(x + y) + 2c - [a x^2 + 2a xy + a y^2 + b x + b y + c - xy - 1] = 0 \) Simplify term by term: First, \( a x^2 - a x^2 = 0 \) Similarly, \( a y^2 - a y^2 = 0 \) Then, \( -2a xy \) because one side has a +0xy and the other has -2a xy Wait, wait, let me clarify: Original left side terms: \( a x^2 + a y^2 + b x + b y + 2c \) Subtract right side terms: \( a x^2 + 2a xy + a y^2 + b x + b y + c - xy - 1 \) So, subtracting: \( (a x^2 - a x^2) + (a y^2 - a y^2) + (2c - c) + (b x - b x) + (b y - b y) -2a xy + xy + 1 = 0 \) Simplify: 0 + 0 + c -2a xy + xy + 1 = 0 So combine like terms: c + (-2a + 1)xy + 1 = 0 This must hold for all real numbers x and y. How can the above equation hold for all x and y? The only way is that each coefficient is zero for the variables, and the constants sum to zero. So, coefficients: 1. For xy: (-2a + 1) must be zero. 2. Constants: c + 1 must be zero. So let's write those equations: 1. -2a + 1 = 0 => a = 1/2 2. c + 1 = 0 => c = -1 So, from this, I know \( a = 1/2 \) and \( c = -1 \). Now, what about \( b \)? Let me see, in the left minus right equation, the coefficients for x and y have canceled out because both sides have terms with \( b x \) and \( b y \), subtracting each other. So, there's no need to set anything for \( b \)? Wait, do I have any equation involving \( b \)? It seems in the difference between the left and right sides, the terms involving \( b x \) and \( b y \) have canceled each other out, meaning there is no condition on \( b \) from that step. So, from this examination, it seems that my initial assumption of \( f(x) = ax^2 + bx + c \) with \( a = 1/2 \), \( c = -1 \), and \( b \) is still free. Wait, but maybe we can get another condition using \( f(1) = 1 \). Given that \( f(1) = 1 \), and our general form is \( f(x) = \frac{1}{2}x^2 + b x - 1 \). So, \( f(1) = \frac{1}{2}(1)^2 + b (1) - 1 = \frac{1}{2} + b - 1 = b - \frac{1}{2} \) Given that \( f(1) = 1 \), set up the equation: \( b - \frac{1}{2} = 1 \Rightarrow b = 1 + \frac{1}{2} = \frac{3}{2} \) So, thus, the function can be written as: \( f(x) = \frac{1}{2}x^2 + \frac{3}{2}x - 1 \) Let me write that more neatly: \( f(x) = \frac{1}{2}x^2 + \frac{3}{2}x - 1 \) Simplify: \( f(x) = \frac{1}{2}x^2 + \frac{3}{2}x - 1 \) Well, let's verify if this satisfies the original functional equation to make sure. Compute \( f(x) + f(y) \): \( \frac{1}{2}x^2 + \frac{3}{2}x - 1 + \frac{1}{2}y^2 + \frac{3}{2}y - 1 \) Simplify: \( \frac{1}{2}(x^2 + y^2) + \frac{3}{2}(x + y) - 2 \) Compare with \( f(x + y) - xy - 1 \): First, compute \( f(x + y) \): \( \frac{1}{2}(x + y)^2 + \frac{3}{2}(x + y) - 1 \) Expand: \( \frac{1}{2}(x^2 + 2xy + y^2) + \frac{3}{2}(x + y) - 1 \) Which is: \( \frac{1}{2}x^2 + xy + \frac{1}{2}y^2 + \frac{3}{2}x + \frac{3}{2}y - 1 \) Subtract \( xy + 1 \): \( f(x + y) - xy -1 = \frac{1}{2}x^2 + xy + \frac{1}{2}y^2 + \frac{3}{2}x + \frac{3}{2}y - 1 - xy - 1 \) Simplify: \( \frac{1}{2}x^2 + \frac{1}{2}y^2 + \frac{3}{2}x + \frac{3}{2}y - 2 \) So, \( f(x + y) - xy - 1 = \frac{1}{2}(x^2 + y^2) + \frac{3}{2}(x + y) - 2 \), which is exactly equal to \( f(x) + f(y) \). Therefore, plugging in, it does satisfy the equation. So, that looks good. So we now have an expression for \( f(x) \): \( f(x) = \frac{1}{2}x^2 + \frac{3}{2}x - 1 \) Simplify this further if possible: Combine terms: \( f(x) = \frac{x^2 + 3x - 2}{2} \) Alternatively, it's a quadratic function. Now, I need to find all integers \( n \) such that \( f(n) = n \). So set: \( f(n) = n \) Which is: \( \frac{n^2 + 3n - 2}{2} = n \) Multiply both sides by 2: \( n^2 + 3n - 2 = 2n \) Bring all terms to one side: \( n^2 + 3n - 2 - 2n = 0 \) Simplify: \( n^2 + n - 2 = 0 \) That's a quadratic equation. Let me solve it. Quadratic equation: \( n^2 + n - 2 = 0 \) Using the quadratic formula: \( n = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a} \), where \( a = 1 \), \( b = 1 \), \( c = -2 \). Compute discriminant: \( 1^2 - 4*1*(-2) = 1 + 8 = 9 \) So, \( n = \frac{-1 \pm \sqrt{9}}{2} = \frac{-1 \pm 3}{2} \) Therefore, 1. \( n = \frac{-1 + 3}{2} = \frac{2}{2} = 1 \) 2. \( n = \frac{-1 - 3}{2} = \frac{-4}{2} = -2 \) So the solutions are \( n = 1 \) and \( n = -2 \). Wait, but these are the integer solutions. So the integers n for which \( f(n) = n \) are 1 and -2. Wait, let me double-check with our function \( f(n) \): Given \( f(x) = \frac{x^2 + 3x - 2}{2} \), let me plug in 1: \( f(1) = \frac{1 + 3 - 2}{2} = \frac{2}{2} = 1 \). Correct, since \( f(1) = 1 \) was given. Now, plug in -2: \( f(-2) = \frac{(-2)^2 + 3*(-2) - 2}{2} = \frac{4 - 6 - 2}{2} = \frac{-4}{2} = -2 \). So, indeed, \( f(-2) = -2 \). Looking back at the pattern we saw earlier: We found in the functional equation \( f(0) = -1 \), \( f(1) = 1 \), \( f(2) = 4 \), \( f(3) = 8 \), \( f(4) = 13 \)... These are all positive integers only at integers 1, 4, 8, 13... But, wait, in our case, the only integer solutions where \( f(n) = n \) is at 1 and -2. Wait, let me check with another integer, 2: \( f(2) = 4 \), so that's \( f(2) = 4 \), which is 4, which is not equal to 2. So n=2 is not a solution. Similarly, n=3: \( f(3) = 8 \neq 3 \), n=4: \( f(4)=13 \neq 4 \), etc. What about negative integers? We saw n=-2: f(-2)= -2. So, that's a fixed point. What about n=-1: \( f(-1) = \frac{(-1)^2 + 3*(-1) - 2}{2} = \frac{1 - 3 - 2}{2} = \frac{-4}{2} = -2 \neq -1 \). n=-3: \( f(-3) = \frac{9 + (-9) -2 }{2} = \frac{-2}{2}= -1 \neq -3 \). n=-4: \( f(-4) = \frac{16 + (-12) - 2 }{2} = \frac{2}{2}= 1 \neq -4 \). Hmmm, so n=-2 is a solution. So the only integers n where f(n) = n are n =1 and n=-2. Therefore, the integers are 1 and -2. Wait, but let me think again—is that all? Because sometimes functional equations can have more solutions unless restricted. But in this case, since we derived the general solution of the function, which is quadratic, solving \(f(n) = n\) gives only quadratic equation, which has at most two solutions (and indeed we found two solutions). So, there are exactly two integers: 1 and -2. Wait, so I think I can be confident the solutions are 1 and -2. **Final Answer** The integers are \boxed{-2} and \boxed{1}. </think> Given the functional equation: \[ f(x) + f(y) = f(x + y) - xy - 1 \] for all real numbers \( x \) and \( y \), and knowing that \( f(1) = 1 \), we need to find all integers \( n \) such that \( f(n) = n \). First, by setting \( x = 0 \) and \( y = 0 \), we find: \[ 2f(0) = f(0) - 1 \implies f(0) = -1. \] Next, setting \( y = 1 \) in the functional equation, we get: \[ f(x) + f(1) = f(x + 1) - x - 1. \] Given \( f(1) = 1 \), this simplifies to: \[ f(x + 1) = f(x) + x + 2. \] Using this recursive relationship, we calculate the values of \( f \) at several points: \[ f(1) = 1, \] \[ f(2) = 4, \] \[ f(3) = 8, \] \[ f(4) = 13. \] We observe that the differences between consecutive values form an arithmetic sequence starting from 2 and increasing by 1 each time. This suggests that \( f(x) \) might be a quadratic function. Assuming \( f(x) \) is quadratic, we write it as: \[ f(x) = ax^2 + bx + c. \] Substituting this into the functional equation and equating coefficients, we find: \[ a = \frac{1}{2}, \] \[ c = -1, \] and using \( f(1) = 1 \), we determine: \[ b = \frac{3}{2}. \] Thus, the quadratic function is: \[ f(x) = \frac{1}{2}x^2 + \frac{3}{2}x - 1. \] To find all integers \( n \) such that \( f(n) = n \), we solve: \[ \frac{n^2 + 3n - 2}{2} = n. \] This simplifies to: \[ n^2 + n - 2 = 0. \] Solving this quadratic equation, we find the roots: \[ n = 1 \quad \text{and} \quad n = -2. \] Thus, the integers \( n \) such that \( f(n) = n \) are: \[ \boxed{-2} \quad \text{and} \quad \boxed{1}. \]",
            1,
        ),
    ],
)
def test_math_extraction_additional_cases(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected
