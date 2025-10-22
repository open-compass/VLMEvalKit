import pytest

from tests.test_all import compare_strings


@pytest.mark.parametrize(
    "gold, pred, expected",
    [
        (
            r"$$\boxed{2}, \boxed{3}, \boxed{5}, \boxed{7}, \boxed{13}$$",
            r"$${2, 3, 5, 7, 13}$$",
            1,
        ),
        (
            r"$$f(y) = y$$",
            r"$$f(y) = y$$",
            1,
        ),
        (
            r"$$-793 < a < 10$$",
            r"$$(-793, 10)$$",
            1,
        ),
        (
            r"$$\text{The midpoints form a regular hexagon}$$",
            r"$$\text{The midpoints form a regular hexagon}$$",
            1,
        ),
        (
            r"$$a_3 = 2, a_{37} = 19$$",
            r"$$\boxed{a_{37} = 19, \quad a_3 = 2}$$",
            1,
        ),
        (
            r"$$(2, 2), (1, 3), (3, 3)$$",
            r"$$\boxed{(2, 2), (1, 3), (3, 3)}$$",
            1,
        ),
        (
            r"$$(1, +\infty)$$",
            r"$$(1, +\infty)$$",
            1,
        ),
        (
            r"$$x = \frac{1}{2}, x = \frac{-1 \pm \sqrt{13}}{4}$$",
            r"$$\boxed{x = \frac{1}{2}, x = \frac{-1 \pm \sqrt{13}}{4}}$$",
            1,
        ),
        (
            r"$$0.06 \text{ yuan}$$",
            r"$$\boxed{0.06 \text{ yuan}}$$",
            1,
        ),
        (
            r"$$(5, 8), (8, 5)$$",
            r"$$\boxed{(x, y) = (5, 8) \quad \text{and} \quad (8, 5)}$$",
            1,
        ),
        (
            r"$$667$$",
            r"$$\boxed{667}$$",
            1,
        ),
        (
            r"$$6 \text{ and } 8$$",
            r"$$\boxed{6 \text{ and } 8}$$",
            1,
        ),
        (
            r"$$(2, 2), (6, 2)$$",
            r"$$\boxed{(2, 2)} \quad \text{and} \quad \boxed{(6, 2)}$$",
            1,
        ),
        (
            r"$$(98, 2), (99, 503)$$",
            r"$$\boxed{(98, 2)} \quad \text{and} \quad \boxed{(99, 503)}$$",
            1,
        ),
        (
            r"$$(3, 3), (1, 1)$$",
            r"$$\boxed{(3, 3)} \text{ for part (a) and } \boxed{(1, 1)}$$",
            1,
        ),
        (
            r"$$k \ge 2$$",
            r"$$\boxed{k \ge 2}$$",
            1,
        ),
        (
            r"$$k = 45, n = 2$$",
            r"$$\boxed{k = 45} \quad \text{and} \quad \boxed{n = 2}$$",
            1,
        ),
        (
            r"$143$",
            r"<think> Alright, let me try to figure out this problem. So, we need to find how many years between 2000 and 3000 inclusive can be written as the sum of two positive integers p and q such that 2p = 5q. The example given is 2016 = 1440 + 576, where 2*1440 = 2880 and 5*576 = 2880, so they are equal. First, I need to understand the relationship between p and q. The equation 2p = 5q implies that p and q are in a specific ratio. Let's solve for one variable in terms of the other. If 2p = 5q, then p = (5/2)q. That means p has to be a multiple of 5/2 times q. Since p and q are positive integers, q must be a multiple of 2 to make p an integer. Let's denote q = 2k, where k is a positive integer. Then p = (5/2)*(2k) = 5k. So, substituting back, the year can be written as p + q = 5k + 2k = 7k. Therefore, the year must be a multiple of 7. Wait, so that simplifies things. If the year must be a multiple of 7, then the problem reduces to finding how many multiples of 7 are there between 2000 and 3000 inclusive. But let me double-check this reasoning. Starting from 2p = 5q, we have p = (5/2)q. Since p must be an integer, q must be even. Let q = 2k, then p = 5k. So, the sum p + q = 5k + 2k = 7k. Therefore, every year that is a multiple of 7 can be expressed in such a way. Conversely, if a year is a multiple of 7, say N = 7k, then we can set q = 2k and p = 5k, which are integers. So, yes, the problem is equivalent to finding the number of multiples of 7 between 2000 and 3000 inclusive. Therefore, I need to count how many numbers divisible by 7 are in the range [2000, 3000]. To do this, I can find the smallest multiple of 7 in this range and the largest multiple of 7, then compute how many terms are in this arithmetic sequence. Let me first find the smallest multiple of 7 greater than or equal to 2000. Dividing 2000 by 7: 2000 ÷ 7 ≈ 285.714. So, 285 * 7 = 1995, which is less than 2000. The next multiple is 286 * 7 = 2002. Therefore, 2002 is the first multiple of 7 in the range. Now, the largest multiple of 7 less than or equal to 3000. Dividing 3000 by 7: 3000 ÷ 7 ≈ 428.571. So, 428 * 7 = 2996. But 428*7=2996, which is less than 3000. The next multiple would be 429*7=3003, which is over 3000. So, the largest multiple is 2996. Now, the number of terms in the arithmetic sequence from 2002 to 2996, with common difference 7. The formula for the number of terms in an arithmetic sequence is ((last term - first term)/common difference) + 1. So, ((2996 - 2002)/7) + 1 = (994/7) + 1. Let me compute 994 divided by 7. 7*142=994. So, 142 + 1 = 143. Therefore, there are 143 multiples of 7 between 2000 and 3000 inclusive. But wait, let me confirm. If we have terms starting at 2002, each term 7 apart, up to 2996. The difference between 2996 and 2002 is 994. Dividing by 7 gives 142, which means there are 142 intervals of 7, hence 143 terms. That seems right. But let me check with another method. The number of multiples of 7 up to n is floor(n/7). So, the number of multiples up to 3000 is floor(3000/7)=428, as 7*428=2996. The number of multiples up to 1999 is floor(1999/7)=285, since 7*285=1995. So, the number of multiples between 2000 and 3000 inclusive is 428 - 285. But wait, 428 counts up to 2996, which is within 3000. However, 285 counts up to 1995, so the next multiple is 2002. So, the number of multiples from 2002 to 2996 is 428 - 285 = 143. But wait, 428 - 285 = 143? Let me compute 428 - 285. 428 - 200 = 228, 228 - 85 = 143. Yes. So that's 143. However, is 2002 included? Because 285 counts up to 1995, so 1995 is the 285th multiple, then the next one is 2002 which is 286th. So, from 286th to 428th multiple, how many terms? 428 - 286 + 1 = 143. Yes, because (428 - 286) = 142, plus 1 is 143. So that confirms the answer is 143. But wait, the problem states \"between 2000 and 3000 inclusive\". However, 2002 is the first multiple of 7 in that range. So, if we need to include 2000, but 2000 is not a multiple of 7, so the count remains 143. Therefore, the answer is 143. But let me check the example given. 2016 is in the range. 2016 divided by 7 is 288, so 2016 is the 288th multiple of 7. So, 288 - 285 = 3, but according to the previous count, 428 - 285 = 143. But 2016 is within the range. So, perhaps my initial calculation is correct. Wait, 428 - 285 = 143. But if the first multiple is 2002 (286th multiple), and the last is 2996 (428th multiple), then the number of terms is 428 - 286 + 1 = 143. So, yes. So, 143 is the answer. Wait, but let me test with smaller numbers. Suppose between 7 and 14 inclusive. How many multiples of 7? 7,14. So two numbers. Applying the formula: floor(14/7)=2, floor(6/7)=0, so 2-0=2. Correct. Similarly, between 15 and 21. The multiples are 21. So floor(21/7)=3, floor(14/7)=2, 3-2=1. Correct. So, the formula works when the upper bound is a multiple. If upper bound is not a multiple, like between 15 and 20. The multiples are 21 is beyond, so 14 is the last multiple below 20. Wait, no, 14 is 14, 21 is 21. Wait, between 15 and 20, there are no multiples of 7. So floor(20/7)=2, floor(14/7)=2, 2-2=0. Correct. So, formula works. So in the original problem, floor(3000/7)=428, floor(1999/7)=285. So, 428-285=143. Therefore, the answer is 143. But wait, the example given is 2016, which is 7*288. So, 288 is within the 286 to 428 range. 288-285=3, but in the total count, it's 143. So, all multiples from 2002 up to 2996. So, 143 years. Therefore, the answer is 143. So, \boxed{143} **Final Answer** \boxed{143} </think> To determine how many years between 2000 and 3000 inclusive can be written as the sum of two positive integers \( p \) and \( q \) such that \( 2p = 5q \), we start by analyzing the relationship between \( p \) and \( q \). Given \( 2p = 5q \), we can express \( p \) in terms of \( q \): \( p = \frac{5}{2}q \). For \( p \) and \( q \) to be integers, \( q \) must be a multiple of 2. Let \( q = 2k \), then \( p = 5k \). The sum \( p + q = 5k + 2k = 7k \), indicating the year must be a multiple of 7. Next, we need to count the multiples of 7 between 2000 and 3000 inclusive. 1. **Finding the smallest multiple of 7 ≥ 2000**: \[ \left\lceil \frac{2000}{7} \right\rceil = 286 \quad \text{(since } 286 \times 7 = 2002\text{)} \] 2. **Finding the largest multiple of 7 ≤ 3000**: \[ \left\lfloor \frac{3000}{7} \right\rfloor = 428 \quad \text{(since } 428 \times 7 = 2996\text{)} \] 3. **Counting the number of multiples of 7 between 2002 and 2996**: \[ \text{Number of terms} = 428 - 286 + 1 = 143 \] Thus, the number of years between 2000 and 3000 inclusive that can be written as the sum of two positive integers \( p \) and \( q \) such that \( 2p = 5q \) is \(\boxed{143}\).",
            1,
        ),
        (
            r"$$143$$",
            r"$$\boxed{20}$$ \boxed{143}",
            1,
        ),
        (
            r"$${1,2,3}$$",
            r"\boxed{1} and \boxed{2} and \boxed{3}",
            1,
        ),
        (
            r"$${1,2,3}$$",
            r"\boxed{1},\boxed{2},\boxed{3}",
            1,
        ),
        # (
        #     r"$$x+z=1$$",
        #     r"$$1$$",
        #     0,
        # ),
        (
            r"$$|AB|=1$$",
            r"$$1$$",
            1,
        ),
        (
            r"$$A:B=1$$",
            r"$$1$$",
            1,
        ),
        (
            r"$$f(x)=1$$",
            r"$$1$$",
            1,
        ),
        (
            r"$x_{1}=10^{\frac{-5+\sqrt{13}}{6}},\quadx_{2}=10^{\frac{-5-\sqrt{13}}{6}}$",
            r"$\boxed{10^{\frac{\sqrt{13} - 5}{6}}} \quad \text{and} \quad \boxed{10^{-\frac{5 + \sqrt{13}}{6}}}$",
            1,
        ),
        (
            r"$y_{1}=-2 x^{2}+4 x+3, y_{2}=3 x^{2}+12 x+10$",
            r"\($y_1 = \boxed{-2(x - 1)^2 + 5} \) and \( y_2 = \boxed{3(x + 2)^2 - 2} \) ",
            1,
        ),
        (
            r"$x_{1}=\frac{1}{2}+\frac{31\sqrt{5}}{216},\quadx_{2}=\frac{1}{2}-\frac{31\sqrt{5}}{216}$",
            r"$\boxed{\dfrac{108 + 31\sqrt{5}}{216}} \quad \text{and} \quad \boxed{\dfrac{108 - 31\sqrt{5}}{216}}$",
            1,
        ),
        (
            r"$x_{1}=10^{\frac{-5+\sqrt{13}}{6}},\quadx_{2}=10^{\frac{-5-\sqrt{13}}{6}}$",
            r"$\boxed{10^{\frac{\sqrt{13} - 5}{6}}} \quad \text{and} \quad \boxed{10^{-\frac{5 + \sqrt{13}}{6}}}$",
            1,
        ),
        (
            r"\boxed{1} papers and \boxed{2} bins",
            r"$\boxed{1,2}$",
            1,
        ),
        (
            r"\boxed{1} no no no no no \boxed{2}",
            r"$\boxed{1,2}$",
            0,
        ),
    ],
)
def test_open_thoughts(gold, pred, expected):
    assert compare_strings(gold, pred, match_types=["latex", "expr"]) == expected
