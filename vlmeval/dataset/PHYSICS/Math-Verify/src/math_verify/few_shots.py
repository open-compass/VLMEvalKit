GSM8K_FEW_SHOTS = [
    {
        "question": (
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest to her neighbors for $2 per egg. How much in dollars does she make per day?"
        ),
        "answer": (
            "Janet's ducks lay 16 eggs per day\nShe eats 3 eggs for breakfast\nThis leaves 16 - 3 = 13 eggs to sell\nEach egg sells for $2\nSo she makes 13 * $2 = $26 per day\nThe final answer is $26. I hope it is correct."
        ),
    },
    {
        "question": (
            "A contractor quotes a job at $5,400. He needs 3 workers who each make $20 per hour. The job takes 40 hours. How much profit does he make?"
        ),
        "answer": (
            "Total labor cost = 3 workers * $20 per hour * 40 hours = $2,400\nRevenue from job = $5,400\nProfit = Revenue - Cost\nProfit = $5,400 - $2,400 = $3,000\nThe final answer is $3,000. I hope it is correct."
        ),
    },
    {
        "question": (
            "Sam has 5 times as many marbles as John. John has 3 less marbles than Steve. If Steve has 8 marbles, how many marbles does Sam have?"
        ),
        "answer": (
            "Steve has 8 marbles\nJohn has 3 less than Steve, so John has 8 - 3 = 5 marbles\nSam has 5 times as many as John\nSo Sam has 5 × 5 = 25 marbles\nThe final answer is 25. I hope it is correct."
        ),
    },
    {
        "question": (
            "Tom is baking cookies. Each batch requires 2.5 cups of flour and makes 12 cookies. If Tom wants to make 60 cookies, how many cups of flour will he need?"
        ),
        "answer": (
            "One batch makes 12 cookies and needs 2.5 cups of flour\nTo make 60 cookies, Tom needs 60 ÷ 12 = 5 batches\nEach batch needs 2.5 cups of flour\nTotal flour needed = 5 × 2.5 = 12.5 cups\nThe final answer is 12.5. I hope it is correct."
        ),
    },
]

MATH_HARD_FEW_SHOTS = [
    {
        "question": (
            "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$."
        ),
        "answer": (
            "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so $x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nThe final answer is $[2,5)$. I hope it is correct."
        ),
    },
    {
        "question": (
            "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$"
        ),
        "answer": (
            "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = (2)(12) = \\boxed{24}.$\nThe final answer is $24$. I hope it is correct."
        ),
    },
    {
        "question": (
            "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?"
        ),
        "answer": (
            "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20=480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\n\\Rightarrow\\qquad n&=480/30=\\boxed{16}\n\\end{align*}\nThe final answer is $16$. I hope it is correct."
        ),
    },
    {
        "question": (
            "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero."
        ),
        "answer": (
            "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nThe final answer is $-\\frac{2}{3}$. I hope it is correct."
        ),
    },
]
