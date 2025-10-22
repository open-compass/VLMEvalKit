from latex2sympy2_extended import latex2sympy
import sys
sys.path.append("..")

# latex = "2\\begin{pmatrix}1&1&1\\\\0&1&1\\\\0&0&1\\end{pmatrix}\\begin{pmatrix}1&1&1\\\\0&1&1\\\\0&0&1\\end{pmatrix}"
latex = "\\frac{a^{2} \\left(3 \\pi - 4 \\sin{\\left(\\pi \\right)} + \\frac{\\sin{\\left(2 \\pi \\right)}}{2}\\right)}{2}"
math = latex2sympy(latex)

print(type(math))
print("latex: %s to math: %s" % (latex, math))
