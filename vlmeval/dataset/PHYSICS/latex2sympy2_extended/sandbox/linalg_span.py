from latex2sympy2_extended import latex2sympy
import sys
sys.path.append("..")

latex = "\\begin{pmatrix}1\\\\2\\\\3\\end{pmatrix}"
math = latex2sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\begin{pmatrix}1\\\\2\\\\3\\end{pmatrix},\\begin{pmatrix}4\\\\3\\\\1\\end{pmatrix}"
math = latex2sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "[\\begin{pmatrix}1\\\\2\\\\3\\end{pmatrix},\\begin{pmatrix}4\\\\3\\\\1\\end{pmatrix}]"
math = latex2sympy(latex)
print("latex: %s to math: %s" % (latex, math))

latex = "\\left\\{\\begin{pmatrix}1\\\\2\\\\3\\end{pmatrix},\\begin{pmatrix}4\\\\3\\\\1\\end{pmatrix}\\right\\}"
math = latex2sympy(latex)
print("latex: %s to math: %s" % (latex, math))
