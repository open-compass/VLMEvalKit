from sympy import And as SympyAnd
from sympy.core.sympify import sympify

class And(SympyAnd):
    """
    Patched version of And that keeps the _unsorted_args attribute
    """
    def __new__(cls, *args, **kwargs):
        args = [sympify(arg) for arg in args]
        obj = super().__new__(cls, *args, **kwargs)
        obj._unsorted_args = args
        return obj


