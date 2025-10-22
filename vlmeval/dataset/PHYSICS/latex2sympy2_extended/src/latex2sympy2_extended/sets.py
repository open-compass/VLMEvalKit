from sympy import S, Basic, Set, ordered, sympify
from sympy.sets import FiniteSet as SympyFiniteSet
from sympy.core.parameters import global_parameters

class FiniteSet(SympyFiniteSet):
    """
    FiniteSet which keeps the _unsorted_args attribute, only available till the first evaluation
    """
    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if evaluate:
            args = list(map(sympify, args))

            if len(args) == 0:
                return S.EmptySet
        else:
            args = list(map(sympify, args))
        unsorted_args = args

        # keep the form of the first canonical arg
        dargs = {}
        for i in reversed(list(ordered(args))):
            if i.is_Symbol:
                dargs[i] = i
            else:
                try:
                    dargs[i.as_dummy()] = i
                except TypeError:
                    # e.g. i = class without args like `Interval`
                    dargs[i] = i
        _args_set = set(dargs.values())
        args = list(ordered(_args_set, Set._infimum_key))
        obj = Basic.__new__(cls, *args)
        obj._args_set = _args_set
        obj._unsorted_args = unsorted_args
        return obj
