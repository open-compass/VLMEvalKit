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

# Heavily inspired by https://github.com/QwenLM/Qwen2.5-Math and https://github.com/huggingface/lm-evaluation-harness
import logging
import re
from itertools import product
from sympy.logic.boolalg import BooleanTrue
from latex2sympy2_extended import is_expr_of_only_symbols
from latex2sympy2_extended.logic import And
from latex2sympy2_extended.sets import FiniteSet
from sympy import (
    Abs,
    Basic,
    E,
    Eq,
    Float,
    GreaterThan,
    Interval,
    LessThan,
    MatrixBase,
    MatrixExpr,
    Mul,
    Number,
    Rational,
    Set,
    StrictGreaterThan,
    StrictLessThan,
    Symbol,
    Tuple,
    default_sort_key,
    nan,
    ordered,
    simplify,
    solve,
    zoo,
)
from sympy import FiniteSet as SympyFiniteSet
from sympy.core.function import UndefinedFunction
from sympy.core.relational import Relational

from math_verify.errors import TimeoutException
from math_verify.utils import timeout

logger = logging.getLogger(__name__)

TIMEOUT_WARNING_SHOWN = False


INVERSE_RELATIONS = {
    GreaterThan: LessThan,
    LessThan: GreaterThan,
    StrictGreaterThan: StrictLessThan,
    StrictLessThan: StrictGreaterThan,
    Eq: Eq,
}


def safe_sympy_doit(a: Basic | MatrixBase):
    """Safely execute doit() on a sympy expression, catching exceptions.
      Doit in sympy will evaluate expressions it will pass the expression tree and evluate nodes.
      For example for 1+1+1 it will evaluate the additions and return 3. One issue with it is that it maybe
      evaluates too much as integrals will also be evaluated.

      As we are using latex2sympy2_extended, evaluates are

    Args:
        a: A sympy Basic or MatrixBase expression to evaluate

    Returns:
        The result of a.doit() if successful, otherwise returns the original expression
    """
    try:
        return a.doit()
    except Exception:
        pass
    return a


def is_atomic_or_pct_atomic(expr: Basic | MatrixBase, atomic_type: type) -> bool:
    """Check if expression is either an atomic type or percentage atomic type.

    Args:
        expr: The sympy expression to check
        atomic_type: The atomic type to check for

    Returns:
        True if expr is atomic_type or percentage atomic type, False otherwise
    """
    return isinstance(expr, atomic_type) or (
        # Check for percentage representation: latex2sympy_extended converts "X%" into X*Rational(1,100)
        # So we detect percentages by looking for this multiplication structure
        isinstance(expr, Mul)
        and len(expr.args) == 2
        and expr.args[1] == Rational(1, 100)
        and isinstance(expr.args[0], atomic_type)
    )


def sympy_numeric_eq(
    a: Basic | MatrixBase,
    b: Basic | MatrixBase,
    float_rounding: int,
    numeric_precision: int,
):
    """Compare two sympy expressions numerically with given precision.

    Args:
        a: First sympy expression
        b: Second sympy expression
        precision: Number of decimal places to compare

    Returns:
        True if expressions are numerically equal within precision, False otherwise
    """
    # Only do this when one of the two is a float, in other cases use symbolic equality as this could lead to false positives
    # E.g we want 1/3 == 0.333333 to work
    if isinstance(a, (MatrixBase, MatrixExpr)) and isinstance(
        b, (MatrixBase, MatrixExpr)
    ):
        a = safe_sympy_doit(a)
        b = safe_sympy_doit(b)
        # If we have matrices and one of them is only made of floats, we can use the same logic as above
        if (
            isinstance(a, (MatrixBase))
            and isinstance(b, (MatrixBase))
            and a.shape == b.shape
        ):
            return all(
                sympy_numeric_eq(a_elem, b_elem, float_rounding, numeric_precision)
                for a_elem, b_elem in zip(a.flat(), b.flat(), strict=False)
            )

    # Ensure this also works for percentage numbers so that 0.333333% = 0.33333333333 with precision 4
    elif is_atomic_or_pct_atomic(a, Number) or is_atomic_or_pct_atomic(b, Number):
        # If one of them is a float or a negative atomic number, we can try to use precision
        if is_atomic_or_pct_atomic(a, Float) or is_atomic_or_pct_atomic(b, Float):
            a = safe_sympy_doit(a)
            b = safe_sympy_doit(b)
            # Now if both are numbers, we can use precision
            if isinstance(a, (Number)) and isinstance(b, (Number)) and a.round(float_rounding) == b.round(float_rounding):
                return True
            elif safe_sympy_doit(a) == safe_sympy_doit(b):
                return True
    try:
        diff_ratio = Abs((a - b) / a).evalf(chop=True, n=numeric_precision)
        if diff_ratio.free_symbols:
            return False
        comparison = diff_ratio <= 10 ** -numeric_precision
        simplified = simplify(comparison)
        if simplified == BooleanTrue():
            return True
    except Exception:
        pass

    return False


def sympy_symbolic_eq(a: Basic | MatrixBase, b: Basic | MatrixBase) -> bool:
    """Compare two sympy expressions symbolically.

    Args:
        a: First sympy expression
        b: Second sympy expression

    Returns:
        True if expressions are symbolically equal, False otherwise
    """
    try:
        a_b_diff = simplify((a - b))  # type: ignore
        if isinstance(a_b_diff, MatrixBase) and a_b_diff.is_zero_matrix:
            return True
        elif isinstance(a_b_diff, Basic) and a_b_diff.is_zero:
            return True
    except Exception:
        pass

    return False


def sympy_deep_compare_set_and_tuple(
    gold: SympyFiniteSet | Tuple,
    pred: SympyFiniteSet | Tuple,
    float_rounding: int,
    numeric_precision: int,
) -> bool:
    """Compare two finite sets by comparing each element with given precision.

    Args:
        a: First finite set
        b: Second finite set
        precision: Number of decimal places to compare

    Returns:
        True if sets contain equal elements within precision, False otherwise

    Note: in order to fully support finite sets, we should ideally do kartesian product comparison
    but this is not implemented yet. We kinda hope sympy will order the elements.
    """

    def unwrap_eq(s):
        if is_assignment_relation(s):
            return take_last_relation(s).rhs
        return s

    def sort_key(x):
        try:
            return default_sort_key(unwrap_eq(x).evalf())
        except Exception:
            return default_sort_key(unwrap_eq(x))

    # This ensures it works for {1/3} and {0.333333}
    if len(gold) == len(pred):
        if isinstance(gold, SympyFiniteSet):
            gold_args = list(ordered(gold.args, keys=sort_key, default=False))
            pred_args = list(ordered(pred.args, keys=sort_key, default=False))

        elif isinstance(gold, Tuple) and isinstance(pred, FiniteSet):
            # We treat the pred as tuple too
            pred_args = pred._unsorted_args
            gold_args = gold.args

        elif isinstance(pred, SympyFiniteSet):
            pred_args = list(ordered(pred.args, keys=sort_key, default=False))
            gold_args = gold.args
        else:
            gold_args = gold.args
            pred_args = pred.args

        return all(
            sympy_expr_eq(a, b, float_rounding, numeric_precision)
            for a, b in zip(gold_args, pred_args, strict=False)
        )

    return False


def sympy_compare_interval(
    a: Interval, b: Interval, float_rounding: int, numeric_precision: int
) -> bool:
    """Compare two intervals.

    Args:
        a: First interval
        b: Second interval
        precision: Number of decimal places to compare endpoints

    Returns:
        True if intervals are equal, False otherwise
    """
    return (
        a.left_open == b.left_open
        and a.right_open == b.right_open
        and sympy_expr_eq(a.start, b.start, float_rounding, numeric_precision)
        and sympy_expr_eq(a.end, b.end, float_rounding, numeric_precision)
    )


def sympy_solve_and_compare(
    gold: Relational, pred: Relational, float_rounding: int, numeric_precision: int
) -> bool:
    solved_gold = list(ordered(solve(gold, gold.free_symbols)))
    solved_pred = list(ordered(solve(pred, pred.free_symbols)))

    if not solved_gold or not solved_pred:
        return False

    if isinstance(gold, Eq) and isinstance(pred, Eq):
        try:
            return all(
                # 分两种情况：如果g/p是dict，就比较items；否则直接比较g和p本身
                all(
                    g_k == p_k and sympy_expr_eq(g_v, p_v, float_rounding, numeric_precision)
                    for (g_k, g_v), (p_k, p_v) in zip(sorted(g.items()), sorted(p.items()), strict=False)
                ) if isinstance(g, dict) and isinstance(p, dict)
                else sympy_expr_eq(g, p, float_rounding, numeric_precision)
                for g, p in zip(solved_gold, solved_pred, strict=False)
            )
        except Exception as e:
            print(f"[Warning] sympy_solve_and_compare failed: {e}")
            return False
    else:
        return sympy_expr_eq(solved_gold, solved_pred, float_rounding, numeric_precision)


def sympy_compare_relational(
    gold: Relational | And,
    pred: Relational | And,
    float_rounding: int,
    numeric_precision: int,
) -> bool:
    """Compare two relational expressions.

    Args:
        gold: First relational expression
        pred: Second relational expression
        precision: Number of decimal places to compare

    Returns:
        True if relations are equivalent, False otherwise
    """

    if isinstance(gold, And) and isinstance(pred, And):
        return all(
            sympy_compare_relational(g, p, float_rounding, numeric_precision)
            for g, p in zip(gold._unsorted_args, pred._unsorted_args, strict=False)
        )

    elif not isinstance(gold, Relational) or not isinstance(pred, Relational):
        return False

    # Helper to check if expressions are equivalent when flipped
    def are_flipped_inequalities_equal(a: Relational, b: Relational) -> bool:
        try:
            return sympy_expr_eq(
                a.lhs - a.rhs, b.rhs - b.lhs, float_rounding, numeric_precision
            )  # type: ignore
        except Exception:
            pass
        return False

    # Same type of relation (e.g. both <= or both >=)

    try:
        if type(gold) is type(pred) and sympy_expr_eq(
            gold.lhs - gold.rhs, pred.lhs - pred.rhs, float_rounding, numeric_precision
        ):  # type: ignore
            return True
    except Exception:
        pass

    # Check flipped inequalities (a <= b equals b >= a)
    if INVERSE_RELATIONS[type(gold)] is type(pred) and are_flipped_inequalities_equal(  # type: ignore
        gold, pred
    ):
        return True

    if sympy_solve_and_compare(gold, pred, float_rounding, numeric_precision):
        return True

    return False


def sympy_str_eq(a: Basic | MatrixBase, b: Basic | MatrixBase) -> bool:
    """Compare two sympy expressions by string representation.

    Args:
        a: First sympy expression
        b: Second sympy expression

    Returns:
        True if string representations are equal, False otherwise
    """
    # We can't evaluate nan or zoo
    if a == nan or a == zoo:
        raise ValueError("Can't evaluate nan or zoo")
    try:
        return a == b
    except Exception:
        pass
    return False


def sympy_compare_sets(
    gold: Set | Basic | MatrixBase | Tuple,
    pred: Set | Basic | MatrixBase | Tuple,
    float_rounding: int,
    numeric_precision: int,
) -> bool:
    """Compare two sympy sets for equality using multiple methods.

    Args:
        gold: First sympy set (expected)
        pred: Second sympy set (predicted)
        precision: Number of decimal places to compare

    Returns:
        True if sets are equal by any comparison method, False otherwise
    """
    # Convert non-sets to singleton sets
    a_set = gold if isinstance(gold, (Set, Tuple)) else SympyFiniteSet(gold)
    b_set = pred if isinstance(pred, (Set, Tuple)) else SympyFiniteSet(pred)

    # If both are intervals, use interval comparison
    if isinstance(a_set, Interval) and isinstance(b_set, Interval):
        return sympy_compare_interval(a_set, b_set, float_rounding, numeric_precision)

    # Try direct set equality
    if a_set == b_set:
        return True

    # If both are sets, check if they are equal
    try:
        if (
            isinstance(a_set, Set)
            and isinstance(b_set, Set)
            and a_set.symmetric_difference(b_set).is_empty
        ):
            return True
    except Exception:
        pass

    # For finite sets, compare elements
    if isinstance(a_set, (SympyFiniteSet, Tuple)) and isinstance(
        b_set, (SympyFiniteSet, Tuple)
    ):
        return sympy_deep_compare_set_and_tuple(
            a_set, b_set, float_rounding, numeric_precision
        )

    # Because (1,2) is parsed as Interval(1,2,left_open=True,right_open=True), it could have that the
    # correct is (1,2) and predicted is 1,2, which is parsed as Set(1,2)
    if isinstance(a_set, Interval) and isinstance(b_set, (SympyFiniteSet, Tuple)):
        if a_set.is_open and len(b_set) == 2:
            return sympy_deep_compare_set_and_tuple(
                Tuple(a_set.start, a_set.end), b_set, float_rounding, numeric_precision
            )

    if isinstance(b_set, Interval) and isinstance(a_set, (SympyFiniteSet, Tuple)):
        if b_set.is_open and len(a_set) == 2:
            return sympy_deep_compare_set_and_tuple(
                a_set, Tuple(b_set.start, b_set.end), float_rounding, numeric_precision
            )

    return False


def sympy_compare_symbols(gold: Basic | MatrixBase, pred: Basic | MatrixBase) -> bool:
    """Compare two sympy expressions where at least one is a Symbol.

    Handles special cases:
    - One is Symbol and other is E (limitation of parsed expressions)
    - One is multiplication of symbols and other is single symbol (concatenated comparison)

    Args:
        gold: First sympy expression (expected)
        pred: Second sympy expression (predicted)
        precision: Number of decimal places to compare

    Returns:
        True if expressions are equal by any comparison method, False otherwise
    """
    # Handle E vs symbol case
    if (isinstance(gold, Symbol) and gold.name.lower() == "e" and pred == E) or (
        isinstance(pred, Symbol) and pred.name.lower() == "e" and gold == E
    ):
        return True

    # Handle multiplication of symbols vs single symbol, because parsing return $abc$ -> abc
    # We also handle E as it's a symbol, because E will be always parsed as exp
    if (
        isinstance(gold, Symbol)
        and isinstance(pred, Mul)
        and all(arg == E or isinstance(arg, (Symbol)) for arg in pred.args)
    ):
        concat_pred = "".join(
            arg.name if isinstance(arg, Symbol) else "e" for arg in pred.args
        )
        return gold.name.lower() == concat_pred.lower()

    if (
        isinstance(pred, Symbol)
        and isinstance(gold, Mul)
        and all(arg == E or isinstance(arg, (Symbol)) for arg in gold.args)
    ):
        concat_gold = "".join(
            arg.name if isinstance(arg, Symbol) else "e" for arg in gold.args
        )
        return pred.name.lower() == concat_gold.lower()

    # Simple
    if isinstance(gold, Symbol) and isinstance(pred, Symbol):
        g_name = gold.name
        p_name = pred.name
        if len(p_name) > 1:
            p_name = p_name.lower()
        if len(g_name) > 1:
            g_name = g_name.lower()
        return g_name == p_name

    return False


def is_relation(expr: Basic | MatrixBase) -> bool:
    """Check if an expression is a relational expression.

    Args:
        expr: The expression to check
    Returns:
        bool: True if expr is a relational expression or And of relations, False otherwise
    """
    if isinstance(expr, Relational):
        return True

    if isinstance(expr, And) and len(expr._unsorted_args) > 0:
        return all(isinstance(arg, Relational) for arg in expr._unsorted_args)

    return False


def is_equation(expr: Basic | MatrixBase) -> bool:
    """Check if an expression is an equation.

    Args:
        expr: The expression to check
    Returns:
        bool: True if expr is an equation, False otherwise
    """
    if isinstance(expr, Eq):
        return True

    if isinstance(expr, And) and len(expr._unsorted_args) > 0:
        return all(isinstance(arg, Eq) for arg in expr._unsorted_args)

    return False


def is_assignment_relation(expr: Basic | MatrixBase) -> bool:
    """Check if an expression is an assignment relation. E.g a=1

    Args:
        expr: The expression to check
    Returns:
        bool: True if expr is a relational expression or And of relations, False otherwise
    """
    if isinstance(expr, Eq) and is_expr_of_only_symbols(expr.lhs):
        return True

    if isinstance(expr, And) and len(expr._unsorted_args) > 0:
        return all(
            isinstance(arg, Eq) for arg in expr._unsorted_args
        ) and is_expr_of_only_symbols(expr._unsorted_args[0].lhs)

    return False


def take_last_relation(expr: And | Relational) -> Relational:
    """Take the last relation from an And expression."""
    if isinstance(expr, And):
        return take_last_relation(expr._unsorted_args[-1])
    return expr


def take_first_relation(expr: And | Relational) -> Relational:
    """Take the first relation from an And expression."""
    if isinstance(expr, And):
        return expr._unsorted_args[0]
    return expr


def unwrap_fcs(expr: Basic | MatrixBase) -> Basic | MatrixBase:
    """Unwrap function calls to their arguments.

    For example, Function('f')(x) becomes Symbol('f_x')

    Args:
        expr: The expression to unwrap

    Returns:
        The unwrapped expression with functions replaced by concatenated symbols
    """
    # Base case - not a Basic type
    if not isinstance(expr, Basic):
        return expr

    # Handle function case
    if hasattr(expr, "func") and isinstance(expr.func, UndefinedFunction):
        # Get function name and arguments
        func_name = expr.func.__name__
        # Recursively unwrap arguments before converting to string
        unwrapped_args = [str(unwrap_fcs(arg)) for arg in expr.args]
        # Create new symbol by concatenating function name and args
        return Symbol(f"{func_name}_{'_'.join(unwrapped_args)}")

    # Recursively unwrap all arguments
    try:
        new_args = [unwrap_fcs(arg) for arg in expr.args]
        if new_args:
            return expr.func(*new_args)
    except Exception:
        pass

    return expr


def sympy_expr_eq(
    gold: Basic | MatrixBase,
    pred: Basic | MatrixBase,
    float_rounding: int,
    numeric_precision: int,
    strict: bool = True,
) -> bool:
    """Compare two sympy expressions for equality using multiple methods.

    Args:
        gold: First sympy expression (expected)
        pred: Second sympy expression (predicted)
        precision: Number of decimal places to compare
        strict: If true, variables do matter otherwise they don't

    Returns:
        True if expressions are equal by any comparison method, False otherwise
    """

    # This ensures that f(x) == f(y) is true
    if not strict:
        try:
            gold_variables = gold.free_symbols
            pred_variables = pred.free_symbols
            if len(gold_variables) == len(pred_variables):
                pred = pred.subs(
                    list(zip(pred_variables, gold_variables, strict=False))
                )
        except Exception:
            pass

    # If both are assigments, we don't want to unwrap them, so that x=1 != y=1
    # But if one is assignment and other is equation, we want to unwrap both

    # We always want to truncate if it's assignment, assignment

    is_gold_assignment = is_assignment_relation(gold)
    is_pred_assignment = is_assignment_relation(pred)
    is_gold_equation = is_equation(gold)
    is_pred_equation = is_equation(pred)

    # Truncate equations chains in case of assignment, this doesn't change any of the above values,
    # so no need to recompute them
    if is_gold_assignment:
        gold = Eq(
            take_first_relation(gold).lhs, take_last_relation(gold).rhs, evaluate=False
        )
    if is_pred_assignment:
        pred = Eq(
            take_first_relation(pred).lhs, take_last_relation(pred).rhs, evaluate=False
        )

    # We follow what the gold format is
    # 1 and 9=1 -> 1,1
    if is_pred_equation and not is_gold_equation:
        # Unwrap pred
        pred = take_last_relation(pred).rhs

    # We respect what the pred format is only if the gold is assignment so that x=1 and 1 -> 1,1, but not 2x + z = 1 and 1 -> 1,1
    elif is_gold_assignment and not is_pred_equation:
        gold = take_last_relation(gold).rhs

    if is_relation(gold) and isinstance(pred, Set):
        # This is to ensure that 1 < x < 2 equals (-oo, 1) U (2, oo)
        # We also unwrap the functions because othewise it creates some conditional set based on the function name
        try:
            gold = unwrap_fcs(gold).as_set()
        except Exception:
            pass
    
    # Start with simple str and expr comparisson as it's the fastest
    # str comparison is better, than simple eq, because it will also handle missarangments
    if sympy_str_eq(gold, pred):
        return True
    
    # Support for equations
    if is_relation(gold) and is_relation(pred):
        return sympy_compare_relational(gold, pred, float_rounding, numeric_precision)

    elif isinstance(gold, (Set, Tuple)) or isinstance(pred, (Set, Tuple)):
        return sympy_compare_sets(gold, pred, float_rounding, numeric_precision)
    
    # Handles $\text{answer}$ == $answer$, one is symbol, is multiplication of symbols (a*n*s*w*e*r)
    elif isinstance(gold, Symbol) or isinstance(pred, Symbol):
        return sympy_compare_symbols(gold, pred)

    elif isinstance(gold, (Basic, MatrixBase)) and isinstance(
        pred, (Basic, MatrixBase)
    ):
        # Mostly so that 0.333333 = 1/3
        if sympy_numeric_eq(gold, pred, float_rounding, numeric_precision):
            return True
        # Then try symbolic equality
        if sympy_symbolic_eq(gold, pred):
            return True

    return False


complex_number_pattern = re.compile(
    r"""
    # Complex number indicators
    \\mathbb\{C\}|        # Complex number set ℂ
    \\i\b|                # Complex i
    \bi\b|                # Standalone i
    \\text\{i\}|          # Text i
    \\mathrm\{i\}|        # Roman i
    \\imath\b|            # Alternative i notation

    # Matrix operations
    \\det|                # Determinant
    \\operatorname\{tr\}| # Trace
    \\operatorname\{rank\}| # Rank
    \\text\{rank\}|
    \\arg\{|              # Complex argument
    \\Re\{|               # Real part
    \\Im\{|               # Imaginary part
    \\operatorname\{Re\}| # Real part alternate
    \\operatorname\{Im\}| # Imaginary part alternate
    \\text\{Re\}|         # Real part text
    \\text\{Im\}          # Imaginary part text
""",
    re.VERBOSE,
)


def should_treat_as_complex(latex_str: str) -> bool:
    """
    Returns True if the latex string likely contains complex numbers, matrices, or vectors.
    """

    return bool(complex_number_pattern.search(latex_str))


def verify(
    gold: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    target: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
    timeout_seconds: int | None = 5,
) -> bool:
    """Verifies if the target expression matches the gold expression using multiple comparison strategies.

    This function implements a comprehensive comparison system for mathematical expressions,
    handling various types of mathematical objects (numbers, expressions, sets, matrices, etc.)
    with multiple fallback strategies.

    Note:
        - It's expected that both gold and pred has been parsed with math_verify.parse function.
        - Function is not symmetric, gold answer should be passed as gold and prediction as pred. The non-symmetric nature appears at assignment simplification and equation interval conversion.

    Args:
        gold: The reference/correct expression(s). Can be:
            - A single SymPy expression (Basic or MatrixBase)
            - A string
            - A list of any of the above
        target: The expression(s) to verify. Same types as gold.
        float_rounding: Number of decimal places to round floats to. Defaults to 6.
        numeric_precision: Number of decimal places to consider for numeric comparisons. Defaults to 15.
            - If you know the evaluated expressions will be small, you should increase this. See: https://docs.sympy.org/latest/modules/evalf.html
        strict: Whether to enforce strict comparison mode. Defaults to True.
            - In strict mode: Variables matter and sets are not comparable with tuples
            - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        timeout_seconds: Maximum time in seconds to spend on any single comparison operation.
            Defaults to 5 seconds. Any timeout seconds > 0 or not None will result in the function to raise a ValueError if it's called in a threaded environment.

    Returns:
        bool: True if target matches gold according to any of the comparison strategies,
              False otherwise.

    Comparison Strategy:
        1. String to String comparison
        2. Numeric expressions: Comparison within specified precision
        3. Symbolic equality through simplification
        4. Special handling for:
            - Relational expressions (equations/inequalities)
            - Sets and intervals
            - Matrices and vectors
            - Complex numbers
        5. Robust error handling with timeout protection

    Example:
        >>> verify(sympy.Rational(1, 3), 0.333333)  # Numeric comparison
        True
        >>> verify(sympy.Symbol('x') + 1, sympy.Symbol('y') + 1, strict=False)  # Variable matching
        True
        >>> verify(sympy.FiniteSet(1, 2), sympy.Tuple(1, 2), strict=False)  # Set-tuple comparison
        True
    """

    global TIMEOUT_WARNING_SHOWN
    if not TIMEOUT_WARNING_SHOWN and (timeout_seconds is None or timeout_seconds <= 0):
        logger.warning(
            "Timeout is disabled as timeout_seconds is None or <= 0, you must provide \
                        the logic for timeout interuption yourself to prevent code getting stuck."
        )
        TIMEOUT_WARNING_SHOWN = True

    @timeout(timeout_seconds=timeout_seconds)
    def compare_single_extraction(
        gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str
    ) -> bool:
        # If both are sympy expressions, we can use sympy to compare them
        if isinstance(gold, (Basic, MatrixBase)) and isinstance(
            target, (Basic, MatrixBase)
        ):
            return sympy_expr_eq(
                gold, target, float_rounding, numeric_precision, strict
            )

        # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
        # of this happening are very low.  The only why one of them is not converted to sympy expression
        # is usually because the parsing logic failed in this case we should improve the parsing logic
        # instead of somehow fixing adhoc.
        elif isinstance(gold, str) and isinstance(target, str):
            # We just do string comparison for everything else
            gold = gold.strip()
            target = target.strip()

            # Ensure it's both not empty and equal
            return len(gold) > 0 and len(target) > 0 and gold == target

        return False

    def compare_single_extraction_wrapper(g, t):
        try:
            return compare_single_extraction(g, t)

        except ValueError as e:
            if str(e) == "signal only works in main thread of the main interpreter":
                raise ValueError(
                    "Math-Verify doesn't support threaded environment due to usage of signal.alarm() in timeout mechanism. If you need to run in multithreaded environment it's recommended to set the parsing_timeout=None, which will run without timeout (and signal handling). In this case you need to handle the timeouting yourself."
                ) from e
            else:
                logger.exception("Error during comparison")
                return False
        except Exception:
            #! Do not attempt to print out the g and t during handling of exception
            # Because a) it can throw an exception itself and b) it can cause it to be stuck forever during str conversion
            logger.exception("Error during comparison")
            return False
        except TimeoutException:
            logger.error("Timeout during comparison")
            return False

    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(
        compare_single_extraction_wrapper(g, t) for g, t in product(gold, target)
    )
