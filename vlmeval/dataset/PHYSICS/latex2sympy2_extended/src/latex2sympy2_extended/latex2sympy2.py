from dataclasses import dataclass
import sympy
import re
from sympy import Basic, Matrix, MatrixBase, Number, Pow, Rational, matrix_symbols, simplify, factor, expand, apart, expand_trig
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.ErrorListener import ErrorListener
from latex2sympy2_extended.symbols import get_symbol
from latex2sympy2_extended.math_normalization import normalize_latex, NormalizationConfig
from latex2sympy2_extended.antlr_parser import PSParser, PSLexer
import sympy.functions.elementary.trigonometric as sympy_trig
import sympy.functions.elementary.hyperbolic as sympy_hyperbolic
import sympy.functions.elementary.miscellaneous as sympy_misc
import sympy.functions.elementary.integers as sympy_integers
from sympy.core.relational import Relational
from sympy.printing.str import StrPrinter
from sympy.matrices import GramSchmidt
from latex2sympy2_extended.sets import FiniteSet
from latex2sympy2_extended.logic import And
from sympy.parsing.sympy_parser import parse_expr

@dataclass(frozen=True)
class ConversionConfig:
    interpret_as_mixed_fractions: bool = True
    interpret_simple_eq_as_assignment: bool = False
    interpret_contains_as_eq: bool = True
    lowercase_symbols: bool = False
    """
    Args:
        interpret_as_mixed_fractions (bool): Whether to interpert 2 \frac{1}{2} as 2/2 or 2 + 1/2
        interpret_simple_eq_as_assignment (bool): Whether to interpret simple equations as assignments k=1 -> 1
        interpret_contains_as_eq (bool): Whether to interpret contains as equality x \\in {1,2,3} -> x = {1,2,3}
        lowercase_symbols (bool): Whether to lowercase all symbols
    """


def flatten_list(l):
    return [item for sublist in l for item in sublist]

def convert_number(number: str):
    # If it's 0,111 it's a float
    if "," in number and number.startswith("0"):
        number = number.replace(",", ".")

    integer = number.translate(str.maketrans("", "", ", ")).lstrip("0")
    if len(integer) == 0:
        integer = "0"
    return Number(integer)

def is_expr_of_only_symbols(expr):
    if hasattr(expr, 'is_Symbol') and expr.is_Symbol:
        return True

    # To allow A/S
    if hasattr(expr, 'is_Pow') and expr.is_Pow and expr.args[1] == -1 and (
        hasattr(expr.args[0], 'is_Symbol') and expr.args[0].is_Symbol
        or hasattr(expr.args[0], 'args') and all(is_expr_of_only_symbols(arg) for arg in expr.args[0].args)
    ):
        return True
    
    if hasattr(expr, 'args') and len(expr.args) > 0:
        return all(is_expr_of_only_symbols(arg) for arg in expr.args)
    return False


comma_number_regex = re.compile(r'^\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*$')

class _Latex2Sympy:
    def __init__(self, variable_values: dict | None = None, is_real=None, convert_degrees: bool = False, config: ConversionConfig = ConversionConfig()):
        # Instance variables
        self.is_real = is_real
        self.variances = {}  # For substituting
        self.var = {var:val if isinstance(val, Basic) or isinstance(val, MatrixBase) else parse_expr(val) for var, val in variable_values.items()} if variable_values else {}
        self.convert_degrees = convert_degrees
        self.config = config

    def create_parser(self, latex_str):
        """Create parser for latex string"""
        stream = InputStream(latex_str)
        lex = PSLexer(stream)
        lex.removeErrorListeners()
        lex.addErrorListener(self.MathErrorListener(latex_str))
        tokens = CommonTokenStream(lex)
        parser = PSParser(tokens)
        parser.removeErrorListeners()
        parser.addErrorListener(self.MathErrorListener(latex_str))
        return parser
    
    def parse(self, latex_str: str):
        """Main entry point to parse latex string"""
        # setup listener
        parser = self.create_parser(latex_str)

        # process the input
        math = parser.math()

        # if set relation
        if math.set_relation():
            return self.convert_set_relation(math.set_relation())
        
        if math.set_elements():
            # The issue with 333,333 or 3,333 is that it makess sets and numbers with commas ambigous
            # is that 333333 or {333,333}?
            # What we therefore do is that default to numbers with commas
            # We make the regex match directly on latex_str, because otherwise don't know if there is space
            # between the comma and the number, in this case it should be a set
            if comma_number_regex.match(latex_str):
                return convert_number(latex_str)
            return self.convert_set_elements(math.set_elements())
        
        if math.set_elements_relation():
            return self.convert_set_elements_relation(math.set_elements_relation())

        raise Exception('Nothing matched')

    class MathErrorListener(ErrorListener):
        def __init__(self, src):
            super(ErrorListener, self).__init__()
            self.src = src

        def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
            fmt = "%s\n%s\n%s"
            marker = "~" * column + "^"

            if msg.startswith("missing"):
                err = fmt % (msg, self.src, marker)
            elif msg.startswith("no viable"):
                err = fmt % ("I expected something else here", self.src, marker)
            elif msg.startswith("mismatched"):
                names = PSParser.literalNames
                expected = [names[i] for i in e.getExpectedTokens() if i < len(names)]
                if len(expected) < 10:
                    expected = " ".join(expected)
                    err = (fmt % ("I expected one of these: " + expected,
                                self.src, marker))
                else:
                    err = (fmt % ("I expected something else here", self.src, marker))
            else:
                err = fmt % ("I don't understand this", self.src, marker)
            raise Exception(err)

    def convert_relation(self, rel):
        if rel.expr():
            return self.convert_expr(rel.expr())
        
        lh = self.convert_relation(rel.relation(0))
        rh = self.convert_relation(rel.relation(1))

        if rel.LT():
            if isinstance(lh, And):
                return And(*lh._unsorted_args, sympy.StrictLessThan(lh._unsorted_args[-1].rhs, rh, evaluate=False))
            elif isinstance(lh, Relational):
                return And(lh, sympy.StrictLessThan(lh.rhs, rh, evaluate=False))
            return sympy.StrictLessThan(lh, rh, evaluate=False)
        elif rel.LTE():
            if isinstance(lh, And):
                return And(*lh._unsorted_args, sympy.LessThan(lh._unsorted_args[-1].rhs, rh, evaluate=False))
            elif isinstance(lh, Relational):
                return And(lh, sympy.LessThan(lh.rhs, rh, evaluate=False))
            return sympy.LessThan(lh, rh, evaluate=False)
        elif rel.GT():
            if isinstance(lh, And):
                return And(*lh._unsorted_args, sympy.StrictGreaterThan(lh._unsorted_args[-1].rhs, rh, evaluate=False))
            elif isinstance(lh, Relational):
                return And(lh, sympy.StrictGreaterThan(lh.rhs, rh, evaluate=False))
            return sympy.StrictGreaterThan(lh, rh, evaluate=False)
        elif rel.GTE():
            if isinstance(lh, And):
                return And(*lh._unsorted_args, sympy.GreaterThan(lh._unsorted_args[-1].rhs, rh, evaluate=False))
            elif isinstance(lh, Relational):
                return And(lh, sympy.GreaterThan(lh.rhs, rh, evaluate=False))
            return sympy.GreaterThan(lh, rh, evaluate=False)
        elif rel.EQUAL():
            if isinstance(lh, And):
                return And(*lh._unsorted_args, sympy.Eq(lh._unsorted_args[-1].rhs, rh, evaluate=False))
            elif isinstance(lh, Relational):
                return And(lh, sympy.Eq(lh.rhs, rh, evaluate=False))
            return sympy.Eq(lh, rh, evaluate=False)
        elif rel.ASSIGNMENT():
            # !Use Global variances
            if self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(lh):
                # set value
                self.variances[lh] = rh
                self.var[str(lh)] = rh
                return rh
            else:
                if isinstance(lh, And):
                    return And(*lh._unsorted_args, sympy.Eq(lh._unsorted_args[-1].rhs, rh, evaluate=False))
                elif isinstance(lh, Relational):
                    return And(lh, sympy.Eq(lh.rhs, rh, evaluate=False))
                return sympy.Eq(lh, rh, evaluate=False)
        elif rel.APPROX():
            if is_expr_of_only_symbols(lh):
                self.variances[lh] = rh
                self.var[str(lh)] = rh
                return rh
            else:
                # We don't want approximation, so we jsut take the non-approximated value
                return lh
        elif rel.IN():
            # !Use Global variances
            if hasattr(rh, 'is_Pow') and rh.is_Pow and hasattr(rh.exp, 'is_Mul'):
                n = rh.exp.args[0]
                m = rh.exp.args[1]
                if n in self.variances:
                    n = self.variances[n]
                if m in self.variances:
                    m = self.variances[m]
                rh = sympy.MatrixSymbol(lh, n, m)
                self.variances[lh] = rh
                self.var[str(lh)] = rh
            elif self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(lh):
                self.variances[lh] = rh
                self.var[str(lh)] = rh
                return rh
            else:
                raise Exception('Unrecognized relation')
            return lh
        elif rel.UNEQUAL():
            if isinstance(lh, And):
                return And(*lh._unsorted_args, sympy.Ne(lh._unsorted_args[-1].rhs, rh, evaluate=False))
            elif isinstance(lh, Relational):
                return And(lh, sympy.Ne(lh.rhs, rh, evaluate=False))
            return sympy.Ne(lh, rh, evaluate=False)


    def convert_set_relation(self, expr):
        if expr.atom_expr_list():
            left = self.convert_atom_expr_list(expr.atom_expr_list())
            right = self.convert_set_relation(expr.set_relation()[0])
            if expr.IN():
                if self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(left):
                    # set value
                    self.variances[left] = right
                    self.var[str(left)] = right
                    return right
                elif self.config.interpret_contains_as_eq:
                    return sympy.Eq(left, right, evaluate=False)
                else:
                    return sympy.Contains(left, right, evaluate=False)
            elif expr.ASSIGNMENT():
                if self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(left):
                    # set value
                    self.variances[left] = right
                    self.var[str(left)] = right
                    return right
                else:
                    return sympy.Eq(left, right, evaluate=False)
            elif expr.NOTIN():
                if self.config.interpret_contains_as_eq:
                    val = (sympy.S.Reals if self.is_real else sympy.S.Complexes) - right
                    if self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(left):
                        self.variances[left] = val
                        self.var[str(left)] = val
                        return val
                    else:
                        return sympy.Not(sympy.Eq(left, right, evaluate=False), evaluate=False)
                else:
                    return sympy.Not(right.contains(left))

        if expr.set_relation():
            left = self.convert_set_relation(expr.set_relation()[0])
            right = self.convert_set_relation(expr.set_relation()[1])
            if expr.SUBSET():
                return left.is_subset(right)
            if expr.SUPSET():
                return right.is_subset(left)
            raise Exception('Unrecognized set relation')
        return self.convert_set_minus(expr.minus_expr())

    def convert_elements_to_set_or_tuple(self, elements):
        """Helper function to convert elements to either a FiniteSet or Tuple based on content"""
        if len(elements) == 1:
            if len(elements[0]) == 1:
                return elements[0][0]
            return FiniteSet(*elements[0])
        elif all(len(elem) == 1 for elem in elements):
            return FiniteSet(*[elem[0] for elem in elements])
        else:
            return FiniteSet(*[
                sympy.Tuple(*l) for l in elements
            ])

    def convert_set_elements_relation(self, expr):
        semicolon_elements_no_relation = self.convert_semicolon_elements_no_relation(expr.semicolon_elements_no_relation())
        set_elements = self.convert_elements_to_set_or_tuple(semicolon_elements_no_relation)

        atom_expressions = self.convert_atom_expr_list(expr.atom_expr_list())
        if expr.IN():
            if self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(atom_expressions):
                # set value
                self.variances[atom_expressions] = set_elements
                self.var[str(atom_expressions)] = set_elements
                return set_elements
            elif self.config.interpret_contains_as_eq:
                return sympy.Eq(atom_expressions, set_elements, evaluate=False)
            else:
                return sympy.Contains(atom_expressions, set_elements, evaluate=False)
        elif expr.ASSIGNMENT():
            if self.config.interpret_simple_eq_as_assignment and is_expr_of_only_symbols(atom_expressions):
                # set value
                self.variances[atom_expressions] = set_elements
                self.var[str(atom_expressions)] = set_elements
                return set_elements
            else:
                return sympy.Eq(atom_expressions, set_elements, evaluate=False)
        return set_elements

    def convert_set_elements(self, expr):
        semicolon_elements = self.convert_semicolon_elements(expr.semicolon_elements())
        return self.convert_elements_to_set_or_tuple(semicolon_elements)


    def convert_set_minus(self, expr):
        if expr.union_expr():
            return self.convert_set_union(expr.union_expr())

        left = self.convert_set_minus(expr.minus_expr()[0])
        right = self.convert_set_minus(expr.minus_expr()[1])
        return sympy.Complement(left, right, evaluate=False)

    def convert_set_union(self, expr):
        if expr.intersection_expr():
            return self.convert_set_intersection(expr.intersection_expr())

        left = self.convert_set_union(expr.union_expr()[0])
        right = self.convert_set_union(expr.union_expr()[1])
        
        # It's hard to know what the user meant, but clearly we cant do intersection with tuple
        if isinstance(left, sympy.Tuple):
            left = FiniteSet(*left)

        if isinstance(right, sympy.Tuple):
            right = FiniteSet(*right)

        return sympy.Union(left, right, evaluate=False)

    def convert_set_intersection(self, expr):
        if expr.set_group():
            return self.convert_set_group(expr.set_group())

        left = self.convert_set_intersection(expr.intersection_expr()[0])
        right = self.convert_set_intersection(expr.intersection_expr()[1])

        if isinstance(left, sympy.Tuple):
            left = FiniteSet(*left)

        if isinstance(right, sympy.Tuple):
            right = FiniteSet(*right)

        return sympy.Intersection(left, right, evaluate=False)


    def convert_set_group(self, expr):
        if expr.set_atom():
            return self.convert_set_atom(expr.set_atom())

        return self.convert_set_minus(expr.minus_expr())

    def convert_set_atom(self, expr):
        if expr.literal_set():
            return self.convert_literal_set(expr.literal_set())
        if expr.interval():
            return self.convert_interval(expr.interval())
        if expr.ordered_tuple():
            return self.convert_ordered_tuple(expr.ordered_tuple())
        if expr.finite_set():
            return self.convert_finite_set(expr.finite_set())
        raise Exception('Unrecognized set atom')

    def convert_interval(self, expr):
        left_open = expr.L_PAREN() is not None or expr.L_GROUP() is not None or expr.L_PAREN_VISUAL() is not None
        right_open = expr.R_PAREN() is not None or expr.R_GROUP() is not None or expr.R_PAREN_VISUAL() is not None

        left = self.convert_expr(expr.expr()[0])
        right = self.convert_expr(expr.expr()[1])

        # It doesn't make sense to have interval which represents an empty set, in this case we treat it as a finite set
        try:
            if (left_open and right_open and right <= left) or (not left_open and not right_open and right < left):
                return sympy.Tuple(left, right)
        except Exception:
            pass

        return sympy.Interval(left, right, left_open=left_open, right_open=right_open)

    def convert_ordered_tuple(self, expr):
        elements = self.convert_semicolon_elements(expr.semicolon_elements())
        # We don't support 1 element tuples
        if len(elements) == 1 and len(elements[0]) == 1:
            return elements[0][0]
        return sympy.Tuple(*flatten_list(elements))

    def convert_finite_set(self, expr):
        content = self.convert_semicolon_elements(expr.semicolon_elements())
        # Sometimes people wrap either \boxed{a,b,c}, which we want to be a set,
        # but also \boxed{1} which we want to be a number
        if expr.BOXED_CMD():
            return self.convert_elements_to_set_or_tuple(content)
        return FiniteSet(*flatten_list(content))

    def convert_semicolon_elements(self, expr):
        result = [self.convert_comma_elements(element) for element in expr.comma_elements()]
        return result

    def convert_semicolon_elements_no_relation(self, expr):
        result = [self.convert_comma_elements_no_relation(element) for element in expr.comma_elements_no_relation()]
        return result

    def convert_comma_elements(self, expr):
        result = flatten_list(self.convert_element(element) for element in expr.element())
        return result

    def convert_comma_elements_no_relation(self, expr):
        result = flatten_list(self.convert_element(element) for element in expr.element_no_relation())
        return result

    def as_unary_minus(self, expr):
        if hasattr(expr, 'is_Rational') and expr.is_Rational:
            return sympy.Rational(-expr.p, expr.q)
        elif hasattr(expr, 'is_Integer') and expr.is_Integer:
            return -expr
        return sympy.Mul(-1, expr, evaluate=False)



    def convert_element(self, element):
        if element.plus_minus_expr():
            pm = element.plus_minus_expr()
            if len(pm.expr()) == 1:
                expr = self.convert_expr(pm.expr()[0])
                return [self.as_unary_minus(expr), expr]
            left = self.convert_expr(pm.expr()[0])
            right = self.convert_expr(pm.expr()[1])
            return [sympy.Add(left, right, evaluate=False), sympy.Add(left, self.as_unary_minus(right), evaluate=False)]
        elif element.set_atom():
            return [self.convert_set_atom(element.set_atom())]
        
        elif hasattr(element, 'relation') and element.relation():
            return [self.convert_relation(element.relation())]

        elif hasattr(element, 'expr') and element.expr():
            return [self.convert_expr(element.expr())]
        else:
            raise Exception('Unrecognized comma element')
    

        # Fallback because for some reason finites set wtih paren parses sometimes first
        # instead of interval
        return elements

    def convert_literal_set(self, expr):
        if expr.SET_NATURALS():
            return sympy.S.Naturals
        elif expr.SET_INTEGERS():
            return sympy.S.Integers
        elif expr.SET_RATIONALS():
            return sympy.S.Rationals
        elif expr.SET_REALS():
            return sympy.S.Reals
        elif expr.SET_COMPLEX():
            return sympy.S.Complexes
        elif expr.SET_EMPTY() or expr.L_BRACE() and expr.R_BRACE():
            return sympy.S.EmptySet
        raise Exception('Unrecognized literal set')


    def convert_expr(self, expr):
        if expr.additive():
            return self.convert_add(expr.additive())


    def convert_elementary_transform(self, matrix, transform):
        if transform.transform_scale():
            transform_scale = transform.transform_scale()
            transform_atom = transform_scale.transform_atom()
            k = None
            num = int(transform_atom.NUMBER().getText()) - 1
            if transform_scale.expr():
                k = self.convert_expr(transform_scale.expr())
            elif transform_scale.group():
                k = self.convert_expr(transform_scale.group().expr())
            elif transform_scale.SUB():
                k = -1
            else:
                k = 1
            if transform_atom.LETTER_NO_E().getText() == 'r':
                matrix = matrix.elementary_row_op(op='n->kn', row=num, k=k)
            elif transform_atom.LETTER_NO_E().getText() == 'c':
                matrix = matrix.elementary_col_op(op='n->kn', col=num, k=k)
            else:
                raise Exception('Row and col don\'s match')

        elif transform.transform_swap():
            first_atom = transform.transform_swap().transform_atom()[0]
            second_atom = transform.transform_swap().transform_atom()[1]
            first_num = int(first_atom.NUMBER().getText()) - 1
            second_num = int(second_atom.NUMBER().getText()) - 1
            if first_atom.LETTER_NO_E().getText() != second_atom.LETTER_NO_E().getText():
                raise Exception('Row and col don\'s match')
            elif first_atom.LETTER_NO_E().getText() == 'r':
                matrix = matrix.elementary_row_op(op='n<->m', row1=first_num, row2=second_num)
            elif first_atom.LETTER_NO_E().getText() == 'c':
                matrix = matrix.elementary_col_op(op='n<->m', col1=first_num, col2=second_num)
            else:
                raise Exception('Row and col don\'s match')

        elif transform.transform_assignment():
            first_atom = transform.transform_assignment().transform_atom()
            second_atom = transform.transform_assignment().transform_scale().transform_atom()
            transform_scale = transform.transform_assignment().transform_scale()
            k = None
            if transform_scale.expr():
                k = self.convert_expr(transform_scale.expr())
            elif transform_scale.group():
                k = self.convert_expr(transform_scale.group().expr())
            elif transform_scale.SUB():
                k = -1
            else:
                k = 1
            first_num = int(first_atom.NUMBER().getText()) - 1
            second_num = int(second_atom.NUMBER().getText()) - 1
            if first_atom.LETTER_NO_E().getText() != second_atom.LETTER_NO_E().getText():
                raise Exception('Row and col don\'s match')
            elif first_atom.LETTER_NO_E().getText() == 'r':
                matrix = matrix.elementary_row_op(op='n->n+km', k=k, row1=first_num, row2=second_num)
            elif first_atom.LETTER_NO_E().getText() == 'c':
                matrix = matrix.elementary_col_op(op='n->n+km', k=k, col1=first_num, col2=second_num)
            else:
                raise Exception('Row and col don\'s match')

        return matrix


    def convert_matrix(self, matrix):
        # build matrix
        row = matrix.matrix_row()
        tmp = []
        rows = 0
        mat = None

        for r in row:
            tmp.append([])
            for expr in r.expr():
                tmp[rows].append(self.convert_expr(expr))
            rows = rows + 1

        mat = sympy.Matrix(tmp)

        if hasattr(matrix, 'MATRIX_XRIGHTARROW') and matrix.MATRIX_XRIGHTARROW():
            transforms_list = matrix.elementary_transforms()
            if len(transforms_list) == 1:
                for transform in transforms_list[0].elementary_transform():
                    mat = self.convert_elementary_transform(mat, transform)
            elif len(transforms_list) == 2:
                # firstly transform top of xrightarrow
                for transform in transforms_list[1].elementary_transform():
                    mat = self.convert_elementary_transform(mat, transform)
                # firstly transform bottom of xrightarrow
                for transform in transforms_list[0].elementary_transform():
                    mat = self.convert_elementary_transform(mat, transform)

        return mat


    def add_flat(self, lh, rh):
        if hasattr(lh, 'is_Add') and lh.is_Add or hasattr(rh, 'is_Add') and rh.is_Add:
            args = []
            if hasattr(lh, 'is_Add') and lh.is_Add:
                args += list(lh.args)
            else:
                args += [lh]
            if hasattr(rh, 'is_Add') and rh.is_Add:
                args = args + list(rh.args)
            else:
                args += [rh]
            return sympy.Add(*args, evaluate=False)
        else:
            return sympy.Add(lh, rh, evaluate=False)


    def mat_add_flat(self, lh, rh):
        if hasattr(lh, 'is_MatAdd') and lh.is_MatAdd or hasattr(rh, 'is_MatAdd') and rh.is_MatAdd:
            args = []
            if hasattr(lh, 'is_MatAdd') and lh.is_MatAdd:
                args += list(lh.args)
            else:
                args += [lh]
            if hasattr(rh, 'is_MatAdd') and rh.is_MatAdd:
                args = args + list(rh.args)
            else:
                args += [rh]
            # Previously there doit, but I don't think it's needed
            return sympy.MatAdd(*[arg for arg in args], evaluate=False)
        else:
            return sympy.MatAdd(lh, rh, evaluate=False)


    def mul_flat(self, lh, rh):
        if hasattr(lh, 'is_Mul') and lh.is_Mul or hasattr(rh, 'is_Mul') and rh.is_Mul:
            args = []
            if hasattr(lh, 'is_Mul') and lh.is_Mul:
                args += list(lh.args)
            else:
                args += [lh]
            if hasattr(rh, 'is_Mul') and rh.is_Mul:
                args = args + list(rh.args)
            else:
                args += [rh]
            return sympy.Mul(*args, evaluate=False)
        else:
            return sympy.Mul(lh, rh, evaluate=False)


    def mat_mul_flat(self, lh, rh):
        if hasattr(lh, 'is_MatMul') and lh.is_MatMul or hasattr(rh, 'is_MatMul') and rh.is_MatMul:
            args = []
            if hasattr(lh, 'is_MatMul') and lh.is_MatMul:
                args += list(lh.args)
            else:
                args += [lh]
            if hasattr(rh, 'is_MatMul') and rh.is_MatMul:
                args = args + list(rh.args)
            else:
                args += [rh]
            return sympy.MatMul(*[arg for arg in args], evaluate=False)
        else:
            # We don't have to doit there
            # if hasattr(lh, 'is_Matrix'):
            #     lh = lh.doit()
            # if hasattr(rh, 'is_Matrix'):
            #     rh = rh.doit()
            return sympy.MatMul(lh, rh, evaluate=False)


    def convert_add(self, add):
        if add.ADD():
            lh = self.convert_add(add.additive(0))
            rh = self.convert_add(add.additive(1))

            if (hasattr(lh, 'is_Matrix') and lh.is_Matrix) or (hasattr(rh, 'is_Matrix') and rh.is_Matrix):
                return self.mat_add_flat(lh, rh)
            else:
                return self.add_flat(lh, rh)
        elif add.SUB():
            lh = self.convert_add(add.additive(0))
            rh = self.convert_add(add.additive(1))

            if (hasattr(lh, 'is_Matrix') and lh.is_Matrix) or (hasattr(rh, 'is_Matrix') and rh.is_Matrix):
                return self.mat_add_flat(lh, self.mat_mul_flat(-1, rh))
            else:
                # If we want to force ordering for variables this should be:
                # return Sub(lh, rh, evaluate=False)
                if not (hasattr(rh, 'is_Matrix') and rh.is_Matrix) and (hasattr(rh, 'func') and rh.func.is_Number):
                    rh = -rh
                else:
                    rh = self.mul_flat(-1, rh)
                return self.add_flat(lh, rh)
        else:
            return self.convert_mp(add.mp())


    def convert_mp(self, mp):
        if hasattr(mp, 'mp'):
            mp_left = mp.mp(0)
            mp_right = mp.mp(1)
        else:
            mp_left = mp.mp_nofunc(0)
            mp_right = mp.mp_nofunc(1)

        if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
            lh = self.convert_mp(mp_left)
            rh = self.convert_mp(mp_right)

            if (hasattr(lh, 'is_Matrix') and lh.is_Matrix) or (hasattr(rh, 'is_Matrix') and rh.is_Matrix):
                return self.mat_mul_flat(lh, rh)
            else:
                return self.mul_flat(lh, rh)
        elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
            lh = self.convert_mp(mp_left)
            rh = self.convert_mp(mp_right)
            if (hasattr(lh, 'is_Matrix') and lh.is_Matrix) or (hasattr(rh, 'is_Matrix') and rh.is_Matrix):
                return sympy.MatMul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
            
            # If both are numbers, we convert to sympy.Rational
            elif hasattr(lh, 'is_Integer') and lh.is_Integer and hasattr(rh, 'is_Integer') and rh.is_Integer:
                return sympy.Rational(lh, rh)
            else:
                return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
        elif mp.CMD_MOD():
            lh = self.convert_mp(mp_left)
            rh = self.convert_mp(mp_right)
            if (hasattr(rh, 'is_Matrix') and rh.is_Matrix):
                raise Exception("Cannot perform modulo operation with a matrix as an operand")
            else:
                return sympy.Mod(lh, rh, evaluate=False)
        else:
            if hasattr(mp, 'unary'):
                return self.convert_unary(mp.unary())
            else:
                return self.convert_unary(mp.unary_nofunc())


    def convert_unary(self, unary):
        if hasattr(unary, 'unary'):
            nested_unary = unary.unary()
        else:
            nested_unary = unary.unary_nofunc()
        if hasattr(unary, 'postfix_nofunc'):
            first = unary.postfix()
            tail = unary.postfix_nofunc()
            postfix = [first] + tail
        else:
            postfix = unary.postfix()

        if unary.ADD():
            return self.convert_unary(nested_unary)
        elif unary.SUB():
            tmp_convert_nested_unary = self.convert_unary(nested_unary)
            if (hasattr(tmp_convert_nested_unary, 'is_Matrix') and tmp_convert_nested_unary.is_Matrix):
                return self.mat_mul_flat(-1, tmp_convert_nested_unary)
            else:
                if (hasattr(tmp_convert_nested_unary, 'func') and tmp_convert_nested_unary.func.is_Number):
                    return -tmp_convert_nested_unary
                
                elif hasattr(tmp_convert_nested_unary, 'is_Number') and tmp_convert_nested_unary.is_Number:
                    return -tmp_convert_nested_unary
                else:
                    return self.mul_flat(-1, tmp_convert_nested_unary)
        elif postfix:
            return self.convert_postfix_list(postfix)


    def convert_postfix_list(self, arr, i=0):
        if i >= len(arr):
            raise Exception("Index out of bounds")

        res = self.convert_postfix(arr[i])

        if isinstance(res, sympy.Expr) or isinstance(res, sympy.Matrix):
            if i == len(arr) - 1:
                return res  # nothing to multiply by
            else:
                # multiply by next
                rh = self.convert_postfix_list(arr, i + 1)

                if (hasattr(res, 'is_Matrix') and res.is_Matrix) or (hasattr(rh, 'is_Matrix') and rh.is_Matrix):
                    return self.mat_mul_flat(res, rh)
                # Support for mixed fractions, 2 \frac{1}{2}
                elif hasattr(res, 'is_Integer') and res.is_Integer and hasattr(rh, 'is_Rational') and rh.is_Rational and rh.p > 0 and rh.q > 0:
                    if res < 0:
                        return sympy.Rational(res*rh.q - rh.p, rh.q)
                    else:
                        return sympy.Rational(res*rh.q + rh.p, rh.q)
                else:
                    return self.mul_flat(res, rh)
        elif isinstance(res, list) and len(res) == 1:  # must be derivative
            wrt = res[0]
            if i == len(arr) - 1:
                raise Exception("Expected expression for derivative")
            else:
                expr = self.convert_postfix_list(arr, i + 1)
                return sympy.Derivative(expr, wrt)
        
        return res


    def do_subs(self, expr, at):
        if at.expr():
            at_expr = self.convert_expr(at.expr())
            syms = at_expr.atoms(sympy.Symbol)
            if len(syms) == 0:
                return expr
            elif len(syms) > 0:
                sym = next(iter(syms))
                return expr.subs(sym, at_expr)
        elif at.equality():
            lh = self.convert_expr(at.equality().expr(0))
            rh = self.convert_expr(at.equality().expr(1))
            return expr.subs(lh, rh)


    def convert_postfix(self, postfix):
        if hasattr(postfix, 'exp'):
            exp_nested = postfix.exp()
        else:
            exp_nested = postfix.exp_nofunc()

        exp = self.convert_exp(exp_nested)
        for op in postfix.postfix_op():
            if op.BANG():
                if isinstance(exp, list):
                    raise Exception("Cannot apply postfix to derivative")
                exp = sympy.factorial(exp, evaluate=False)
            elif op.eval_at():
                ev = op.eval_at()
                at_b = None
                at_a = None
                if ev.eval_at_sup():
                    at_b = self.do_subs(exp, ev.eval_at_sup())
                if ev.eval_at_sub():
                    at_a = self.do_subs(exp, ev.eval_at_sub())
                if at_b is not None and at_a is not None:
                    exp = self.add_flat(at_b, self.mul_flat(at_a, -1))
                elif at_b is not None:
                    exp = at_b
                elif at_a is not None:
                    exp = at_a
            elif op.transpose():
                try:
                    exp = exp.T
                except Exception:
                    try:
                        exp = sympy.transpose(exp)
                    except Exception:
                        pass
                    pass
            elif op.degree() and self.convert_degrees:
                try:
                    exp = sympy.Mul(exp, sympy.pi/180)
                except Exception:
                    pass

        return exp


    def convert_exp(self, exp):
        if hasattr(exp, 'exp'):
            exp_nested = exp.exp()
        else:
            exp_nested = exp.exp_nofunc()

        if exp_nested:
            base = self.convert_exp(exp_nested)
            if isinstance(base, list):
                raise Exception("Cannot raise derivative to power")
            if exp.atom():
                exponent = self.convert_atom(exp.atom())
            else:
                # It's expr
                exponent = self.convert_expr(exp.expr())

            return sympy.Pow(base, exponent, evaluate=False)
        else:
            if hasattr(exp, 'comp'):
                return self.convert_comp(exp.comp())
            else:
                return self.convert_comp(exp.comp_nofunc())


    def convert_comp(self, comp):
        if comp.group():
            return self.convert_expr(comp.group().expr())
        elif comp.formatting_group():
            return self.convert_expr(comp.formatting_group().expr())
        elif comp.norm_group():
            return self.convert_expr(comp.norm_group().expr()).norm()
        elif comp.abs_group():
            return sympy.Abs(self.convert_expr(comp.abs_group().expr()), evaluate=False)
        elif comp.floor_group():
            return self.handle_floor(self.convert_expr(comp.floor_group().expr()))
        elif comp.ceil_group():
            return self.handle_ceil(self.convert_expr(comp.ceil_group().expr()))
        elif comp.atom():
            return self.convert_atom(comp.atom())
        elif comp.frac():
            return self.convert_frac(comp.frac())
        elif comp.binom():
            return self.convert_binom(comp.binom())
        elif comp.matrix():
            return self.convert_matrix(comp.matrix())
        elif comp.det():
            # !Use Global variances
            return self.convert_matrix(comp.det()).subs(self.variances).det()
        elif comp.func():
            return self.convert_func(comp.func())


    def convert_atom_expr(self, atom_expr):
        # find the atom's text
        atom_text = ''
        if atom_expr.LETTER_NO_E():
            atom_text = atom_expr.LETTER_NO_E().getText()
            if atom_text == "I":
                return sympy.I
        elif atom_expr.GREEK_CMD():
            atom_text = atom_expr.GREEK_CMD().getText()
        elif atom_expr.OTHER_SYMBOL_CMD():
            atom_text = atom_expr.OTHER_SYMBOL_CMD().getText()
        elif atom_expr.ACCENT():
            atom_text = atom_expr.ACCENT().getText()
            # Remove the command by striping first { and last }
            text_start = atom_text.index('{')
            accent_name = atom_text[1:text_start]
            accent_text = atom_text[text_start + 1:-1].replace(" ", "")
            # exception: check if bar or overline which are treated both as bar
            if accent_name in ["bar", "overline"]:
                accent_name = "bar"
            elif accent_name in ["vec", "overrightarrow"]:
                accent_name = "vec"
            elif accent_name in ["tilde", "widetilde"]:
                accent_name = "tilde"
            elif "text" in accent_name or "mbox" in accent_name:
                # We ignore text accents so that $C$ == $\\text{C}$
                accent_name = ""
                # Remove the parentheses
                accent_text = accent_text.replace("(", "").replace(")", "")
            elif "math" in accent_name:
                accent_name = "math"
            
            if accent_name:
                atom_text = f"{accent_name}{{{accent_text}}}"
            else:
                atom_text = accent_text

        # find atom's subscript, if any
        subscript_text = ''
        if atom_expr.subexpr():
            subexpr = atom_expr.subexpr()
            subscript = None
            if subexpr.expr():  # subscript is expr
                subscript = subexpr.expr().getText().strip()
            elif subexpr.atom():  # subscript is atom
                subscript = subexpr.atom().getText().strip()
            elif subexpr.args():  # subscript is args
                subscript = subexpr.args().getText().strip()
            subscript_inner_text = StrPrinter().doprint(subscript)
            if len(subscript_inner_text) > 1:
                subscript_text = '_{' + subscript_inner_text + '}'
            else:
                subscript_text = '_' + subscript_inner_text

        # construct the symbol using the text and optional subscript
        atom_symbol = get_symbol(atom_text.strip() + subscript_text, self.is_real, self.config.lowercase_symbols)
        # for matrix symbol
        matrix_symbol = None
        if atom_text + subscript_text in self.var:
            try:
                rh = self.var[atom_text + subscript_text]
                shape = sympy.shape(rh)
                matrix_symbol = sympy.MatrixSymbol(atom_text + subscript_text, shape[0], shape[1])
                self.variances[matrix_symbol] = self.variances[atom_symbol]
            except Exception:
                pass

        # find the atom's superscript, and return as a Pow if found
        if atom_expr.supexpr():
            supexpr = atom_expr.supexpr()
            func_pow = None
            if supexpr.expr():
                func_pow = self.convert_expr(supexpr.expr())
            else:
                func_pow = self.convert_atom(supexpr.atom())
            return sympy.Pow(atom_symbol, func_pow, evaluate=False)

        return atom_symbol if not matrix_symbol else matrix_symbol

    def convert_atom_expr_list(self, atom_expr_list):
        converted_atoms = [self.convert_atom_expr(atom_expr) for atom_expr in atom_expr_list.atom_expr()]
        if len(converted_atoms) == 1:
            return converted_atoms[0]
        return sympy.Tuple(*converted_atoms)

    def create_symbol(self, text, enforce_case=False):
        if self.config.lowercase_symbols and not enforce_case:
            return sympy.Symbol(text.lower(), real=self.is_real)
        else:
            return sympy.Symbol(text, real=self.is_real)

    def convert_atom(self, atom):
        if atom.atom_expr():
            return self.convert_atom_expr(atom.atom_expr())
        elif atom.SYMBOL():
            s = atom.SYMBOL().getText().replace("\\$", "").replace("\\%", "")
            if s == "\\infty":
                return sympy.oo
            else:
                raise Exception("Unrecognized symbol")
        elif atom.number_subexpr():
            # We just ignore the subexpr right now
            s = atom.number_subexpr().NUMBER().getText()
            number = self.parse_number(s)
            return number
        elif atom.E_NOTATION():
            s = atom.E_NOTATION().getText()
            return self.parse_number(s)
        elif atom.E_NOTATION_E():
            return self.create_symbol('E')
        elif atom.DIFFERENTIAL():
            diff_var = self.get_differential_var(atom.DIFFERENTIAL())
            return sympy.Symbol('d' + diff_var.name, real=self.is_real)
        elif atom.VARIABLE():
            text = atom.VARIABLE().getText()
            is_percent = text.endswith("\\%")
            trim_amount = 3 if is_percent else 1
            atom_text = text[10:]
            atom_text = atom_text[0:len(atom_text) - trim_amount]

            # Hynek: I don't think we want this to happen
            # replace the variable for already known variable values
            # if atom_text in self.var:
            #     symbol = self.var[atom_text]
            # else:
            symbol = self.create_symbol(atom_text)

            if is_percent:
                return convert_to_pct(symbol)

            # return the symbol
            return symbol

        elif atom.PERCENT_NUMBER():
            text = atom.PERCENT_NUMBER().getText().replace("\\%", "").replace("%", "").replace(",", "")
            number = self.parse_number(text)
            percent = sympy.Mul(number, Rational(1, 100), evaluate=False)
            return percent
    def parse_number(self, text):
        text = text.replace(",", "")
        # If it's made only of digits, remove the starting 0
        if text.isdigit():
            while len(text) > 1 and text[0] == '0':
                text = text[1:]
        return sympy.Number(text)


    def rule2text(self, ctx):
        stream = ctx.start.getInputStream()
        # starting index of starting token
        startIdx = ctx.start.start
        # stopping index of stopping token
        stopIdx = ctx.stop.stop

        return stream.getText(startIdx, stopIdx)


    def convert_frac(self, frac):
        diff_op = False
        partial_op = False
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1
        wrt = None
        if (frac.lower.start == frac.lower.stop and
                frac.lower.start.type == PSLexer.DIFFERENTIAL):
            wrt = self.get_differential_var_str(frac.lower.start.text)
            diff_op = True
        elif (lower_itv_len == 2 and
            frac.lower.start.type == PSLexer.SYMBOL and
            frac.lower.start.text == '\\partial' and
            (frac.lower.stop.type == PSLexer.LETTER_NO_E or frac.lower.stop.type == PSLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == PSLexer.SYMBOL:
                wrt = wrt[1:]

        if diff_op or partial_op:
            wrt = self.create_symbol(wrt, enforce_case=True)
            if (diff_op and frac.upper.start == frac.upper.stop and
                frac.upper.start.type == PSLexer.LETTER_NO_E and
                    frac.upper.start.text == 'd'):
                return [wrt]
            elif (partial_op and frac.upper.start == frac.upper.stop and
                frac.upper.start.type == PSLexer.SYMBOL and
                frac.upper.start.text == '\\partial'):
                return [wrt]
            upper_text = self.rule2text(frac.upper)

            expr_top = None
            if diff_op and upper_text.startswith('d'):
                expr_top = self.parse(upper_text[1:])
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = self.parse(upper_text[len('\\partial'):])
            if expr_top:
                return sympy.Derivative(expr_top, wrt)

        expr_top = self.convert_expr(frac.upper)
        expr_bot = self.convert_expr(frac.lower)
        if hasattr(expr_top, 'is_Matrix') and expr_top.is_Matrix or hasattr(expr_bot, 'is_Matrix') and expr_bot.is_Matrix:
            return sympy.MatMul(expr_top, sympy.Pow(expr_bot, -1, evaluate=False), evaluate=False)
        
        elif hasattr(expr_top, 'is_Integer') and expr_top.is_Integer and hasattr(expr_bot, 'is_Integer') and expr_bot.is_Integer:
            return sympy.Rational(expr_top, expr_bot)
        else:
            return sympy.Mul(expr_top, sympy.Pow(expr_bot, -1, evaluate=False), evaluate=False)


    def convert_binom(self, binom):
        expr_top = self.convert_expr(binom.upper)
        expr_bot = self.convert_expr(binom.lower)
        return sympy.binomial(expr_top, expr_bot)


    def convert_func(self, func):
        if func.func_normal_single_arg():
            if func.func_single_arg():  # function called with parenthesis
                arg = self.convert_func_arg(func.func_single_arg())
            else:
                arg = self.convert_func_arg(func.func_single_arg_noparens())

            name = func.func_normal_single_arg().start.text[1:]


            # get pow
            func_pow = None
            if func.supexpr():
                if func.supexpr().expr():
                    func_pow = self.convert_expr(func.supexpr().expr())
                else:
                    func_pow = self.convert_atom(func.supexpr().atom())

            # change arc<trig> -> a<trig>
            if name in ["arcsin", "arccos", "arctan", "arccsc", "arcsec",
                        "arccot"]:
                name = "a" + name[3:]
                expr = getattr(sympy_trig, name)(arg, evaluate=False)
            elif name in ["arsinh", "arcosh", "artanh"]:
                name = "a" + name[2:]
                expr = getattr(sympy_hyperbolic, name)(arg, evaluate=False)
            elif name in ["arcsinh", "arccosh", "arctanh"]:
                name = "a" + name[3:]
                expr = getattr(sympy_hyperbolic, name)(arg, evaluate=False)
            elif name == "operatorname":
                operatorname = func.func_normal_single_arg().func_operator_name.getText()

                if operatorname in ["arsinh", "arcosh", "artanh"]:
                    operatorname = "a" + operatorname[2:]
                    expr = getattr(sympy_hyperbolic, operatorname)(arg, evaluate=False)
                elif operatorname in ["arcsinh", "arccosh", "arctanh"]:
                    operatorname = "a" + operatorname[3:]
                    expr = getattr(sympy_hyperbolic, operatorname)(arg, evaluate=False)
                elif operatorname == "floor":
                    expr = self.handle_floor(arg)
                elif operatorname == "ceil":
                    expr = self.handle_ceil(arg)
                elif operatorname == 'eye':
                    expr = sympy.eye(arg)
                elif operatorname == 'rank':
                    expr = sympy.Integer(arg.rank())
                elif operatorname in ['trace', 'tr']:
                    expr = arg.trace()
                elif operatorname == 'rref':
                    expr = arg.rref()[0]
                elif operatorname == 'nullspace':
                    expr = arg.nullspace()
                elif operatorname == 'norm':
                    expr = arg.norm()
                elif operatorname == 'cols':
                    expr = [arg.col(i) for i in range(arg.cols)]
                elif operatorname == 'rows':
                    expr = [arg.row(i) for i in range(arg.rows)]
                elif operatorname in ['eig', 'eigen', 'diagonalize']:
                    expr = arg.diagonalize()
                elif operatorname in ['eigenvals', 'eigenvalues']:
                    expr = arg.eigenvals()
                elif operatorname in ['eigenvects', 'eigenvectors']:
                    expr = arg.eigenvects()
                elif operatorname in ['svd', 'SVD']:
                    expr = arg.singular_value_decomposition()
                else:
                    expr = sympy.Function(operatorname)(arg, evaluate=False)
            elif name in ["log", "ln"]:
                base = 10
                if func.subexpr():
                    if func.subexpr().atom():
                        base = self.convert_atom(func.subexpr().atom())
                    else:
                        base = self.convert_expr(func.subexpr().expr())
                elif name == "log":
                    base = 10
                else:
                    # it's ln
                    base = sympy.E
                expr = sympy.log(arg, base, evaluate=False)
            elif name in ["exp", "exponentialE"]:
                expr = sympy.exp(arg, evaluate=False)
            elif name == "floor":
                expr = self.handle_floor(arg)
            elif name == "ceil":
                expr = self.handle_ceil(arg)
            elif name == 'det':
                expr = arg.det()
            
            elif name in ["sin", "cos", "tan", "csc", "sec", "cot"]:
                if func_pow == -1:
                    name = "a" + name
                    func_pow = None
                expr = getattr(sympy_trig, name)(arg, evaluate=False)
            
            elif name in ["sinh", "cosh", "tanh"]:
                if func_pow == -1:
                    name = "a" + name
                    func_pow = None
                expr = getattr(sympy_hyperbolic, name)(arg, evaluate=False)
            
            else:
                expr = sympy.Function(name)(arg, evaluate=False)

            if func_pow:
                expr = sympy.Pow(expr, func_pow, evaluate=False)

            return expr

        elif func.func_normal_multi_arg():
            if func.func_multi_arg():  # function called with parenthesis
                args = func.func_multi_arg().getText().split(",")
            else:
                args = func.func_multi_arg_noparens().split(",")

            args = list(map(lambda arg: self.parse(arg), args))
            name = func.func_normal_multi_arg().start.text[1:]

            if name == "operatorname":
                operatorname = func.func_normal_multi_arg().func_operator_name.getText()
                if operatorname in ["gcd", "lcm"]:
                    expr = self.handle_gcd_lcm(operatorname, args)
                elif operatorname == 'zeros':
                    expr = sympy.zeros(*args)
                elif operatorname == 'ones':
                    expr = sympy.ones(*args)
                elif operatorname == 'diag':
                    expr = sympy.diag(*args)
                elif operatorname == 'hstack':
                    expr = sympy.Matrix.hstack(*args)
                elif operatorname == 'vstack':
                    expr = sympy.Matrix.vstack(*args)
                elif operatorname in ['orth', 'ortho', 'orthogonal', 'orthogonalize']:
                    if len(args) == 1:
                        arg = args[0]
                        expr = GramSchmidt([arg.col(i) for i in range(arg.cols)], True)
                    else:
                        expr = GramSchmidt(args, True)
                else:
                    expr = sympy.Function(operatorname)(*args, evaluate=False)
            elif name in ["gcd", "lcm"]:
                expr = self.handle_gcd_lcm(name, args)
            elif name in ["max", "min"]:
                name = name[0].upper() + name[1:]
                expr = getattr(sympy_misc, name)(*args, evaluate=False)
            else:
                expr = sympy.Function(name)(*args, evaluate=False)

            func_pow = None
            should_pow = True
            if func.supexpr():
                if func.supexpr().expr():
                    func_pow = self.convert_expr(func.supexpr().expr())
                else:
                    func_pow = self.convert_atom(func.supexpr().atom())

            if func_pow and should_pow:
                expr = sympy.Pow(expr, func_pow, evaluate=False)

            return expr

        elif func.atom_expr_no_supexpr():
            # define a function
            f = sympy.Function(func.atom_expr_no_supexpr().getText())
            # args
            args = func.func_common_args().getText().split(",")
            if args[-1] == '':
                args = args[:-1]
            args = [self.parse(arg) for arg in args]
            # supexpr
            if func.supexpr():
                if func.supexpr().expr():
                    expr = self.convert_expr(func.supexpr().expr())
                else:
                    expr = self.convert_atom(func.supexpr().atom())
                return sympy.Pow(f(*args), expr, evaluate=False)
            else:
                return f(*args)
        elif func.FUNC_INT():
            return self.handle_integral(func)
        elif func.FUNC_SQRT():
            expr = self.convert_expr(func.base)
            if func.root:
                r = self.convert_expr(func.root)
                return sympy.Pow(expr, 1 / r, evaluate=False)
            else:
                return sympy.Pow(expr, sympy.S.Half, evaluate=False)
        elif func.FUNC_SUM():
            return self.handle_sum_or_prod(func, "summation")
        elif func.FUNC_PROD():
            return self.handle_sum_or_prod(func, "product")
        elif func.FUNC_LIM():
            return self.handle_limit(func)
        elif func.EXP_E():
            return self.handle_exp(func)


    def convert_func_arg(self, arg):
        if hasattr(arg, 'expr'):
            return self.convert_expr(arg.expr())
        else:
            return self.convert_mp(arg.mp_nofunc())


    def handle_integral(self, func):
        if func.additive():
            integrand = self.convert_add(func.additive())
        elif func.frac():
            integrand = self.convert_frac(func.frac())
        else:
            integrand = 1

        int_var = None
        if func.DIFFERENTIAL():
            int_var = self.get_differential_var(func.DIFFERENTIAL())
        else:
            for sym in integrand.atoms(sympy.Symbol):
                s = str(sym)
                if len(s) > 1 and s[0] == 'd':
                    if s[1] == '\\':
                        int_var = self.create_symbol(s[2:], enforce_case=True)
                    else:
                        int_var = self.create_symbol(s[1:], enforce_case=True)
                    int_sym = sym
            if int_var:
                integrand = integrand.subs(int_sym, 1)
            else:
                # Assume dx by default
                int_var = self.create_symbol('x', enforce_case=True)

        if func.subexpr():
            if func.subexpr().atom():
                lower = self.convert_atom(func.subexpr().atom())
            else:
                lower = self.convert_expr(func.subexpr().expr())
            if func.supexpr().atom():
                upper = self.convert_atom(func.supexpr().atom())
            else:
                upper = self.convert_expr(func.supexpr().expr())
            return sympy.Integral(integrand, (int_var, lower, upper))
        else:
            return sympy.Integral(integrand, int_var)


    def handle_sum_or_prod(self, func, name):
        val = self.convert_mp(func.mp())
        iter_var = self.convert_expr(func.subeq().equality().expr(0))
        start = self.convert_expr(func.subeq().equality().expr(1))
        if func.supexpr().expr():  # ^{expr}
            end = self.convert_expr(func.supexpr().expr())
        else:  # ^atom
            end = self.convert_atom(func.supexpr().atom())

        if name == "summation":
            return sympy.Sum(val, (iter_var, start, end))
        elif name == "product":
            return sympy.Product(val, (iter_var, start, end))


    def handle_limit(self, func):
        sub = func.limit_sub()
        if sub.LETTER_NO_E():
            sub_var = self.create_symbol(sub.LETTER_NO_E().getText(), enforce_case=True)
        elif sub.GREEK_CMD():
            sub_var = get_symbol(sub.GREEK_CMD().getText().strip(), self.is_real)
        elif sub.OTHER_SYMBOL_CMD():
            sub_var = get_symbol(sub.OTHER_SYMBOL_CMD().getText().strip(), self.is_real)
        else:
            sub_var = self.create_symbol('x', enforce_case=True)
        if sub.SUB():
            direction = "-"
        else:
            direction = "+"
        approaching = self.convert_expr(sub.expr())
        content = self.convert_mp(func.mp())

        return sympy.Limit(content, sub_var, approaching, direction)


    def handle_exp(self, func):
        if func.supexpr():
            if func.supexpr().expr():  # ^{expr}
                exp_arg = self.convert_expr(func.supexpr().expr())
            else:  # ^atom
                exp_arg = self.convert_atom(func.supexpr().atom())
        else:
            exp_arg = 1
        return sympy.exp(exp_arg)


    def handle_gcd_lcm(self, f, args):
        """
        Return the result of gcd() or lcm(), as UnevaluatedExpr

        f: str - name of function ("gcd" or "lcm")
        args: List[Expr] - list of function arguments
        """

        args = tuple(map(sympy.nsimplify, args))

        # gcd() and lcm() don't support evaluate=False
        return sympy.UnevaluatedExpr(getattr(sympy, f)(args))


    def handle_floor(self, expr):
        """
        Apply floor() then return the floored expression.

        expr: Expr - sympy expression as an argument to floor()
        """
        return sympy_integers.floor(expr, evaluate=False)


    def handle_ceil(self, expr):
        """
        Apply ceil() then return the ceil-ed expression.

        expr: Expr - sympy expression as an argument to ceil()
        """
        return sympy_integers.ceiling(expr, evaluate=False)



    def get_differential_var(self, d):
        text = self.get_differential_var_str(d.getText())
        return self.create_symbol(text, enforce_case=True)


    def get_differential_var_str(self, text):
        for i in range(1, len(text)):
            c = text[i]
            if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
                idx = i
                break
        text = text[idx:]
        if text[0] == "\\":
            text = text[1:]
        return text

# # Set image value
# latex2latex('i=I')
# latex2latex('j=I')
# # set Identity(i)
# for i in range(1, 10):
#     lh = sympy.Symbol(r'\bm{I}_' + str(i), real=False)
#     lh_m = sympy.MatrixSymbol(r'\bm{I}_' + str(i), i, i)
#     rh = sympy.Identity(i).as_mutable()
#     variances[lh] = rh
#     variances[lh_m] = rh
#     var[str(lh)] = rh

# Common regex

def convert_to_pct(number: Number):
    return sympy.Mul(number, sympy.Rational(1, 100), evaluate=False)

def latex2sympy(latex_str: str, variable_values: dict | None = None, is_real=None, convert_degrees: bool = False, normalization_config: NormalizationConfig | None = NormalizationConfig(), conversion_config: ConversionConfig = ConversionConfig()):
    converter = _Latex2Sympy(variable_values, is_real, convert_degrees, config=conversion_config)
    if normalization_config is not None:
        latex_str = normalize_latex(latex_str, normalization_config)
    return converter.parse(latex_str)


if __name__ == "__main__":
    # print(normalize_latex("20 \\%", NormalizationConfig(basic_latex=True, units=True, malformed_operators=False, nits=True, boxed=False, equations=True)))
    print(latex2sympy(r"\boxed{\text{C,  E}}"))
    print(latex2sympy(r"0.111"))
