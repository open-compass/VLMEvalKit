grammar PS;

options {
   language=Python3;
}

// Lexer rules

WS: [ \t\r\n]+ -> skip;
// Spaces
THINSPACE: ('\\,' | '\\thinspace') -> skip;
MEDSPACE: ('\\:' | '\\medspace') -> skip;
THICKSPACE: ('\\;' | '\\thickspace') -> skip;
QUAD: '\\quad' -> skip;
QQUAD: '\\qquad' -> skip;
NEGTHINSPACE: ('\\!' | '\\negthinspace') -> skip;
NEGMEDSPACE: '\\negmedspace' -> skip;
NEGTHICKSPACE: '\\negthickspace' -> skip;
DOLLAR_SIGN: '\\$' -> skip;

IGNORE:
	(
		'\\vrule'
		| '\\vcenter'
		| '\\vbox'
		| '\\vskip'
		| '\\vspace'
		| '\\hfil'
	) -> skip;

ADD: '+' | '∔' | '⊕' | '⊞';
SUB: '-' | '−' | '∸';
MUL: '*' | '×' | '⋅' | '⋄' | '⊡' | '∗' | '⋆' | '∘' | '∙' | '⊗' | '⊠' | '⋈' | '⋉' | '⋊';
DIV: '/' | '÷' | '∕' | '\\over';

L_PAREN: '(';
R_PAREN: ')';
L_PAREN_VISUAL: '\\(';
R_PAREN_VISUAL: '\\)';
L_GROUP: '\\lgroup';
R_GROUP: '\\rgroup';
L_BRACE: '{';
R_BRACE: '}';
L_BRACE_VISUAL: '\\{';
R_BRACE_VISUAL: '\\}';
L_BRACE_CMD: '\\lbrace';
R_BRACE_CMD: '\\rbrace';
L_BRACKET: '[';
R_BRACKET: ']';
L_BRACK: '\\lbrack';
R_BRACK: '\\rbrack';

PHANTOM_CMD: '\\phantom';
BOXED_CMD: '\\boxed';

BAR: '|';
L_VERT: '\\lvert';
R_VERT: '\\rvert';
VERT: '\\vert';

NORM: '\\|';

// Dot products
L_ANGLE: '\\langle';
R_ANGLE: '\\rangle';


L_FLOOR: '\\lfloor';
R_FLOOR: '\\rfloor';
LL_CORNER: '\\llcorner';
LR_CORNER: '\\lrcorner';

L_CEIL: '\\lceil';
R_CEIL: '\\rceil';
UL_CORNER: '\\ulcorner';
UR_CORNER: '\\urcorner';

//functions
FUNC_LIM:  '\\lim';
LIM_APPROACH_SYM: '\\to' | '\\rightarrow' | '\\Rightarrow' | '\\longrightarrow' | '\\Longrightarrow';
FUNC_INT:  '\\int';
FUNC_SUM:  '\\sum';
FUNC_PROD: '\\prod';

FUNC_LOG:  '\\log';
FUNC_LN:   '\\ln';
FUNC_EXP: '\\exp';
FUNC_SIN:  '\\sin';
FUNC_COS:  '\\cos';
FUNC_TAN:  '\\tan';
FUNC_CSC:  '\\csc';
FUNC_SEC:  '\\sec';
FUNC_COT:  '\\cot';

FUNC_ARCSIN: '\\arcsin';
FUNC_ARCCOS: '\\arccos';
FUNC_ARCTAN: '\\arctan';
FUNC_ARCCSC: '\\arccsc';
FUNC_ARCSEC: '\\arcsec';
FUNC_ARCCOT: '\\arccot';

FUNC_SINH: '\\sinh';
FUNC_COSH: '\\cosh';
FUNC_TANH: '\\tanh';
FUNC_ARSINH: '\\arsinh';
FUNC_ARCOSH: '\\arcosh';
FUNC_ARTANH: '\\artanh';
FUNC_ARCSINH: '\\arcsinh';
FUNC_ARCCOSH: '\\arccosh';
FUNC_ARCTANH: '\\arctanh';

FUNC_ARSINH_NAME: 'arsinh';
FUNC_ARCSINH_NAME: 'arcsinh';
FUNC_ARCOSH_NAME: 'arcosh';
FUNC_ARCCOSH_NAME: 'arccosh';
FUNC_ARTANH_NAME: 'artanh';
FUNC_ARCTANH_NAME: 'arctanh';
FUNC_GCD_NAME: 'gcd';
FUNC_LCM_NAME: 'lcm';
FUNC_FLOOR_NAME: 'floor';
FUNC_CEIL_NAME: 'ceil';

FUNC_SQRT: '\\sqrt';
FUNC_GCD: '\\gcd';
FUNC_LCM: '\\lcm';
FUNC_FLOOR: '\\floor';
FUNC_CEIL: '\\ceil';
FUNC_MAX: '\\max';
FUNC_MIN: '\\min';

FUNC_DET: '\\det';

FUNC_EYE_NAME: 'eye';
FUNC_ZEROS_NAME: 'zeros';
FUNC_ONES_NAME: 'ones';
FUNC_COLS_NAME: 'cols';
FUNC_ROWS_NAME: 'rows';
FUNC_DIAG_NAME: 'diag';
FUNC_NORM_NAME: 'norm';
FUNC_RANK_NAME: 'rank';
FUNC_TRACE_NAME: 'trace' | 'tr';
FUNC_RREF_NAME: 'rref';
FUNC_HSTACK_NAME: 'hstack';
FUNC_VSTACK_NAME: 'vstack';
FUNC_ORTHOGONALIZE_NAME: 'orth' | 'ortho' | 'orthogonal' | 'orthogonalize';
FUNC_NULLSPACE_NAME: 'nullspace';
FUNC_DIAGONALIZE_NAME: 'eig' | 'eigen' | 'diagonalize';
FUNC_EIGENVALS_NAME: 'eigenvals' | 'eigenvalues';
FUNC_EIGENVECTORS_NAME: 'eigenvects' | 'eigenvectors';
FUNC_SVD_NAME: 'svd' | 'SVD';

//commands
CMD_TIMES: '\\times';
CMD_CDOT: '\\cdot';
CMD_DIV: '\\div';
CMD_FRAC:  '\\frac' | '\\dfrac' | '\\tfrac' | '\\cfrac';
CMD_BINOM: '\\binom' | '\\tbinom' | '\\dbinom';
CMD_CHOOSE: '\\choose';
CMD_MOD: '\\mod';

CMD_MATHIT: '\\mathit';

CMD_OPERATORNAME: '\\operatorname';

//matrix test

MATRIX_TYPE_MATRIX: 'matrix';
MATRIX_TYPE_PMATRIX: 'pmatrix';
MATRIX_TYPE_BMATRIX: 'bmatrix';
MATRIX_TYPE_DET: 'vmatrix';
MATRIX_TYPES: MATRIX_TYPE_MATRIX | MATRIX_TYPE_PMATRIX | MATRIX_TYPE_BMATRIX;
CMD_MATRIX_START: '\\begin' L_BRACE MATRIX_TYPES R_BRACE;
CMD_MATRIX_END: '\\end' L_BRACE MATRIX_TYPES R_BRACE;

CMD_ARRAY_START: '\\begin' L_BRACE 'array' R_BRACE L_BRACE ('c' | 'l' | 'r')* R_BRACE;
CMD_ARRAY_END: '\\end' L_BRACE 'array' R_BRACE;

CMD_DET_START: '\\begin' L_BRACE MATRIX_TYPE_DET R_BRACE;
CMD_DET_END: '\\end' L_BRACE MATRIX_TYPE_DET R_BRACE;
MATRIX_DEL_COL: '&';
MATRIX_DEL_ROW: '\\\\';

UNDERSCORE: '_';
CARET: '^';
COLON: ':';
SEMICOLON: ';';
COMMA: ',';
PERIOD: '.';

fragment WS_CHAR: [ \t\r\n];
DIFFERENTIAL: 'd' WS_CHAR*? ([a-zA-Z] | '\\' [a-zA-Z]+);

EXP_E: 'e' | '\\exponentialE';
E_NOTATION_E: 'E';
LETTER_NO_E: [a-df-zA-DF-Z]; // exclude e for exponential function and e notation
fragment LETTER: [a-zA-Z];
fragment DIGIT: [0-9];

MATRIX_XRIGHTARROW: '\\xrightarrow' | '\\xRightarrow';
TRANSFORM_EXCHANGE: '<->' | '<=>' | '\\leftrightarrow' | '\\Leftrightarrow';

// There is ofc huge issue with the fact that we can't distinguish between numbers and sets
// in case of 3,333 can be either 3333 or {3,333}, we default to numbers in this case
// Fixing this is a huge task and I am not going to do it
// If you want to interpret it as set change * to *?
NUMBER:
    DIGIT+
    | DIGIT* PERIOD DIGIT+;

E_NOTATION: NUMBER E_NOTATION_E (SUB | ADD)? DIGIT+;

IN: '\\in';
ASSIGNMENT: '=' | '≡' | '≣' | '≟' | '≎' | '≏' | '≐' | '≑' | '≒' | '≓' | '≔' | '≕' | '≖' | '≗';
EQUAL: '==' | '\\equiv';
APPROX: '~=' | '≈' | '≊' | '≋' | '≃' | '≅' | '\\approx';
LT: '<' | '≪' | '≺' | '⋖' | '\\lt';
LTE: '\\leq' | '\\le' | '\\leqslant' | '≤' | '≦' | '≲' | '≾' | '<=';
GT: '>' | '≫' | '≻' | '⋗' | '\\gt';
GTE: '\\geq' | '\\ge' | '\\geqslant' | '≥' | '≧' | '≳' | '≿' | '>=';
UNEQUAL: '!=' | '!==' | '\\ne' | '\\neq' | '\\not\\equiv' | '≠' | '≁' | '≄' | '≇' | '≉' | '≢';

BANG: '!';

fragment PERCENT_SIGN: '\\%' | '%';
PERCENT_NUMBER: NUMBER WS_CHAR? PERCENT_SIGN;

//Excludes some letters for use as e.g. constants in SYMBOL
fragment GREEK_LETTER:
    '\\alpha' | 'α' | '\\char"000391' | '\\char000391'
    | '\\beta' | 'β' | '\\char"000392' | '\\char000392'
    | '\\gamma' | 'γ'
    | '\\Gamma' | 'Γ'
    | '\\delta' | 'δ'
    | '\\Delta' | 'Δ'
    | '\\epsilon' | 'ε' | '\\char"000190' | '\\char000190'
    | '\\varepsilon' | 'ϵ'
    | '\\zeta' | 'ζ' | '\\char"000396' | '\\char000396'
    | '\\eta' | 'η' | '\\char"000397' | '\\char000397'
    | '\\theta' | 'θ'
    | '\\Theta' | 'Θ'
    | '\\vartheta' | 'ϑ'
    | '\\iota' | 'ι' | '\\char"000399' | '\\char000399'
    | '\\kappa' | 'κ' | '\\char"00039A' | '\\char00039A'
    | '\\lambda' | 'λ'
    | '\\Lambda' | 'Λ'
    | '\\mu' | 'μ' | '\\char"00039C' | '\\char00039C'
    | '\\nu' | 'ν' | '\\char"00039D' | '\\char00039D'
    | '\\xi' | 'ξ'
    | '\\Xi' | 'Ξ'
    | '\\omicron' | 'ο' | '\\char"00039F' | '\\char00039F'
    | '\\pi' | 'π'
    | '\\Pi' | 'Π'
    | '\\varpi' | 'ϖ'
    | '\\rho' | 'ρ' | '\\char"0003A1' | '\\char0003A1'
    | '\\varrho' | 'ϱ'
    | '\\sigma' | 'σ'
    | '\\Sigma' | 'Σ'
    | '\\varsigma' | 'ς'
    | '\\tau' | 'τ' | '\\char"0003A4' | '\\char0003A4'
    | '\\upsilon' | 'υ'
    | '\\Upsilon' | 'Υ'
    | '\\phi' | 'φ'
    | '\\Phi' | 'Φ'
    | '\\varphi' | 'ϕ'
    | '\\chi' | 'χ' | '\\char"0003A7' | '\\char0003A7'
    | '\\psi' | 'ψ'
    | '\\Psi' | 'Ψ'
    | '\\omega' | 'ω'
    | '\\Omega' | 'Ω'
    ;

GREEK_CMD: GREEK_LETTER [ ]?;

fragment OTHER_SYMBOL:
    '\\Bbbk'  |
    '\\wp'  |
    '\\nabla'  |
    '\\bigstar'  |
    '\\angle'  |
    '\\nexists'  |
    '\\diagdown'  |
    '\\measuredangle'  |
    '\\eth'  |
    'ℵ' |
    'ℶ' |
    'ℷ' |
    'ℸ' |
    '\\diagup'  |
    '\\sphericalangle'  |
    '\\clubsuit'  |
    '\\varnothing'  |
    '\\Diamond'  |
    '\\complement'  |
    '\\diamondsuit'  |
    '\\imath'  |
    '\\Finv'  |
    '\\triangledown'  |
    '\\heartsuit'  |
    '\\jmath'  |
    '\\Game'  |
    '\\triangle'  |
    '\\spadesuit'  |
    '\\ell'  |
    '\\hbar'  |
    '\\vartriangle'  |
    '\\hslash'  |
    '\\blacklozenge'  |
    '\\lozenge'  |
    '\\blacksquare'  |
    '\\mho'  |
    '\\blacktriangle'  |
    '\\sharp'  |
    '\\prime'  |
    '\\Im'  |
    '\\flat'  |
    '\\square'  |
    '\\backprime'  |
    '\\Re'  |
    '\\natural'  |
    '\\surd'  |
    '\\circledS';
OTHER_SYMBOL_CMD: OTHER_SYMBOL [ ]?;

fragment INFTY_CMD: '\\infty';
fragment PARTIAL_CMD: '\\partial';
fragment INFTY: INFTY_CMD | DOLLAR_SIGN INFTY_CMD | INFTY_CMD PERCENT_SIGN;
SYMBOL: PARTIAL_CMD | INFTY ;

fragment VARIABLE_CMD: '\\variable';
fragment VARIABLE_SYMBOL: (GREEK_CMD | OTHER_SYMBOL_CMD | LETTER | DIGIT)+ (UNDERSCORE ((L_BRACE (GREEK_CMD | OTHER_SYMBOL_CMD | LETTER | DIGIT | COMMA)+ R_BRACE) | (GREEK_CMD | OTHER_SYMBOL_CMD | LETTER | DIGIT)))?;
VARIABLE: VARIABLE_CMD L_BRACE VARIABLE_SYMBOL R_BRACE PERCENT_SIGN?;

SET_NATURALS: '\\mathbb{N}' | 'ℕ';
SET_INTEGERS: '\\mathbb{Z}' | 'ℤ';
SET_RATIONALS: '\\mathbb{Q}' | 'ℚ';
SET_REALS: '\\mathbb{R}' | 'ℝ';
SET_COMPLEX: '\\mathbb{C}' | 'ℂ';
SET_PRIMES: '\\mathbb{P}' | 'ℙ';

//collection of accents
fragment ACCENT_CMD:
    '\\acute'  |
    '\\bar'  |
    '\\overline'  |
    '\\breve'  |
    '\\check'  |
    '\\widecheck'  |
    '\\dot'  |
    '\\ddot'  |
    '\\grave'  |
    '\\hat'  |
    '\\tilde'  |
    '\\widetilde'  |
    '\\vec'  |
    '\\overrightarrow'  |
    '\\bm'  |
    '\\boldsymbol'  |
    '\\text'  |
    '\\textit'  |
    '\\textbf'  |
    '\\textnormal'  |
    '\\mathbb'  |
    '\\mathbin'  |
    '\\mathbf'  |
    '\\mathcal'  |
    '\\mathclap'  |
    '\\mathclose'  |
    '\\mathellipsis'  |
    '\\mathfrak'  |
    '\\mathinner'  |
    '\\mathit'  |
    '\\mathnormal'  |
    '\\mathop'  |
    '\\mathopen'  |
    '\\mathord'  |
    '\\mathpunct'  |
    '\\mathrel'  |
    '\\mathring'  |
    '\\mathrlap'  |
    '\\mathrm'  |
    '\\mathscr'  |
    '\\mathsf'  |
    '\\mathsterling'  |
    '\\mathtt' |
    '\\mbox';

ACCENT: ACCENT_CMD WS_CHAR? L_BRACE ('\\}' | ~[}])*? R_BRACE;

// Set operations (small subsetion)
UNION: '\\cup' | '∪';
INTERSECTION: '\\cap' | '∩';
SET_MINUS: '\\setminus' | '∖';
PLUS_MINUS: '\\pm' | '±' | '∓' | '\\mp';
// We can't add {} to the empty set as otherwise any empty braces will be lexed as empty set
SET_EMPTY: '\\emptyset' | '∅' | L_BRACE_VISUAL R_BRACE_VISUAL | L_BRACE_CMD R_BRACE_CMD;

SUPSET: '\\supseteq' | '⊇';
SUBSET: '\\subseteq' | '⊆';
NOTIN: '\\notin' | '∉';

// Grammar rules


// We also have set elements so that 1,2,3,4 is parsed as a set
math: (set_elements_relation | set_elements | set_relation) EOF;

transpose: '^T' | '^{T}' |  '^{\\\top}' | '\'';
degree: '^\\circ' | '^\\degree' | '^\\circle' | '^°' | '^{\\circ}' | '^{\\degree}' | '^{\\circle}' | '^{°}';

transform_atom: LETTER_NO_E UNDERSCORE (NUMBER | L_BRACE NUMBER R_BRACE);
transform_scale: (expr | group | ADD | SUB) transform_atom;
transform_swap: transform_atom TRANSFORM_EXCHANGE transform_atom;
transform_assignment: transform_atom transform_scale;
elementary_transform: transform_assignment | transform_scale | transform_swap;
elementary_transforms: elementary_transform (COMMA elementary_transform)*;

matrix:
    (CMD_MATRIX_START
    matrix_row (MATRIX_DEL_ROW matrix_row)* MATRIX_DEL_ROW?
    CMD_MATRIX_END | CMD_ARRAY_START
    matrix_row (MATRIX_DEL_ROW matrix_row)* MATRIX_DEL_ROW?
    CMD_ARRAY_END)
    (MATRIX_XRIGHTARROW (L_BRACKET elementary_transforms R_BRACKET)? L_BRACE elementary_transforms R_BRACE)?;

det:
    CMD_DET_START
    matrix_row (MATRIX_DEL_ROW matrix_row)* MATRIX_DEL_ROW?
    CMD_DET_END;

matrix_row:
    expr (MATRIX_DEL_COL expr)*;

relation:
    relation (IN | ASSIGNMENT | EQUAL | APPROX | LT | LTE | GT | GTE | UNEQUAL) relation
    | expr;

equality:
    expr (EQUAL | ASSIGNMENT) expr;

expr: additive;

additive:
    additive (ADD | SUB) additive
    | mp;

// mult part
mp:
    mp (MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON | CMD_MOD) mp
    | unary;

mp_nofunc:
    mp_nofunc (MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON | CMD_MOD) mp_nofunc
    | unary_nofunc;

unary:
    (ADD | SUB) unary
    | postfix+;

unary_nofunc:
    (ADD | SUB) unary_nofunc
    | postfix postfix_nofunc*;

postfix: exp postfix_op*;
postfix_nofunc: exp_nofunc postfix_op*;
postfix_op: BANG | eval_at | transpose | degree;

eval_at:
    BAR (eval_at_sup | eval_at_sub | eval_at_sup eval_at_sub);

eval_at_sub:
    UNDERSCORE L_BRACE
    (expr | equality)
    R_BRACE;

eval_at_sup:
    CARET L_BRACE
    (expr | equality)
    R_BRACE;

exp:
    exp CARET (atom | L_BRACE expr R_BRACE) subexpr?
    | comp;

exp_nofunc:
    exp_nofunc CARET (atom | L_BRACE expr R_BRACE) subexpr?
    | comp_nofunc;

comp:
    group
    | formatting_group
    | norm_group
    | abs_group
    | dot_product
    | floor_group
    | ceil_group
    | func
    | atom
    | frac
    | binom
    | matrix
    | det;

comp_nofunc:
    group
    | formatting_group
    | norm_group
    | abs_group
    | dot_product
    | floor_group
    | ceil_group
    | atom
    | frac
    | binom
    | matrix
    | det;

group:
    L_PAREN expr R_PAREN
    | L_GROUP expr R_GROUP
    | L_BRACE expr R_BRACE
    | L_BRACKET expr R_BRACKET
    | L_BRACE_VISUAL expr R_BRACE_VISUAL
    | L_BRACE_CMD expr R_BRACE_CMD
    | L_BRACK expr R_BRACK;

formatting_group:
    PHANTOM_CMD L_BRACE expr R_BRACE
    | BOXED_CMD L_BRACE expr R_BRACE;


norm_group:
    NORM expr NORM;


abs_group:
    BAR expr BAR
    | L_VERT expr R_VERT
    | VERT expr VERT;


dot_product:
    L_ANGLE expr R_ANGLE;



floor_group:
    L_FLOOR expr R_FLOOR
    | LL_CORNER expr LR_CORNER;


ceil_group:
    L_CEIL expr R_CEIL
    | UL_CORNER expr UR_CORNER;




atom_expr_no_supexpr: (LETTER_NO_E | GREEK_CMD | OTHER_SYMBOL_CMD | ACCENT) subexpr?;
atom_expr: (LETTER_NO_E | GREEK_CMD | OTHER_SYMBOL_CMD | ACCENT) (supexpr subexpr | subexpr supexpr | subexpr | supexpr)?;
atom_expr_list: (L_PAREN atom_expr (COMMA atom_expr)* R_PAREN) | atom_expr;
number_subexpr: (NUMBER) subexpr?;
atom: atom_expr | SYMBOL | number_subexpr | PERCENT_NUMBER | E_NOTATION | E_NOTATION_E | DIFFERENTIAL | VARIABLE;

frac:
    CMD_FRAC L_BRACE
    upper=expr
    R_BRACE L_BRACE
    lower=expr
    R_BRACE;

//a binomial expression
binom:
    L_BRACE upper=expr CMD_CHOOSE lower=expr R_BRACE
    | CMD_BINOM L_BRACE upper=expr R_BRACE L_BRACE lower=expr R_BRACE;

func_normal_functions_single_arg:
    FUNC_LOG | FUNC_LN | FUNC_EXP
    | FUNC_SIN | FUNC_COS | FUNC_TAN
    | FUNC_CSC | FUNC_SEC | FUNC_COT
    | FUNC_ARCSIN | FUNC_ARCCOS | FUNC_ARCTAN
    | FUNC_ARCCSC | FUNC_ARCSEC | FUNC_ARCCOT
    | FUNC_SINH | FUNC_COSH | FUNC_TANH
    | FUNC_ARSINH | FUNC_ARCOSH | FUNC_ARTANH
    | FUNC_ARCSINH | FUNC_ARCCOSH | FUNC_ARCTANH
    | FUNC_FLOOR | FUNC_CEIL | FUNC_DET;

func_normal_functions_multi_arg:
    FUNC_GCD | FUNC_LCM | FUNC_MAX | FUNC_MIN;

func_operator_names_single_arg:
    FUNC_ARSINH_NAME | FUNC_ARCOSH_NAME | FUNC_ARTANH_NAME
    | FUNC_ARCSINH_NAME | FUNC_ARCCOSH_NAME | FUNC_ARCTANH_NAME
    | FUNC_FLOOR_NAME | FUNC_CEIL_NAME | FUNC_EYE_NAME | FUNC_RANK_NAME | FUNC_TRACE_NAME
    | FUNC_RREF_NAME | FUNC_NULLSPACE_NAME | FUNC_DIAGONALIZE_NAME | FUNC_NORM_NAME
    | FUNC_EIGENVALS_NAME | FUNC_EIGENVECTORS_NAME | FUNC_SVD_NAME | FUNC_COLS_NAME | FUNC_ROWS_NAME;

func_operator_names_multi_arg:
    FUNC_GCD_NAME | FUNC_LCM_NAME | FUNC_ZEROS_NAME | FUNC_ORTHOGONALIZE_NAME
    | FUNC_ONES_NAME | FUNC_DIAG_NAME | FUNC_HSTACK_NAME | FUNC_VSTACK_NAME;

func_normal_single_arg:
    (func_normal_functions_single_arg)
    |
    (CMD_OPERATORNAME L_BRACE func_operator_name=func_operator_names_single_arg R_BRACE);

func_normal_multi_arg:
    (func_normal_functions_multi_arg)
    |
    (CMD_OPERATORNAME L_BRACE func_operator_name=func_operator_names_multi_arg R_BRACE);

func:
    func_normal_single_arg
    (subexpr? supexpr? | supexpr? subexpr?)
    (
        L_PAREN func_single_arg R_PAREN |
        func_single_arg_noparens
    )
    
    | func_normal_multi_arg
    (subexpr? supexpr? | supexpr? subexpr?)
    (
        L_PAREN func_multi_arg R_PAREN |
        func_multi_arg_noparens
    )
    | atom_expr_no_supexpr supexpr?
    (
        L_PAREN func_common_args R_PAREN |
        L_BRACKET func_common_args R_BRACKET |
        L_BRACE L_PAREN func_common_args R_PAREN R_BRACE |
        L_BRACE L_BRACKET func_common_args R_BRACKET R_BRACE
    )
    | FUNC_INT
    (subexpr supexpr | supexpr subexpr | (UNDERSCORE L_BRACE R_BRACE) (CARET L_BRACE R_BRACE) | (CARET L_BRACE R_BRACE) (UNDERSCORE L_BRACE R_BRACE) )?
    (additive? DIFFERENTIAL | frac | additive)

    | FUNC_SQRT
    (L_BRACKET root=expr R_BRACKET)?
    L_BRACE base=expr R_BRACE

    | (FUNC_SUM | FUNC_PROD)
    (subeq supexpr | supexpr subeq)
    mp
    | FUNC_LIM limit_sub mp
    | EXP_E supexpr?; //Exponential function e^x

args: (expr ',' args) | expr;

func_common_args: atom | (expr ',') | (expr ',' args);

limit_sub:
    UNDERSCORE L_BRACE
    (LETTER_NO_E | GREEK_CMD | OTHER_SYMBOL_CMD)
    LIM_APPROACH_SYM
    expr (CARET L_BRACE (ADD | SUB) R_BRACE)?
    R_BRACE;

func_single_arg: expr;
func_single_arg_noparens: mp_nofunc;

func_multi_arg: expr | (expr ',' func_multi_arg);
func_multi_arg_noparens: mp_nofunc;

subexpr: UNDERSCORE (atom | L_BRACE (expr | args) R_BRACE);
supexpr: CARET (atom | L_BRACE expr R_BRACE);

subeq: UNDERSCORE L_BRACE equality R_BRACE;
supeq: UNDERSCORE L_BRACE equality R_BRACE;

set_relation:
    set_relation (SUBSET | SUPSET) set_relation |
    atom_expr_list (IN | NOTIN | ASSIGNMENT) set_relation |
    minus_expr;

minus_expr:
    minus_expr SET_MINUS minus_expr |
    union_expr;

union_expr: 
    union_expr UNION union_expr |
    intersection_expr;

intersection_expr:
    intersection_expr INTERSECTION intersection_expr |
    set_group;

set_group:
    L_PAREN minus_expr R_PAREN
    | set_atom;

set_atom:
    interval |
    literal_set |
    ordered_tuple |
    finite_set;


interval:
    (L_BRACKET | L_PAREN | L_PAREN_VISUAL | L_BRACK | L_GROUP) 
    expr COMMA expr 
    (R_BRACKET | R_PAREN | R_PAREN_VISUAL | R_BRACK | R_GROUP);

ordered_tuple:
    (L_PAREN semicolon_elements R_PAREN) |
    (L_PAREN_VISUAL semicolon_elements R_PAREN_VISUAL) |
    (L_BRACKET semicolon_elements R_BRACKET);


// We allow boxed a set for reason that llms like to use , in boxed
finite_set:
    (L_BRACE semicolon_elements R_BRACE) |
    (L_BRACE_VISUAL semicolon_elements R_BRACE_VISUAL) |
    (BOXED_CMD L_BRACE semicolon_elements R_BRACE);
    


// We need two targets so that:
// a=1,b=2 is parsed as {a=1, b=2}
// while
// a=1,2,3,4 is parsed as {a=1,2,3,4}
set_elements_relation:
    atom_expr_list (IN | ASSIGNMENT) semicolon_elements_no_relation;

set_elements:
    semicolon_elements;

semicolon_elements:
    comma_elements (SEMICOLON comma_elements)*;

semicolon_elements_no_relation:
    comma_elements_no_relation (SEMICOLON comma_elements_no_relation)*;

comma_elements:
    element (COMMA element)*;

comma_elements_no_relation:
    element_no_relation (COMMA element_no_relation)*;

element_no_relation:
    plus_minus_expr | set_atom | expr;
    
element:
    plus_minus_expr | set_atom | relation;

plus_minus_expr:
    expr PLUS_MINUS expr | PLUS_MINUS expr;

literal_set:
    SET_NATURALS | SET_INTEGERS | SET_RATIONALS | SET_REALS | SET_COMPLEX | SET_PRIMES | SET_EMPTY | L_BRACE R_BRACE;
