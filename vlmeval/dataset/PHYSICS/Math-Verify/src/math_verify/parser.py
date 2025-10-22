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

import logging
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import groupby
from typing import Literal, Sequence

import sympy
from latex2sympy2_extended.latex2sympy2 import (
    NormalizationConfig,
    latex2sympy,
    normalize_latex,
)
from latex2sympy2_extended.sets import FiniteSet
from sympy import Basic, MatrixBase, Number
from sympy.parsing import parse_expr

from math_verify.errors import TimeoutException
from math_verify.grader import should_treat_as_complex
from math_verify.utils import timeout

logger = logging.getLogger(__name__)

TIMEOUT_WARNING_SHOWN = False


@dataclass(frozen=True)
class LatexExtractionConfig:
    """Config for extracting latex from the prediction.

    Attributes:
        try_extract_without_anchor (bool): Whether to try extracting latex without requiring specific anchors like "answer:" or "final answer is"
        boxed_match_priority (int): Priority for matching boxed expressions (e.g., \boxed{}).
            - 0: Highest priority (matched first)
            - 50: Default priority (matched after final answer patterns)
            - -1: Disable boxed expression matching
        normalization_config (NormalizationConfig): Configuration for LaTeX normalization.
            Controls preprocessing of LaTeX expressions including:
            - Basic LaTeX cleanup
            - Unit handling
            - Operator formatting
            - Boxed expression extraction
            - Equation parsing
            Defaults to a comprehensive normalization configuration.
    """

    try_extract_without_anchor: bool = True
    boxed_match_priority: int = 50
    normalization_config: NormalizationConfig = field(
        default_factory=lambda: NormalizationConfig(
            basic_latex=True,
            units=True,
            malformed_operators=True,
            nits=True,
            boxed="all",
            equations=False,
        )
    )


@dataclass(frozen=True)
class ExprExtractionConfig:
    """Config for extracting mathematical expressions.

    Attributes:
        try_extract_without_anchor (bool): Whether to try extracting expressions without requiring specific anchors like "answer:" or "final answer is"
    """

    try_extract_without_anchor: bool = True


@dataclass(frozen=True)
class StringExtractionConfig:
    """Config for extracting literal strings.

    Attributes:
        strings (tuple[str]): The strings to extract
        try_extract_without_anchor (bool): Whether to try extracting strings without requiring specific anchors like "answer:" or "final answer is"
    """

    strings: tuple[str, ...] = field(default_factory=lambda: ("A", "B", "C", "D"))
    try_extract_without_anchor: bool = True
    lowercase: bool = True


ExtractionTarget = LatexExtractionConfig | ExprExtractionConfig | StringExtractionConfig


@lru_cache(maxsize=10)
def lazy_string_regex(
    string_extraction_config: StringExtractionConfig,
) -> list[tuple[re.Pattern[str], int]]:
    # First get indices to predict
    string_keys = f"(?P<string_keys>{'|'.join([re.escape(i) for i in string_extraction_config.strings])})"

    # The strings are either surrounded with <space>**answer**., or '<space>answer.' or the same without the dot
    full_stop_re = r"\."
    comma_re = r","
    colon_re = r":"
    space_re = r"\s"

    answer_prefix_re = rf"(^|{space_re})(?:\*\*)?"
    answer_suffix_re = (
        rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|$)"
    )
    answer_re = f"{answer_prefix_re}{string_keys}{answer_suffix_re}"
    answer_re_start = rf"^(?:\*\*)?{string_keys}{answer_suffix_re}"

    answer_word = "(?i:answer)"

    regexes = []

    final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{string_keys}\.?\s?I hope"

    # To allow stuff like "final answer is to your question"
    final_answer_prefixed_just_is = (
        rf"(?i:final answer.{{0,100}}?)\s+is\:?\s*{string_keys}"
    )
    regexes.extend(
        [
            (final_answer_prefixed_re, 0),
            (final_answer_prefixed_just_is, 50),
        ]
    )

    regexes.extend(
        [
            # Most specific patterns first
            (f"{answer_word}{colon_re}.{{0,50}}?{answer_re}", 100),
            # Answer word patterns
            (f"{answer_word}.{{0,50}}?{answer_re}", 200),
        ]
    )

    if string_extraction_config.try_extract_without_anchor:
        # Start of line patterns
        regexes.append((answer_re_start, 250))
        # Plain string patterns
        regexes.append((answer_re, 300))

    return [(re.compile(pattern), priority) for pattern, priority in regexes]


# All of the regexes are cached, to avoid repeated compiling during processing of same task
@lru_cache(maxsize=1)
def lazy_expr_regex(
    expr_config: ExprExtractionConfig,
) -> list[tuple[re.Pattern[str], int]]:
    # Basic number patterns (no LaTeX)
    number_re = (
        # Format 1: Numbers with thousand separators (e.g., "1,234.56" or "1 234.56")
        r"(?<!\d)(?:"
        r"(?P<integer1>-?[1-9]\d{0,2}(?:[ ,]\d{3})+)(?P<decimal1>\.\d+)?|"
        # Format 2: Simple numbers with decimal point or comma (e.g., "123.45" or "123,45")
        r"(?P<integer2>-?\d+)(?P<decimal2>[.,]\d+)|"
        # Format 3: Decimal part only (e.g., ".123")
        r"(?P<decimal3>\.\d+)|"
        # Format 4: Integer only (e.g., "123")
        r"(?P<integer3>-?\d+)"
        r")(?P<percent>\s*(?:%|[Pp]ercent|\s*[Pp]ercentage|\s*[Pp]ct))?"
    )

    # Expressions such as 1/2
    operators = [r"\+", r"\-", r"\*", r"\×", r"\/", r"\^", r"\(", r"\)", r"\÷"]
    operators_re = "".join(operators)
    all_expr_chars = r"[\d\.\s" + operators_re + r"]"
    # Expression should have at minimum at least one operator and must start with a digit
    expr_re = (
        rf"(?P<expr>-?\(?-?\d{all_expr_chars}*[{operators_re}]{all_expr_chars}+\)?)"
    )

    # Punctuation regexes
    full_stop_re = r"\."
    comma_re = r","
    colon_re = r":"
    space_re = r"\s"

    currency_units = re.escape("$€£¥₹₽₪₩₫฿₡₢₣₤₥₦₧₨₩₪₫₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿")
    expr_prefix_re = rf"(?:^|{space_re}|\=)(?:\*\*)?"
    expr_suffix_re = (
        rf"(?:\*\*)?(?:{full_stop_re}|{comma_re}|{colon_re}|{space_re}|\)|\$|$)"
    )
    # Expressions must be prefixed and suffixed while, digits don't need suffix and can have currency units preceeded, this is to ensure
    # That we can extract stuff like $100 or 100m2, while we don't extract XDY2K as 2
    expr_with_anchors = rf"(?:{expr_prefix_re}{expr_re}{expr_suffix_re})"
    number_with_anchors = rf"(?:{expr_prefix_re}[{currency_units}]?{number_re})"
    expr_or_number = rf"(?:{expr_with_anchors}|{number_with_anchors})"
    regexes: list[tuple[str, int]] = []

    final_answer_prefixed_re = (
        rf"(?i:final answer is)\:?\s*{expr_or_number}\.?\s?I hope"
    )
    final_answer_prefixed_just_is = (
        rf"(?i:final answer.{{0,100}}?)\s+is\:?{expr_or_number}"
    )
    regexes.append((final_answer_prefixed_re, 0))
    regexes.append((final_answer_prefixed_just_is, 50))

    answer_prefix_re = r"(?i:answer)"

    # Match after the last equals with answer word - require the number pattern,
    equals_re_colon = rf"{answer_prefix_re}{colon_re}(?:.{{0,100}}=\s*|.{{0,50}}?){expr_or_number}(?!\s*=)"
    equals_re = (
        rf"{answer_prefix_re}(?:.{{0,100}}=\s*|.{{0,50}}?){expr_or_number}(?!\s*=)"
    )
    regexes.extend([(equals_re_colon, 100), (equals_re, 200)])

    if expr_config.try_extract_without_anchor:
        # If everything fails, try to match plain expr/number
        regexes.append((expr_with_anchors, 300))
        regexes.append((number_with_anchors, 300))

    return [(re.compile(pattern), priority) for pattern, priority in regexes]


def make_latex_env_pattern(
    prefix: str = "", context: Literal["boxed", "plain"] = "plain"
) -> str:
    """Creates a LaTeX environment pattern with uniquely prefixed group names.

    Args:
        prefix (str): Prefix to add to group names to make them unique
        context (Literal["boxed", "plain", "fraction"]): Type of content to match inside the environments
            - "boxed": Match environments containing \boxed{...}
            - "plain": Match any LaTeX content
            - "fraction": Match only fractions

    Returns:
        str: Regex pattern for matching LaTeX environments with percent suffix
    """
    percent_re_group = rf"(?P<{prefix}percent>(?:\\?%|[Pp]ercent|[Pp]ercentage|[Pp]ct))"

    # Define base content patterns
    display_dollar_content = r"(?:[^$]|\$(?!\$))"
    # Either \ not followed by ] or everything but \
    display_content_bracket = r"(?:[^\\]|\\(?!\]))"
    inline_dollar_content = r"(?:\\[$]|[^\n$])"
    inline_content_parenthesis = r"(?:[^\\\n]|\\(?!\)))"
    inline_content_bracket = r"[^\n\]\[]"

    if context == "boxed":
        # Rewrite patterns to optionally include boxed content
        display_dollar_content = rf"{display_dollar_content}*?\\boxed{{{display_dollar_content}+?}}{display_dollar_content}*?"
        display_content_bracket = rf"{display_content_bracket}*?\\boxed{{{display_content_bracket}+?}}{display_content_bracket}*?"
        inline_dollar_content = rf"{inline_dollar_content}*?\\boxed{{{inline_dollar_content}+?}}{inline_dollar_content}*?"
        inline_content_parenthesis = rf"{inline_content_parenthesis}*?\\boxed{{{inline_content_parenthesis}+?}}{inline_content_parenthesis}*?"
        inline_content_bracket = rf"{inline_content_bracket}*?\\boxed{{{inline_content_bracket}+?}}{inline_content_bracket}*?"
    else:
        display_dollar_content = rf"{display_dollar_content}+?"
        display_content_bracket = rf"{display_content_bracket}+?"
        inline_dollar_content = rf"{inline_dollar_content}+?"
        inline_content_parenthesis = rf"{inline_content_parenthesis}+?"
        inline_content_bracket = rf"{inline_content_bracket}+?"

    # Build list of regex patterns
    patterns = [
        # Display math environments (allow multiline)
        rf"(?<!\\)\$\$(?P<{prefix}latexDisplayDollar>{display_dollar_content})(?<!\\)\$\$",
        rf"(?<!\\)\\\[(?P<{prefix}latexDisplayBracket>{display_content_bracket})(?<!\\)\\\]",
        # Inline math environments (single line only)
        rf"(?<!\\|\d)\$(?P<{prefix}latexInlineDollar>{inline_dollar_content})(?<!\\)\$",
        rf"(?<!\\)\\\((?P<{prefix}latexInlineParenthesis>{inline_content_parenthesis})(?<!\\)\\\)",
        rf"\s\[(?P<{prefix}latexInlineBracket>{inline_content_bracket})\]\s",
    ]
    if context == "plain":
        simple_number = r"-?\d+(?:[.,]\d+)?"
        patterns.append(
            rf"(?P<{prefix}latexFraction>-?\\frac{{{simple_number}}}{{{simple_number}}})"
        )

    # Join patterns with | and wrap in parentheses
    latex_env_re = rf"(?:(?:{'|'.join(patterns)})\s*{percent_re_group}?)"

    return latex_env_re


@lru_cache(maxsize=1)
def lazy_latex_regex(
    latex_config: LatexExtractionConfig,
) -> list[tuple[re.Pattern[str], int]]:
    # Pattern for multiple latex environments connected by and/or (also considering oxford comma)
    # Create patterns for up to 5 connected expressions
    first_latex_group = make_latex_env_pattern("first_")
    next_groups = "".join(
        [
            rf"(?:\s*(?:,?and|,?or|,)\s*{make_latex_env_pattern(f'next{i}_')})?"
            for i in range(1, 6)
        ]
    )

    latex_envs_re = rf"(?:{first_latex_group}{next_groups})"
    colon_re = r":"
    answer_prefix_re = r"(?i:answer)"

    # We first match boxed env, for some reason that's the most common case of output
    # Then we match the latex with environments, then we try to match the fraction
    regexes: list[tuple[str, int]] = []
    for latex_re in [latex_envs_re]:
        final_answer_prefixed_re = rf"(?i:final answer is)\:?\s*{latex_re}\.?\s?I hope"
        final_answer_prefixed_just_is = (
            rf"(?i:final answer.{{0,100}}?)\s+is\:?\s*{latex_re}"
        )
        regexes.append((final_answer_prefixed_re, 0))
        regexes.append((final_answer_prefixed_just_is, 50))

        # Match with answer word - higher priority than plain latex
        answer_re_colon = f"{answer_prefix_re}{colon_re}.{{0,50}}?{latex_re}"
        answer_re = f"{answer_prefix_re}.{{0,50}}?{latex_re}"

        regexes.extend([(answer_re_colon, 100), (answer_re, 200)])

        # Match plain LaTeX - lowest priority
        if latex_config.try_extract_without_anchor:
            regexes.append((latex_re, 300))

    # This ensures that boxed is matched right after the final answer xxxx
    if latex_config.boxed_match_priority >= 0:
        latex_re_boxed = make_latex_env_pattern(prefix="first_", context="boxed")
        next_groups = "".join(
            [
                rf"\s*(?:\s*(?:,?and|,?or|,)\s*{make_latex_env_pattern(f'next{i}_', context='boxed')})?"
                for i in range(1, 6)
            ]
        )
        latex_re_boxed = rf"{latex_re_boxed}{next_groups}"
        regexes.append((latex_re_boxed, latex_config.boxed_match_priority))
        # Match plain boxed, the issue with plain boxed is that it's impossible to know where it stops, so if there are
        # till last }. We do the actuall extraction in the normalization step.
        regexes.append(
            (r"(?P<first_latexBoxed>\\boxed{.+})", latex_config.boxed_match_priority)
        )

    return [(re.compile(pattern, re.DOTALL), priority) for pattern, priority in regexes]


def get_extraction_regexes(
    target_types: Sequence[ExtractionTarget],
) -> list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]:
    extraction_regexes: list[
        tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]
    ] = [
        (
            (lazy_latex_regex(target_type), target_type)
            if isinstance(target_type, LatexExtractionConfig)
            else (
                (lazy_expr_regex(target_type), target_type)
                if isinstance(target_type, ExprExtractionConfig)
                else (lazy_string_regex(target_type), target_type)
            )
        )
        for target_type in target_types
    ]
    return extraction_regexes


# Small cache, to catche repeated calls invalid parsing
@lru_cache(maxsize=20)
def parse_latex_cached(latex: str):
    # First try to parse the latex as is
    try:
        return latex2sympy(
            latex,
            is_real=not should_treat_as_complex(latex),
            convert_degrees=False,
        )
    except Exception as e:
        # If that fails, try to parse just the last equation
        last_eq_latex = get_last_eq(latex)
        if last_eq_latex != latex:
            return latex2sympy(
                last_eq_latex,
                is_real=not should_treat_as_complex(last_eq_latex),
                convert_degrees=False,
            )
        else:
            raise e


@lru_cache(maxsize=20)
def parse_expr_cached(expr: str):
    return parse_expr(expr, evaluate=False)


def extract_expr(match: re.Match) -> tuple[str | sympy.Expr | None, str]:
    # First combine the number
    groups = match.groupdict()
    # Expr group will always exist because every regex has it
    expr = groups.get("expr", "")
    integer = next(
        (val for name, val in groups.items() if name.startswith("integer") and val), ""
    )
    decimal = next(
        (val for name, val in groups.items() if name.startswith("decimal") and val), ""
    )

    is_percentage = True if groups.get("percent", None) else False

    if integer or decimal:
        # This makes sure we can convert numbers like 0001 to 1. Do note that this can convert 0 to '', so we assume an empty string was 0 and convert it back afterwards.
        integer = integer.translate(str.maketrans("", "", ", ")).lstrip("0")
        if len(integer) == 0:
            integer = "0"

        decimal = decimal.replace(",", ".")
        number_str = f"{integer}{decimal}"
        number = Number(number_str)

        if is_percentage:
            number = convert_to_pct(number)
        return number, number_str

    # Otherwise just return the expression
    # Remove new lines and spaces
    if expr:
        try:
            return (
                parse_expr_cached(expr.replace("\n", " ").replace("^", "**")),
                expr,
            )
        except Exception:
            pass
    return None, expr


def convert_to_pct(number: Number):
    return sympy.Mul(number, sympy.Rational(1, 100), evaluate=False)


equation_split_regex = re.compile(r"(?<!\\|\<|\!|\>)=")


def get_last_eq(latex: str):
    # This is to ensure that a=1,b=2 is not splitted
    if "," not in latex and ";" not in latex:
        eq_parts = equation_split_regex.split(latex)
        # We only shorten if there are more than 2 parts, otherwise we keep equation as is
        if len(eq_parts) > 2:
            return eq_parts[-1]
    return latex


@lru_cache(maxsize=20)
def extract_latex(
    match: re.Match, latex_config: LatexExtractionConfig
) -> tuple[sympy.Expr | str | None, str]:
    latex_exprs = []
    latex_strs = []

    # Get all latex groups (both first_ and nextN_ prefixes)
    first_latex_group = next(
        (
            (val, name)
            for name, val in match.groupdict().items()
            if name.startswith("first_latex") and val
        ),
        None,
    )

    # Get all nextN_ groups
    next_latex_groups = [
        next(
            (
                (val, name)
                for name, val in match.groupdict().items()
                if name.startswith(f"next{i}_latex") and val
            ),
            None,
        )
        for i in range(1, 6)
    ]

    all_latex = list(
        filter(lambda x: x is not None, [first_latex_group] + next_latex_groups)
    )

    for latex, name in all_latex:
        name_without_prefix = name.split("_")[0]
        group_name = name.split("_")[1] if len(name.split("_")) > 1 else None
        is_percentage = (
            True if match.groupdict().get(f"{name_without_prefix}_percent") else False
        )

        # Use modified config if group name is 'boxed'
        config = latex_config.normalization_config
        if group_name == "latexBoxed":
            config = replace(config, boxed="last")  # Use replace to modify single field

        normalized_latex = normalize_latex(
            latex,
            config=config,
        )
        latex_strs.append(normalized_latex)

        try:
            parsed_latex = parse_latex_cached(normalized_latex)
            if is_percentage:
                parsed_latex = convert_to_pct(parsed_latex)
            latex_exprs.append(parsed_latex)
        except Exception:
            latex_exprs.append(None)
            pass

    if not latex_exprs:
        return None, ""

    # If we have multiple expressions and all of them are parsed, wrap them in a Tuple
    if len(latex_exprs) > 1 and all(expr is not None for expr in latex_exprs):
        # To handle solution is: 1,2 and 3
        all_elements = []
        for expr in latex_exprs:
            if isinstance(expr, FiniteSet):
                all_elements.extend(expr.args)
            else:
                all_elements.append(expr)
        return FiniteSet(*all_elements), " and ".join(latex_strs)

    # Otherwise return the single expression
    return latex_exprs[0], latex_strs[0]


def extract_string(match: re.Match, string_config: StringExtractionConfig):
    extracted_str = match.group("string_keys")
    parsed_str = extracted_str
    if string_config.lowercase:
        parsed_str = extracted_str.lower()
    return parsed_str, extracted_str


def extract_match(
    match: re.Match, target_type: ExtractionTarget
) -> tuple[Basic | MatrixBase | str | None, str]:
    """Extracts the match from the regex match.

    Args:
        match (re.Match): The regex match object containing the extracted text
        target_type (ExtractionTarget): The type of extraction to perform (latex, expression, or indices)

    Returns:
        tuple[Basic | MatrixBase | str | None, str]: A tuple containing:
            - The extracted and parsed value (if successful) or None (if parsing failed)
            - The string representation of the extracted text
    """
    if isinstance(target_type, LatexExtractionConfig):
        return extract_latex(match, target_type)
    elif isinstance(target_type, ExprExtractionConfig):
        return extract_expr(match)
    elif isinstance(target_type, StringExtractionConfig):
        return extract_string(match, target_type)


def extract_target_from_pred(
    pred: str,
    target_res: list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]],
    fallback_mode: Literal["no_fallback", "first_match"] = "no_fallback",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
):
    """Extracts targets from a prediction string using regex patterns.
    Returns first sucesffuly extracted match.

    Args:
        pred (str): The prediction string to extract from
        target_res (list[tuple[list[tuple[re.Pattern[str], int]], ExtractionTarget]]): List of regex patterns and their priorities for each target type
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "no_fallback".
            - "no_fallback": Return only successfully parsed match
            - "first_match": Additionaly Include the first string match no matter how parsing finished
        extraction_mode (Literal["first_match", "any_match"], optional): How to handle extraction failures. Defaults to "any_match".
            - "first_match": Only tries to extract the first match
            - "any_match": Tries to extract any match

    Returns:
        list: List of extracted predictions, with first fallbac string appended if fallback_mode is "first_match"
    """
    extracted_predictions = []
    fallbacks = []

    # Get all patterns and sort by priority
    all_patterns = [
        (pattern, target_type, priority)
        for target_patterns, target_type in target_res
        for pattern, priority in target_patterns
    ]

    # Group patterns by priority using itertools.groupby
    match_found = False
    sorted_patterns = sorted(all_patterns, key=lambda x: x[2])
    grouped_patterns = list(
        (gr, list(val)) for gr, val in groupby(sorted_patterns, key=lambda x: x[2])
    )
    for _, patterns_group in grouped_patterns:
        # Find all matches for each pattern in this priority group
        matches_with_pos = (
            (match, match.start(), match.end(), target_type)
            for pattern, target_type, _ in patterns_group
            for match in pattern.finditer(pred)
        )

        # Sort matches by end position (rightmost first) and then by start position (leftmost first)
        matches_with_pos = sorted(
            matches_with_pos, key=lambda x: (x[2], -x[1]), reverse=True
        )

        # Try to extract from each match, starting from rightmost
        for match, _, _, target_type in matches_with_pos:
            extracted_match, str_fallback = extract_match(match, target_type)

            match_found = True
            if str_fallback:
                fallbacks.append(str_fallback)

            if extracted_match is not None:
                extracted_predictions.append(extracted_match)
                break

            if extraction_mode == "first_match":
                break

        # If we extracted something or found something and we're in first_match mode, stop processing other priorities
        if extracted_predictions or (match_found and extraction_mode == "first_match"):
            break

    if fallback_mode == "first_match" and fallbacks:
        extracted_predictions += [fallbacks[0]]

    return extracted_predictions


def parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    parsing_timeout: int = 5,
):
    """Extracts and parses mathematical expressions from a prediction string.

    This function attempts to extract mathematical expressions from text using various strategies
    (LaTeX, plain expressions, etc.) and converts them to SymPy objects.

    Args:
        pred (str): The prediction string to parse.
        extraction_config (Sequence[ExtractionTarget], optional): Configuration for what types of expressions
            to extract and how to extract them. Defaults to [LatexExtractionConfig(), ExprExtractionConfig()].
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "first_match".
            - "no_fallback": Return only successfully parsed expressions
            - "first_match": Include the first string match even if parsing failed
        extraction_mode (Literal["first_match", "any_match"], optional): Strategy for extracting matches. Defaults to "any_match".
            - "first_match": Stop after finding the first match
            - "any_match": Try to extract all possible matches, stops after first sucesful parsing attempt
        parsing_timeout (int, optional): Maximum time in seconds to spend parsing each expression. Defaults to 3. Any timeout seconds > 0 or not None will result in the function to raise a ValueError if it's called in a threaded environment.

    Returns:
        list: List of extracted predictions. Each prediction can be:
            - SymPy expression (for successfully parsed mathematical expressions)
            - String (for fallback matches when fallback_mode="first_match")
            Empty list if no matches are found.

    Examples:
        >>> parse("The answer is $\\frac{1}{2}$")
        [Rational(1, 2)]
        >>> parse("The answer is 1/2")
        [Rational(1, 2)]
        >>> parse("The answer is A", extraction_config=[StringExtractionConfig()])
        ['a']
    """
    global TIMEOUT_WARNING_SHOWN
    if not TIMEOUT_WARNING_SHOWN and (parsing_timeout is None or parsing_timeout <= 0):
        logger.warning(
            "Timeout is disabled as parsing_timeout is None or <= 0, you must provide \
                        the logic for timeout interuption yourself to prevent code getting stuck."
        )
        TIMEOUT_WARNING_SHOWN = True

    try:
        target_res = get_extraction_regexes(extraction_config)
        return timeout(timeout_seconds=parsing_timeout)(extract_target_from_pred)(
            pred,
            target_res,
            fallback_mode=fallback_mode,
            extraction_mode=extraction_mode,
        )
    except ValueError as e:
        # Check if it's the signal error
        if str(e) == "signal only works in main thread of the main interpreter":
            raise ValueError(
                "Math-Verify 'parse' function doesn't support threaded environment due to usage of signal.alarm() in timeout mechanism. If you need to run in multithreaded environment it's recommended to set the parsing_timeout=None, which will run without timeout (and signal handling). In this case you need to handle the timeouting yourself."
            ) from e
        logger.exception(f"Error parsing: {pred[:10]}")
        return []
    except Exception:
        logger.exception(f"Error parsing: {pred[:10]}")
        return []
    except TimeoutException:
        logger.error(f"Timeout during parsing: {pred[:10]}")
        return []
