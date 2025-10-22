import re
from dataclasses import dataclass
from typing import Literal
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class NormalizationConfig:
    """Configuration for latex normalization.
    
    Each field controls a group of related normalizations:
    - basic_latex: Basic latex command replacements (mathrm, displaystyle, etc.)
    - units: Remove units and their variations
    - malformed_operators: Fix malformed operators (sqrt, frac, etc.)
    - nits: Small formatting fixes (spaces, dots, etc.)
    - boxed: Extract content from boxed environments
    - equations: Handle equation splitting and approximations (deprecated)
    """
    basic_latex: bool = True
    units: bool = False
    malformed_operators: bool = False
    nits: bool = False
    boxed: Literal["all", "last", "none"] = "all"
    equations: bool = False

# Compile all regex patterns once at module level
r_left = re.compile(r"\\m?left(\\\{|\{|\\\||\||\[|\(|\\rbracl|\\lgroup|\\lbrace|\\lbrack|\\vert|\\lvert|\\lceil|\\lfloor|\\vert|\\lvert|\\langle|\\llcorner|\\ulcorner)")
r_right = re.compile(r"\\m?right(\\\}|\}|\\\||\||\]|\)|\\rbrack|\\rgroup|\\rbrace|\\rbrack|\\vert|\\rvert|\\rceil|\\rfloor|\\vert|\\rvert|\\rangle|\\lrcorner|\\urcorner)")

# Units regex
units = [
    "integer" "point",
    "feet",
    "sue",
    "digit",
    "pound",
    "meal",
    "edge",
    "student",
    "children ticket",
    "multiple",
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m square",
    " m east",
    "sq m",
    "deg",
    "mile",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "cent",
    "by",
    "gal",
    "kmh",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "month",
    "km",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
    "kilogram",
    "second",
    "ampere",
    "A",
    "K",
    "mol",
    "cd",
    "N",
    "J",
    "W",
    "Pa",
    "Hz",
    "C",
    "V",
    "Ω",
    "F",
    "T",
    "H",
    "eV",
    "kW·h",
    "atm",
    "bar",
    "°C"
]

# We sort here to that when matching from right the longest units are matched first
# E.g "percent" is matched before "cent"

units_regex_pattern = f"(?:{'|'.join(units)})(?:s|es)?"
units_regex = re.compile(f"(\\d|\\}}|\\s)\\s*(?:{units_regex_pattern})\\s*$")

# Basic latex regex
to_remove_regex = re.compile(
    r"\\mathrm\{th\}|"  # "th"
    r"\\!\s*|"  # comma with inverse space
    r"\\text\s*\{\s*\}|" # text with empty braces
    r"\\text\s*\{\s*\}|" # text with empty braces
    r"\\\$|\$|"  # dollar signs
    r"(?<!\\)[\"\']|"  # quotes
    # to display
    r"\\displaystyle"
)

# Text replacement patterns
to_replace_patterns = [
    # (name, pattern, replacement)
    # Not really needed only for units
    ("math", r"\\math(?:rm|it|bf)", r"\text"),
    ("text", r"\\text(?:normal|bf|it|rm)", r"\text"),
    ("frac", r"\\(?:d|t|c)frac", r"\frac"),
    ("decimal_space", r"\s\.", r" 0."),
    ("decimal_brace", r"\{\.", r"{0."),
    ("approx", r"\~\=", r"\approx"),
    ("comma", r"\s*\{\s*,\s*\}", r","),
    ("and_or", r"(?<![a-zA-Z])(,?\s*(?:and|or))(?![a-zA-Z])", r","),
    ("and_or_text", r"(,?\s*\\text{\s*(?:and|or)\s*})", r","),
    ("backslash_space", r"(?<!\\)\\\s", r" "),
    # Empty text
    ("infinity", r"infinity", r"\infty"),
    # Dots
    ("dot", r",?(\\ldots)", r" "),
    ("percent", r"\s*percent", r"\\%"),
    ("percent_in_text", r"\\text{percent}", r"\\%"),
    ("inf", r"((?<!\\)inf(?!inity))", r"\infty"),
    ("sqrt", r" sqrt", r"\sqrt"),
]

# Create regex with named groups
pattern = "|".join(f"(?P<{name}>{pattern})" for name, pattern, _ in to_replace_patterns)
to_replace_regex = re.compile(pattern)

# Create lookup dictionary for replacements
replacements = {name: replacement for name, _, replacement in to_replace_patterns}

command_slash_fix_regex = re.compile(r"\\\\(?=[a-zA-Z])")
permutation_regex = re.compile(r"\(([a-zA-Z0-9+\-*/\\ ]+?)\)_{([a-zA-Z0-9+\-*/\\ ]+?)}")
equation_split_regex = re.compile(r"(?<!\\|\<|\!|\>)=")
unit_superscript_regex = re.compile(r"(\\(?:text|mbox){.*?})(\^\d|\{\^\d\})?$")
approx_split_regex = re.compile(r"\\approx")

# Malformed operators regex
malformed_operators_patterns = [
    (re.compile(r"\^\s?\((.*?)\)"), r"^{\1}"),
    (re.compile(r"sqrt\s?\((.*?)\)"), r"\\sqrt{\1}"),
    (re.compile(r"\\frac\s?(\d)\s?(\d+)"), r"\\frac{\1}{\2}"),
    (re.compile(r"\\log_\s?(\d)\s?(\d+)"), r"\\log_{\1}{\2}"),
    (re.compile(r"\\frac\s?{(.*?)}\s?(\d)"), r"\\frac{\1}{\2}"),
    (re.compile(r"\\frac\s?(\d)\s?{(.*?)}"), r"\\frac{\1}{\2}"),
    (re.compile(r"\\sqrt\s?(\d)"), r"\\sqrt{\1}")
]

def _fix_malformed_operators(text: str) -> str:
    """Fix malformed operators in the given text."""
    expr_str = text
    for pattern, replacement in malformed_operators_patterns:
        expr_str = pattern.sub(replacement, expr_str)
    expr_str = expr_str.replace(" sqrt", "\\sqrt")
    return expr_str

def replace(match):
    # Find which group matched
    # Get corresponding replacement from dict
    return replacements[match.lastgroup]

def replace_in_latex(text: str) -> str:
    return to_replace_regex.sub(replace, text)

VALID_SEPARATOR_PATTERN = re.compile(r'and|or|,|;')
def extract_boxed_content(text: str, mode: Literal["last", "all"] = "last") -> str:
    """
    Find and extract all \\boxed{...} or \\fbox{...} elements from a string, searching from right to left.
    If mode is "last", return content up to the last valid separator.
    If mode is "all", return all boxed contents joined by commas.
    """
    
    def find_content_boundaries(text: str, opening_brace_pos: int, max_pos: int) -> tuple[int, int] | None:
        # Start searching for closing brace from the opening brace position
        i = opening_brace_pos
        num_left_braces_open = 1  # We start after the opening brace
        
        while i + 1 < max_pos:  # Check if next position is within bounds and max_pos
            i += 1
            if text[i] == "{":
                num_left_braces_open += 1
            elif text[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    return opening_brace_pos, i
        return None
    
    def has_valid_separator(text: str, content_end: int, next_boxed_start: int) -> bool:
        between_text = text[content_end + 1:next_boxed_start]
        # Making regex for it not worth it so this works
        return len(between_text) < 70 and bool(VALID_SEPARATOR_PATTERN.search(between_text))
    
    results = []
    current_pos = len(text)
    last_boxed_start = None
    
    max_pos = len(text)
    while True:
        boxed_idx = text.rfind("\\boxed", 0, current_pos)
        fbox_idx = text.rfind("\\fbox", 0, current_pos)
        
        if boxed_idx < 0 and fbox_idx < 0:
            break
            
        start_idx = max(boxed_idx, fbox_idx)
        command_end = start_idx + (6 if boxed_idx > fbox_idx else 5)
        
        # Find opening brace
        next_char_pos = command_end
        while next_char_pos < max_pos and text[next_char_pos].isspace():
            next_char_pos += 1
            
        if next_char_pos >= max_pos:
            break
            
        if text[next_char_pos] == "{":
            boundaries = find_content_boundaries(text, next_char_pos, max_pos)
            if not boundaries:
                # This is our last box
                if len(results) == 0:
                    results.append(text[next_char_pos:])
                break
            content_start, content_end = boundaries
            content = text[content_start + 1:content_end].strip()
            
            if mode == "last" and last_boxed_start is not None:
                if not has_valid_separator(text, content_end, last_boxed_start):
                    break
            
            results.append(content)
            last_boxed_start = start_idx
            max_pos = start_idx
        else:
            # This is our last box
            if len(results) == 0:
                results.append(text[next_char_pos:])
            # Otherwise we just ignore it
            break
            
        
        current_pos = start_idx
    
    if not results:
        return text
        
    return ",".join(reversed(results))

def _fix_fracs(text: str) -> str:
    """
    Fix the formatting of fractions in the given text.
    Copied from: https://github.com/hendrycks/math/blob/357963a7f5501a6c1708cf3f3fb0cdf525642761/modeling/math_equivalence.py#L1

    Args:
        text (str): The input text.

    Returns:
        str: The text with properly formatted fractions.

    Examples:
        >>> _fix_fracs("\\frac12")
        "\\frac{1}{2}"
        >>> _fix_fracs("\\frac{3}{4}")
        "\\frac{3}{4}"
        >>> _fix_fracs("\\frac1{2}")
        "\\frac{1}{2}"
    """
    substrs = text.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            # This allows use to have \\frac{1}{2} and \\ frac1{2}
            substr = substr.lstrip()
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr

            elif len(substr) < 2:
                return text
            else:
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    text = new_str
    return text

def _fix_a_slash_b(text: str) -> str:
    """Source: https://github.com/hendrycks/math
    Reformat fractions formatted as a/b to \\frac{a}{b}.
    Example:
    >>> _fix_a_slash_b("2/3")
    \frac{2}{3}
    """
    if len(text.split("/")) != 2:
        return text
    a_str = text.split("/")[0]
    b_str = text.split("/")[1]
    try:
        a = int(a_str)
        b = int(b_str)
        assert text == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return text


def _fix_sqrt(text: str) -> str:
    """Source: https://github.com/hendrycks/math
    Reformat square roots.
    Example:
    >>> _fix_sqrt("\\sqrt3")
    \\sqrt{3}
    """
    if "\\sqrt" not in text:
        return text
    splits = text.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        split = split.lstrip()
        if len(split) > 0 and split[0] not in ["{", "["]:
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def normalize_latex(text: str, config: NormalizationConfig) -> str:
    """Normalize latex string according to the provided configuration.
    
    Args:
        text: The latex string to normalize
        config: Configuration controlling which normalizations to apply
        
    Returns:
        The normalized latex string
    """
    if config.boxed == "all" or config.boxed == "last":
        text = extract_boxed_content(text, mode=config.boxed)

    if config.basic_latex:
        # Basic latex command replacements
        text = text.replace(r'\mathrm{T}', 'T')
        text = text.replace(r'\mathrm{d}', 'd').replace(r'{\rm d}', 'd')
        text = text.replace(r'\left[\begin{matrix}', r'\begin{bmatrix}').replace(r'\end{matrix}\right]', r'\end{bmatrix}')
        text = r_left.sub(r'\1', text)
        text = r_right.sub(r'\1', text)
        text = permutation_regex.sub(r"\\frac{(\1)!}{((\1)-(\2))!}", text)
        
        # Remove useless latex commands
        text = to_remove_regex.sub("", text)
        text = replace_in_latex(text)
        
        # Remove new lines and simplify tabs
        text = text.replace("\n", " ").replace("\t", " ")
        
        # Fix doubled backslashes in commands
        if "matrix" not in text:
            text = command_slash_fix_regex.sub(r"\\", text)
    
    if config.equations:
        logger.warning("equations is deprecated, as it handled by the parser now")
        # This is to ensure that a=1,b=2 is not splitted
        if not "," in text and not ";" in text:
            eq_parts = equation_split_regex.split(text)
            # We only shorten if there are more than 2 parts, otherwise we keep equation as is
            if len(eq_parts) > 2:
                text = eq_parts[-1]
    
    if config.units:
        # Remove the units and possibly the superscript
        _text = unit_superscript_regex.sub("", text).strip()
        if _text != "" and _text != text:
            text = _text
            
        # Remove unit texts
        for _ in range(2):
            _text = units_regex.sub(r"\1", text)
            if _text != "" and _text != text:
                text = _text
        
        # This can trigger empty \text{...}
        # Make sure not to remove space this created
    
    if config.nits:
        # Fix leading decimal
        if len(text) > 0 and text[0] == ".":
            text = "0" + text
            
        # Fix 0.5 to fraction
        if text == "0.5":
            text = "\\frac{1}{2}"
    
    if config.malformed_operators:
        # Fix malformed operators
        text = _fix_malformed_operators(text)
        text = _fix_sqrt(text)
        text = _fix_fracs(text)
        text = _fix_a_slash_b(text)
    
    return text.strip()