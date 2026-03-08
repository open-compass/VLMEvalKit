# Credit to https://github.com/KameniAlexNea/llm-output-parser, MIT LICENSE

import json
import logging
import re


def parse_json(
    json_str: str,
    allow_incomplete: bool = True,
    strict: bool = False,
    parse_nested_strings: bool = False,
):
    """
    Parses a JSON object from a string that may contain extra text.

    This function attempts multiple approaches to extract JSON:

    1. Directly parsing the entire string.
    2. Extracting JSON enclosed within triple backticks (```json ... ```).
    3. Extracting all valid JSON objects or arrays with balanced delimiters.
    4. If allow_incomplete=True, attempts to repair incomplete/truncated JSON.

    :param json_str: The input string potentially containing a JSON object.
    :type json_str: str
    :param allow_incomplete: Whether to attempt repairing incomplete JSON.
    :type allow_incomplete: bool
    :param strict: Whether to raise errors on failures (when False, returns None on failure).
    :type strict: bool
    :param parse_nested_strings: Whether to parse string values that look like JSON.
    :type parse_nested_strings: bool
    :return: The parsed JSON object if successfully extracted, otherwise None.
    :rtype: dict or list or None
    """
    try:
        resp = json.loads(json_str)
        return resp
    except:
        pass

    try:
        resp = eval(json_str)
        return resp
    except:
        pass

    if '```json' in json_str:
        content = json_str.split('```json')[1].split('```')[0].strip()
        return parse_json(content)

    _validate_input(json_str)
    if '\\' in json_str:
        json_str = json_str.replace('\\', '\\\\')

    # Collect all possible JSON candidates
    candidates = _collect_json_candidates(json_str, allow_incomplete)

    if candidates:
        # Return the best candidate based on complexity
        result = _select_best_candidate(candidates)

        # Parse nested JSON strings if requested
        if parse_nested_strings and result is not None:
            result = _parse_nested_json_strings(result)

        return result
    else:
        return _handle_no_candidates(strict)


def _validate_input(json_str):
    """Validate the input parameters."""
    if json_str is None or not isinstance(json_str, str):
        raise TypeError("Input must be a non-empty string.")
    if not json_str:
        raise ValueError("Input string is empty.")


def _collect_json_candidates(json_str: str, allow_incomplete: bool):
    """Collect all possible JSON candidates from various extraction methods."""
    candidates = []

    # Direct parsing
    _try_direct_parse(json_str, candidates)

    # Code block extraction
    _extract_from_code_blocks(json_str, candidates)

    # Balanced delimiter extraction
    _extract_with_balanced_delimiters(json_str, candidates)

    # Incomplete JSON repair (if enabled)
    if allow_incomplete:
        _extract_repaired_json(json_str, candidates)

    return candidates


def _try_direct_parse(json_str: str, candidates: list):
    """Attempt to parse the entire string as JSON."""
    try:
        parsed = json.loads(json_str)
        candidates.append((parsed, json_str))
    except json.JSONDecodeError:
        pass


def _extract_from_code_blocks(json_str: str, candidates: list):
    """Extract JSON from code blocks delimited by triple backticks."""
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    for match in re.finditer(code_block_pattern, json_str):
        json_block = match.group(1)
        _try_parse_json_string(json_block, candidates)


def _extract_with_balanced_delimiters(json_str: str, candidates: list):
    """Extract JSON using balanced delimiter matching."""
    _extract_json_objects(json_str, "{", "}", candidates)
    _extract_json_objects(json_str, "[", "]", candidates)


def _extract_repaired_json(json_str: str, candidates: list):
    """Extract and repair incomplete JSON."""
    repaired_candidates = _attempt_json_repair(json_str)
    candidates.extend(repaired_candidates)


def _select_best_candidate(candidates):
    """Select the best JSON candidate based on serialized JSON length."""

    def length_key(item):
        parsed_obj, _ = item
        # Simple and clear: just compare the length of the serialized JSON
        return len(json.dumps(parsed_obj))

    sorted_candidates = sorted(candidates, key=length_key, reverse=True)
    return sorted_candidates[0][0]


def _handle_no_candidates(strict: bool):
    """Handle the case when no JSON candidates are found."""
    if strict:
        raise ValueError("Failed to parse JSON from the input string.")
    else:
        return None


def _try_parse_json_string(json_str: str, candidates: list):
    """Try to parse a JSON string with various cleaning approaches."""
    # Try direct parsing first
    if _try_parse_and_add(json_str, candidates):
        return

    # Try with basic cleaning
    if _try_parse_with_basic_cleaning(json_str, candidates):
        return

    # Try with comprehensive cleaning
    if _try_parse_with_comprehensive_cleaning(json_str, candidates):
        return

    # Try with control character handling
    _try_parse_with_control_char_handling(json_str, candidates)


def _try_parse_and_add(json_str: str, candidates: list) -> bool:
    """Try to parse JSON string directly and add to candidates if successful."""
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, (dict, list)):
            candidates.append((parsed, json_str))
            return True
    except json.JSONDecodeError:
        pass
    return False


def _try_parse_with_basic_cleaning(json_str: str, candidates: list) -> bool:
    """Try parsing with basic comment removal and comma cleanup."""
    try:
        # Remove comments and trailing commas
        cleaned = _remove_basic_comments(json_str)
        cleaned = _remove_trailing_commas(cleaned)

        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            candidates.append((parsed, cleaned))
            return True
    except json.JSONDecodeError:
        pass
    return False


def _try_parse_with_comprehensive_cleaning(json_str: str, candidates: list) -> bool:
    """Try parsing with comprehensive comment removal."""
    try:
        cleaned = _remove_comments_comprehensive(json_str)
        cleaned = _remove_trailing_commas(cleaned)

        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            candidates.append((parsed, cleaned))
            return True
    except json.JSONDecodeError:
        pass
    return False


def _try_parse_with_control_char_handling(json_str: str, candidates: list):
    """Try parsing with control character escaping."""
    try:
        escaped = _escape_control_characters(json_str)
        escaped = _remove_comments_comprehensive(escaped)
        escaped = _remove_trailing_commas(escaped)

        parsed = json.loads(escaped)
        if isinstance(parsed, (dict, list)):
            candidates.append((parsed, escaped))
    except (json.JSONDecodeError, Exception):
        pass


def _remove_basic_comments(text: str) -> str:
    """Remove basic JavaScript-style comments."""
    # Remove multi-line comments
    text = re.sub(r"/\*[\s\S]*?\*/", "", text, flags=re.DOTALL)
    # Remove single-line comments
    text = re.sub(r"//.*?(?:\n|$)", "", text, flags=re.MULTILINE)
    return text


def _remove_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing brackets/braces."""
    return re.sub(r",\s*([\]}])", r"\1", text)


def _escape_control_characters(json_str: str) -> str:
    """Escape control characters in JSON string."""
    control_char_map = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}

    # Avoid double escaping
    result = json_str
    for char, escape in control_char_map.items():
        placeholder = f"__PLACEHOLDER_{ord(char)}__"
        result = result.replace(escape, placeholder)

    for char, escape in control_char_map.items():
        result = result.replace(char, escape)

    for char, escape in control_char_map.items():
        placeholder = f"__PLACEHOLDER_{ord(char)}__"
        result = result.replace(placeholder, escape)

    return result


def _json_structure_depth(obj):
    """
    Calculate the depth of a JSON structure.

    :param obj: The JSON object (dict or list)
    :return: The maximum nesting depth
    """
    if isinstance(obj, dict):
        if not obj:
            return 1
        return 1 + max(_json_structure_depth(v) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return 1
        return 1 + max(_json_structure_depth(item) for item in obj)
    else:
        return 0


def _extract_json_objects(
    text: str, open_delimiter: str, close_delimiter: str, results: list
):
    """Extract JSON objects/arrays with balanced delimiters."""
    i = 0
    while i < len(text):
        start = text.find(open_delimiter, i)
        if start == -1:
            break

        end_pos = _find_balanced_delimiter_end(
            text, start, open_delimiter, close_delimiter
        )

        if end_pos is not None:
            json_str = text[start:end_pos]
            _try_parse_json_string(json_str, results)
            i = end_pos
        else:
            i = start + 1


def _find_balanced_delimiter_end(
    text: str, start: int, open_delim: str, close_delim: str
):
    """Find the end position of a balanced delimiter structure."""
    balance = 1
    pos = start + 1
    in_string = False
    escape_char = False

    while pos < len(text) and balance > 0:
        char = text[pos]

        if char == '"' and not escape_char:
            in_string = not in_string
        elif not in_string:
            if char == open_delim:
                balance += 1
            elif char == close_delim:
                balance -= 1

        escape_char = char == "\\" and not escape_char
        pos += 1

    return pos if balance == 0 else None


def _remove_comments_comprehensive(text):
    """
    Comprehensively removes both single-line and multi-line JavaScript style comments.
    Handles complex cases like nested comments and comments inside strings.

    :param text: The JSON text to clean
    :return: Text with all comments removed
    """
    result = []
    i = 0
    in_string = False
    in_single_comment = False
    in_multi_comment = False
    escape_next = False

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if (
            char == '"'
            and not escape_next
            and not in_single_comment
            and not in_multi_comment
        ):
            in_string = not in_string
            result.append(char)
        elif char == "\\" and in_string and not escape_next:
            escape_next = True
            result.append(char)
        elif (
            char == "/"
            and next_char == "/"
            and not in_string
            and not in_single_comment
            and not in_multi_comment
        ):
            in_single_comment = True
            i += 1  # Skip the next '/' character
        elif char == "\n" and in_single_comment:
            in_single_comment = False
            result.append(char)  # Keep the newline
        elif (
            char == "/"
            and next_char == "*"
            and not in_string
            and not in_single_comment
            and not in_multi_comment
        ):
            in_multi_comment = True
            i += 1  # Skip the next '*' character
        elif char == "*" and next_char == "/" and in_multi_comment:
            in_multi_comment = False
            i += 1  # Skip the next '/' character
        elif not in_single_comment and not in_multi_comment:
            result.append(char)

        if escape_next:
            escape_next = False
        i += 1

    return "".join(result)


def _attempt_json_repair(json_str: str):
    """Attempt to repair incomplete or truncated JSON strings."""
    candidates = []

    # Try repairing code blocks
    _repair_code_blocks(json_str, candidates)

    # Try repairing the entire string
    _repair_json_string(json_str, candidates)

    return candidates


def _repair_code_blocks(json_str: str, candidates: list):
    """Extract and repair JSON from code blocks."""
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    for match in re.finditer(code_block_pattern, json_str):
        json_block = match.group(1)
        _repair_json_string(json_block, candidates)


def _repair_json_string(json_str: str, candidates: list):
    """Apply various repair strategies to a JSON string."""
    cleaned = json_str.strip()
    if not cleaned:
        return

    # Try repairing based on the starting character
    if cleaned.startswith("{"):
        _repair_incomplete_structure(cleaned, "{", "}", candidates)
    elif cleaned.startswith("["):
        _repair_incomplete_structure(cleaned, "[", "]", candidates)

    # Look for partial structures within the text
    _extract_and_repair_partial_json(cleaned, candidates)


def _repair_incomplete_structure(
    json_str: str, open_char: str, close_char: str, candidates: list
):
    """Repair incomplete JSON objects or arrays."""
    # Clean and count delimiters
    cleaned = _clean_json_for_repair(json_str)
    open_count = cleaned.count(open_char)
    close_count = cleaned.count(close_char)

    if open_count <= close_count:
        return  # Already balanced or over-closed

    # Try sophisticated repair first
    repaired = _complete_unfinished_pairs(cleaned, close_char)
    if repaired and _try_parse_repaired(repaired, candidates):
        return

    # Fallback to simple closing
    simple_repair = cleaned + close_char * (open_count - close_count)
    _try_parse_repaired(simple_repair, candidates)


def _clean_json_for_repair(json_str: str) -> str:
    """Clean JSON string for repair operations."""
    cleaned = _remove_comments_comprehensive(json_str)
    return _remove_trailing_commas(cleaned)


def _try_parse_repaired(repaired: str, candidates: list) -> bool:
    """Try to parse repaired JSON and add to candidates if successful."""
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, (dict, list)):
            candidates.append((parsed, repaired))
            struct_type = "object" if isinstance(parsed, dict) else "array"
            logging.info(
                f"Successfully repaired incomplete JSON {struct_type}: {repaired[:100]}..."
            )
            return True
    except json.JSONDecodeError:
        pass
    return False


def _complete_unfinished_pairs(json_str: str, close_char: str):
    """
    Attempts to complete unfinished key-value pairs or array elements.

    :param json_str: The JSON string to repair
    :param close_char: The closing character ('}' or ']')
    :return: Repaired JSON string or None
    """
    # Look for common incomplete patterns
    patterns_and_replacements = [
        # Incomplete key without value: {"key":
        (r'("[^"]*")\s*:\s*$', r"\1: null"),
        # Incomplete key without colon: {"key"
        (r'("[^"]*")\s*$', r"\1: null"),
        # Incomplete key without quotes: {"key
        (r'\{\s*"?([^",:}]+)"?\s*$', r'{"key": null'),
        # Trailing comma: {"key": "value",
        (r",\s*$", ""),
        # Incomplete nested object: {"key": {
        (r":\s*\{\s*$", ": {}"),
        # Incomplete nested array: {"key": [
        (r":\s*\[\s*$", ": []"),
        # Incomplete string value: {"key": "value
        (r':\s*"([^"]*)\s*$', r': "\1"'),
    ]

    repaired = json_str.rstrip()

    for pattern, replacement in patterns_and_replacements:
        if re.search(pattern, repaired):
            repaired = re.sub(pattern, replacement, repaired)
            break

    # Add closing character if needed
    if close_char == "}":
        open_count = repaired.count("{")
        close_count = repaired.count("}")
    else:  # close_char == ']'
        open_count = repaired.count("[")
        close_count = repaired.count("]")

    if open_count > close_count:
        repaired += close_char * (open_count - close_count)

    return repaired if repaired != json_str else None


def _extract_and_repair_partial_json(text: str, candidates: list):
    """Extract and repair partial JSON structures from text."""
    # Look for potential JSON start patterns
    patterns = [
        (r"\{[^}]*$", "{", "}"),  # Object that doesn't close
        (r"\[[^\]]*$", "[", "]"),  # Array that doesn't close
    ]

    for pattern, open_char, close_char in patterns:
        for match in re.finditer(pattern, text, re.DOTALL):
            partial = match.group(0).strip()
            if partial:
                _repair_incomplete_structure(partial, open_char, close_char, candidates)


def _parse_nested_json_strings(obj):
    """
    Recursively parse string values that look like JSON into their actual JSON types.

    :param obj: The JSON object to process
    :return: The processed object with nested JSON strings parsed
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(value, str):
                # Try to parse the string as JSON
                parsed_value = _try_parse_json_string_value(value)
                result[key] = parsed_value if parsed_value is not None else value
            else:
                # Recursively process nested structures
                result[key] = _parse_nested_json_strings(value)
        return result
    elif isinstance(obj, list):
        result = []
        for item in obj:
            if isinstance(item, str):
                # Try to parse the string as JSON
                parsed_item = _try_parse_json_string_value(item)
                result.append(parsed_item if parsed_item is not None else item)
            else:
                # Recursively process nested structures
                result.append(_parse_nested_json_strings(item))
        return result
    else:
        # Return primitive values unchanged
        return obj


def _try_parse_json_string_value(value: str):
    """
    Try to parse a string value as JSON if it looks like JSON.

    :param value: The string value to potentially parse
    :return: Parsed JSON object/array if successful, None otherwise
    """
    # Skip obviously non-JSON strings
    if not value or len(value) < 2:
        return None

    # Only try to parse strings that look like JSON
    stripped = value.strip()
    if not (
        (stripped.startswith("{") and stripped.endswith("}"))
        or (stripped.startswith("[") and stripped.endswith("]"))
    ):
        return None

    try:
        parsed = json.loads(stripped)
        # Only return the parsed result if it's a dict or list
        if isinstance(parsed, (dict, list)):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    return None
