import json
import re


def parse_jsons(json_str: str):
    """
    Parses all JSON objects from a string that may contain extra text.

    This function attempts several approaches to extract JSON:

    1. Directly parsing the entire string.
    2. Extracting JSON enclosed within triple backticks (```json ... ```).
    3. Extracting all valid JSON objects or arrays with balanced delimiters.

    :param json_str: The input string potentially containing JSON objects.
    :type json_str: str
    :return: A list of all parsed JSON objects found in the input string.
    :rtype: list of dict or list
    """
    if json_str is None or not isinstance(json_str, str):
        raise TypeError("Input must be a non-empty string.")
    if not json_str:
        raise ValueError("Input string is empty.")

    # Store all successfully parsed JSON objects with their span coordinates
    parsed_jsons_with_spans = []

    # Attempt 1: Try to load the entire string as JSON.
    try:
        parsed = json.loads(json_str)
        parsed_jsons_with_spans.append((parsed, 0, len(json_str)))
    except json.JSONDecodeError:
        pass

    # Attempt 2: Look for JSON blocks delimited by ```json and ```.
    # Find all code blocks and try to parse each one
    code_block_matches = re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", json_str)
    for match in code_block_matches:
        json_block = match.group(1)
        try:
            parsed = json.loads(json_block)
            # Store with the position in the original string
            block_start = match.start()
            block_end = match.end()
            parsed_jsons_with_spans.append((parsed, block_start, block_end))
        except json.JSONDecodeError:
            pass

    # Attempt 3: Extract JSON objects with balanced delimiters
    _extract_json_objects(json_str, "{", "}", parsed_jsons_with_spans)

    # Attempt 4: Extract JSON arrays with balanced delimiters
    _extract_json_objects(json_str, "[", "]", parsed_jsons_with_spans)

    if not parsed_jsons_with_spans:
        raise ValueError("Failed to parse any JSON from the input string.")

    # Remove overlapping JSON objects (keep non-overlapping ones)
    non_overlapping = _remove_overlapping_jsons(parsed_jsons_with_spans)

    # Return just the parsed JSON objects without the spans
    return [item[0] for item in non_overlapping]


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
    """
    Extracts all valid JSON objects or arrays from the text with properly balanced delimiters.

    :param text: The text to search in
    :param open_delimiter: Opening delimiter ('{' or '[')
    :param close_delimiter: Closing delimiter ('}' or ']')
    :param results: List to append results to (tuple of (parsed_json, start_pos, end_pos))
    """
    i = 0
    while i < len(text):
        # Find the next opening delimiter
        start = text.find(open_delimiter, i)
        if start == -1:
            break

        # Track balanced delimiters
        balance = 1
        pos = start + 1
        in_string = False
        escape_char = False

        # Scan for the matching closing delimiter
        while pos < len(text) and balance > 0:
            char = text[pos]

            # Handle string literals (ignore delimiters inside strings)
            if char == '"' and not escape_char:
                in_string = not in_string
            elif not in_string:
                if char == open_delimiter:
                    balance += 1
                elif char == close_delimiter:
                    balance -= 1

            # Track escape characters
            if char == "\\" and not escape_char:
                escape_char = True
            else:
                escape_char = False

            pos += 1

        # If we found a balanced object
        if balance == 0:
            # Extract the object string including delimiters
            json_str = text[start:pos]

            # Try multiple parsing approaches
            _try_parse_with_approaches(json_str, start, pos, results)

        # Move to position after the current match to look for more
        i = pos if balance == 0 else start + 1


def _try_parse_with_approaches(
    json_str: str, start_pos: int, end_pos: int, results: list
):
    """
    Attempts to parse a JSON string using multiple approaches.

    :param json_str: The JSON string to parse
    :param start_pos: The starting position of the JSON string in the original text
    :param end_pos: The ending position of the JSON string in the original text
    :param results: List to append results to with position information
    """
    # Approach 1: Direct parsing
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, (dict, list)):
            results.append((parsed, start_pos, end_pos))
        return  # Successfully parsed, no need to try other approaches
    except json.JSONDecodeError:
        pass

    # Approach 2: Clean up common formatting issues
    try:
        # Remove JavaScript-style comments (improved for multi-line handling)
        uncleaned = json_str

        # First, handle multi-line comments (/* ... */)
        # This pattern uses a non-greedy match to properly handle nested structures
        cleaned = re.sub(r"/\*[\s\S]*?\*/", "", uncleaned, flags=re.DOTALL)

        # Then handle single-line comments
        cleaned = re.sub(r"//.*?(?:\n|$)", "", cleaned, flags=re.MULTILINE)

        # Remove trailing commas in objects and arrays
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            results.append((parsed, start_pos, end_pos))
        return  # Successfully parsed after cleaning
    except json.JSONDecodeError:
        pass

    # Approach 3: More aggressive cleaning if the above failed
    try:
        # Try a more comprehensive cleaning approach
        # First remove all comments completely
        aggressive_cleaned = _remove_comments_comprehensive(json_str)

        # Then fix trailing commas
        aggressive_cleaned = re.sub(r",\s*([\]}])", r"\1", aggressive_cleaned)

        parsed = json.loads(aggressive_cleaned)
        if isinstance(parsed, (dict, list)):
            results.append((parsed, start_pos, end_pos))
        return  # Successfully parsed after aggressive cleaning
    except json.JSONDecodeError:
        pass

    # Approach 4: Manual handling of control characters
    try:
        # Replace literal control characters with their proper JSON escape sequences
        control_char_map = {
            "\b": "\\b",  # backspace
            "\f": "\\f",  # form feed
            "\n": "\\n",  # line feed
            "\r": "\\r",  # carriage return
            "\t": "\\t",  # tab
        }

        # First, unescape any already escaped sequences to avoid double escaping
        unescaped = json_str
        for char, escape in control_char_map.items():
            # Replace the escaped version with a placeholder
            placeholder = f"__PLACEHOLDER_{ord(char)}__"
            unescaped = unescaped.replace(escape, placeholder)

        # Then replace actual control characters with proper escapes
        for char, escape in control_char_map.items():
            unescaped = unescaped.replace(char, escape)

        # Restore placeholders to their proper escaped form
        for char, escape in control_char_map.items():
            placeholder = f"__PLACEHOLDER_{ord(char)}__"
            unescaped = unescaped.replace(placeholder, escape)

        # Apply all cleanings together as a last resort
        unescaped = _remove_comments_comprehensive(unescaped)
        unescaped = re.sub(r",\s*([\]}])", r"\1", unescaped)

        parsed = json.loads(unescaped)
        if isinstance(parsed, (dict, list)):
            results.append((parsed, start_pos, end_pos))
    except (json.JSONDecodeError, Exception):
        # If all approaches fail, don't add anything to results
        pass


def _remove_comments_comprehensive(text):
    """
    Comprehensively removes both single-line and multi-line JavaScript style comments.
    Handles complex cases like nested comments and comments inside strings.

    :param text: The JSON text to clean
    :return: Text with all comments removed
    """
    # Process the string character by character to properly handle comments vs strings
    result = []
    i = 0
    in_string = False
    in_single_comment = False
    in_multi_comment = False
    escape_next = False

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        # Handle string literals
        if (
            char == '"'
            and not escape_next
            and not in_single_comment
            and not in_multi_comment
        ):
            in_string = not in_string
            result.append(char)

        # Handle escape character within strings
        elif char == "\\" and in_string and not escape_next:
            escape_next = True
            result.append(char)

        # Handle start of single-line comment
        elif (
            char == "/"
            and next_char == "/"
            and not in_string
            and not in_single_comment
            and not in_multi_comment
        ):
            in_single_comment = True
            i += 1  # Skip the next '/' character

        # Handle end of single-line comment
        elif char == "\n" and in_single_comment:
            in_single_comment = False
            result.append(char)  # Keep the newline

        # Handle start of multi-line comment
        elif (
            char == "/"
            and next_char == "*"
            and not in_string
            and not in_single_comment
            and not in_multi_comment
        ):
            in_multi_comment = True
            i += 1  # Skip the next '*' character

        # Handle end of multi-line comment
        elif char == "*" and next_char == "/" and in_multi_comment:
            in_multi_comment = False
            i += 1  # Skip the next '/' character

        # Add character to result if not in a comment
        elif not in_single_comment and not in_multi_comment:
            result.append(char)

        # Reset escape flag
        if escape_next:
            escape_next = False

        i += 1

    return "".join(result)


def _remove_overlapping_jsons(jsons_with_spans):
    """
    Removes overlapping JSON objects from the results.

    When JSONs overlap, we keep the one that appears first in the original text.

    :param jsons_with_spans: List of tuples (parsed_json, start_pos, end_pos)
    :return: List of non-overlapping JSON objects with their spans
    """
    if not jsons_with_spans:
        return []

    # Sort by start position to process in order of appearance
    sorted_jsons = sorted(jsons_with_spans, key=lambda x: x[1])

    non_overlapping = [sorted_jsons[0]]
    last_end = sorted_jsons[0][2]

    for json_item, start, end in sorted_jsons[1:]:
        # If this item starts after the last one ends (no overlap)
        if start >= last_end:
            non_overlapping.append((json_item, start, end))
            last_end = end

    return non_overlapping
