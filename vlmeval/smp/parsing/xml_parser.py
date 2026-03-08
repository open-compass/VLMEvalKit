import re
import xml.etree.ElementTree as ET
from typing import Dict


def parse_xml(xml_str: str) -> Dict:
    """
    Parses an XML document from a string that may contain extra text and converts it to JSON format.

    This function attempts three approaches to extract XML:

    1. Directly parsing the entire string.
    2. Extracting XML enclosed within triple backticks (```xml ... ```).
    3. Extracting XML with properly balanced tags.

    :param xml_str: The input string potentially containing an XML document.
    :type xml_str: str
    :return: The parsed XML converted to a JSON-compatible dictionary.
    :rtype: dict
    """
    if xml_str is None or not isinstance(xml_str, str):
        raise TypeError("Input must be a non-empty string.")
    if not xml_str:
        raise ValueError("Input string is empty.")

    # Store all successfully parsed XML objects
    parsed_xmls = []

    # Attempt 1: Try to parse the entire string as XML
    try:
        root = ET.fromstring(xml_str)
        parsed_xmls.append((root, xml_str))
    except ET.ParseError:
        pass

    # Attempt 2: Look for XML blocks delimited by ```xml and ```
    code_block_matches = re.finditer(r"```(?:xml)?\s*([\s\S]*?)\s*```", xml_str)
    for match in code_block_matches:
        xml_block = match.group(1)
        try:
            root = ET.fromstring(xml_block)
            parsed_xmls.append((root, xml_block))
        except ET.ParseError:
            pass

    # Attempt 3: Extract XML with balanced tags
    xml_patterns = [
        r"<\?xml.*?\?>.*?<([a-zA-Z0-9_:]+)(?:\s+[^>]*)?>(.*?)</\1>",  # XML with declaration
        r"<([a-zA-Z0-9_:]+)(?:\s+[^>]*)?>(.*?)</\1>",  # XML without declaration
    ]

    for pattern in xml_patterns:
        matches = re.finditer(pattern, xml_str, re.DOTALL)
        for match in matches:
            try:
                full_match = match.group(0)
                root = ET.fromstring(full_match)
                parsed_xmls.append((root, full_match))
            except ET.ParseError:
                pass

    if parsed_xmls:
        # Sort by complexity: First by length of XML string, then by depth
        sorted_xmls = sorted(
            parsed_xmls,
            key=lambda x: (len(x[1]), _xml_structure_depth(x[0])),
            reverse=True,
        )
        # Convert the most complex XML to JSON format
        return _xml_to_json(sorted_xmls[0][0])
    else:
        raise ValueError("Failed to parse XML from the input string.")


def _xml_structure_depth(element: ET.Element) -> int:
    """
    Calculate the depth of an XML element structure.

    :param element: The XML element
    :return: The maximum nesting depth
    """
    if len(element) == 0:
        return 1
    return 1 + max(_xml_structure_depth(child) for child in element)


def _xml_to_json(element: ET.Element) -> Dict:
    """
    Convert an XML element to a JSON-compatible dictionary.

    :param element: The XML element to convert
    :return: A dictionary representation of the XML
    """
    result = {}

    # Add attributes as "@attribute_name": value
    if element.attrib:
        result.update({f"@{k}": v for k, v in element.attrib.items()})

    # Process child elements
    child_elements = {}
    for child in element:
        child_json = _xml_to_json(child)
        if child.tag in child_elements:
            # If this tag already exists, convert to a list or append to existing list
            if isinstance(child_elements[child.tag], list):
                child_elements[child.tag].append(child_json)
            else:
                child_elements[child.tag] = [child_elements[child.tag], child_json]
        else:
            child_elements[child.tag] = child_json

    # Add child elements to result
    result.update(child_elements)

    # Add text content if present and not whitespace-only
    if element.text and element.text.strip():
        # If the element has no attributes and no children, just return the text
        if not result:
            return element.text.strip()
        else:
            result["#text"] = element.text.strip()

    return result
