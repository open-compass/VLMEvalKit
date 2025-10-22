import pytest

from math_verify import StringExtractionConfig, parse


@pytest.mark.parametrize(
    "pred,expected,config",
    [
        # Test basic string extraction
        ("The answer is A.", ["A", "A"], StringExtractionConfig(lowercase=False)),
        # Test lowercase
        ("The answer is A.", ["a", "A"], StringExtractionConfig(lowercase=True)),
        # Test with different string options
        ("Final answer is B", ["b", "B"], StringExtractionConfig()),
        # Test no match
        ("No valid answer here", [], StringExtractionConfig()),
        # Test start regex
        ("A. Because B is not valid", ["a", "A"], StringExtractionConfig()),
        # Test space truncate
        # Test different strings
        ("The answer is U.", ["u", "U"], StringExtractionConfig(strings=("U",))),
        # Test any plain string
        ("Because B is valid", ["b", "B"], StringExtractionConfig()),
    ],
)
def test_string_extraction(pred, expected, config):
    assert parse(pred, [config]) == expected
