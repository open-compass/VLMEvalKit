import sys
import unittest
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parents[1] / "vlmeval" / "dataset" / "utils"
sys.path.insert(0, str(UTILS_DIR))

from ccocr_evaluator.doc_parsing_evaluator import ParsingEvaluator  # noqa: E402
from ccocr_evaluator.doc_parsing_evaluator import extract_table_html  # noqa: E402


def evaluate_single(prediction, ground_truth):
    evaluator = ParsingEvaluator("doc_parsing")
    return evaluator.eval_table({"sample": prediction}, {"sample": ground_truth})


class CCOCRTableEvaluatorTest(unittest.TestCase):
    def test_table_attributes_are_preserved_and_th_is_not_td(self):
        prediction = """
        ```HTML
        <table border="1" style="width: 100%">
          <tr><th colspan="2">A B</th></tr>
          <tr><td>1</td><td>2</td></tr>
        </table>
        ```
        """
        ground_truth = """
        <table>
          <tr><td colspan="2">AB</td></tr>
          <tr><td>1</td><td>2</td></tr>
        </table>
        """

        self.assertLess(evaluate_single(prediction, ground_truth), 1.0)

    def test_rowspan_and_text_whitespace_are_normalized(self):
        prediction = """
        <table class="result">
          <tr><td rowspan="2">foo bar</td><td>A</td></tr>
          <tr><td>B</td></tr>
        </table>
        """
        ground_truth = (
            '<table><tr><td rowspan="2">foobar</td><td>A</td></tr>'
            '<tr><td>B</td></tr></table>'
        )

        self.assertEqual(evaluate_single(prediction, ground_truth), 1.0)

    def test_non_html_fence_does_not_hide_table(self):
        response = """
        Here is a note:
        ```text
        This is not HTML.
        ```
        <table><tr><td>value</td></tr></table>
        """

        self.assertEqual(
            extract_table_html(response),
            "<table><tr><td>value</td></tr></table>",
        )

    def test_table_is_extracted_from_html_fence(self):
        response = """
        Here is the requested table:
        ```html
        <table><tr><td>value</td></tr></table>
        ```
        """

        self.assertEqual(
            extract_table_html(response),
            "<table><tr><td>value</td></tr></table>",
        )

    def test_ground_truth_fullwidth_characters_are_normalized(self):
        prediction = '<table><tr><td>AB,1</td></tr></table>'
        ground_truth = '<table><tr><td>ＡＢ，１</td></tr></table>'

        self.assertEqual(evaluate_single(prediction, ground_truth), 1.0)

    def test_missing_table_scores_zero(self):
        self.assertEqual(
            evaluate_single(
                "No table was generated.",
                "<table><tr><td>x</td></tr></table>",
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
