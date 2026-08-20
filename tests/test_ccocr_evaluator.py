import unittest

from vlmeval.dataset.utils.ccocr_evaluator.doc_parsing_evaluator import (
    CustomConfig,
    ParsingEvaluator,
    TableTree,
)


class TestCCOCREvaluator(unittest.TestCase):
    def test_teds_cell_distance_supports_long_content(self):
        content_length = 2140
        predicted = TableTree("td", 1, 1, list("a" * content_length))
        ground_truth = TableTree("td", 1, 1, list("a" * (content_length - 1) + "b"))

        score = CustomConfig().rename(predicted, ground_truth)

        self.assertAlmostEqual(score, 1 / content_length)

    def test_doc_distance_supports_long_content(self):
        content_length = 2140
        predicted = "a" * content_length
        ground_truth = "a" * (content_length - 1) + "b"

        score = ParsingEvaluator("doc_parsing").eval_doc({"sample": predicted}, {"sample": ground_truth})

        self.assertAlmostEqual(score, 1 - 1 / content_length)


if __name__ == "__main__":
    unittest.main()
