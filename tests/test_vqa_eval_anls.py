import unittest

from vlmeval.dataset.utils.vqa_eval import anls_compute, hit_calculate


class TestAnlsCompute(unittest.TestCase):

    def test_whitespace_padding_is_not_credit(self):
        # The distance is computed on normalized strings, so the length must be too.
        self.assertEqual(anls_compute('cat', ' ' * 1000), 1.0)
        self.assertEqual(anls_compute('cat', '\n' * 50), 1.0)

    def test_padding_does_not_change_the_distance(self):
        expected = anls_compute('invoice 2019', 'dog')
        self.assertEqual(anls_compute('invoice 2019', 'dog' + ' ' * 200), expected)
        self.assertGreater(expected, 0.5)

    def test_padded_correct_answer_still_matches(self):
        self.assertEqual(anls_compute('cat', '  cat  \n'), 0.0)
        self.assertEqual(anls_compute('a b', 'a   b'), 0.0)

    def test_unpadded_scores_are_unchanged(self):
        self.assertEqual(anls_compute('cat', 'cat'), 0.0)
        self.assertEqual(anls_compute('cat', 'CAT'), 0.0)
        self.assertAlmostEqual(anls_compute('invoice 2019', 'invoice 2018'), 1 / 12)

    def test_padded_wrong_answer_scores_zero_on_docvqa(self):
        result = [{'match': [anls_compute('cat', ' ' * 1000)]}]
        self.assertEqual(hit_calculate(result, 'DocVQA'), [0.0])


if __name__ == '__main__':
    unittest.main()
