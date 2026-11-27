import unittest

from vlmeval.dataset.utils.vmcbench import parse_multi_choice_response

CHOICES = ['A', 'B', 'C', 'D']
INDEX2ANS = {'A': 'Sociological', 'B': 'Theme', 'C': 'Plot', 'D': 'Setting'}


def parse(response):
    return parse_multi_choice_response(response, CHOICES, INDEX2ANS)


class TestVmcbenchParse(unittest.TestCase):

    def test_plain_boxed_answer(self):
        self.assertEqual(parse('\\boxed{B}'), 'B')
        self.assertEqual(parse('The answer is \\boxed{C}.'), 'C')
        self.assertEqual(parse('Answer: \\boxed{D}'), 'D')

    def test_boxed_wins_over_letters_in_reasoning(self):
        # Reasoning mentions other option letters; the explicit final answer is boxed.
        response = (
            'Plot is the sequence of events and setting is the time and place, '
            'but the question asks for the central message. Therefore, the answer is B. Theme. '
            'Answer: \\boxed{B}'
        )
        self.assertEqual(parse(response), 'B')
        response = (
            'A. Sociological looks plausible at first, but the play is really about its central message. '
            'Answer: \\boxed{B}'
        )
        self.assertEqual(parse(response), 'B')

    def test_boxed_letter_outside_choices_falls_back(self):
        # E is not among all_choices, so the original heuristics still apply.
        self.assertEqual(parse('I would say (B) but formally \\boxed{E}'), 'B')

    def test_unboxed_responses_unchanged(self):
        self.assertEqual(parse('The answer is (C)'), 'C')
        self.assertEqual(parse('Answer: B. Theme'), 'B')
        self.assertEqual(parse('The plot of the story is about a plot'), 'C')


if __name__ == '__main__':
    unittest.main()
