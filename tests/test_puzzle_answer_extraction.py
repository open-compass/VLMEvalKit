import importlib.util
import sys
import types
import unittest
from unittest import mock

import pandas as pd


def _load_util(name):
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    smp = types.ModuleType('vlmeval.smp')
    smp.__path__ = ['vlmeval/smp']
    smp.load = lambda data: data
    smp_file = types.ModuleType('vlmeval.smp.file')
    smp_file.load = lambda data: data

    modules = {
        'vlmeval': vlmeval,
        'vlmeval.smp': smp,
        'vlmeval.smp.file': smp_file,
    }
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            f'vlmeval.dataset.utils.{name}',
            f'vlmeval/dataset/utils/{name}.py',
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


class TestPuzzleAnswerExtraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modules = [_load_util('puzzlevqa'), _load_util('visualpuzzles')]

    def test_reads_plain_and_parenthesized_letters(self):
        # The prompt lists options as "(A) ...", so models often echo "(B)".
        for module in self.modules:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.extract_answer('Reasoning.\nAnswer: B'), 'B')
                self.assertEqual(module.extract_answer('Reasoning.\nAnswer: (B)'), 'B')
                self.assertEqual(module.extract_answer('Answer: (A)\nAnswer: (C)'), 'C')

    def test_does_not_read_a_word_as_a_letter(self):
        for module in self.modules:
            with self.subTest(module=module.__name__):
                self.assertEqual(module.extract_answer('Answer: Because of symmetry'), 'Z')

    def test_puzzlevqa_accuracy_counts_parenthesized_answers(self):
        module = self.modules[0]
        # PuzzleVQA_acc reports every category, so give each one a row.
        categories = [
            'grid_number', 'size_grid', 'rectangle_height_color', 'shape_size_hexagon',
            'rectangle_height_number', 'shape_reflect', 'circle_size_number', 'polygon_sides_color',
            'color_overlap_squares', 'color_size_circle', 'venn', 'size_cycle', 'polygon_sides_number',
            'shape_size_grid', 'color_hexagon', 'grid_number_color', 'color_grid', 'shape_morph',
            'triangle', 'color_number_hexagon',
        ]
        data = pd.DataFrame({
            'category': categories,
            'options': ["['1', '2', '3', '4']"] * len(categories),
            'answer': ['2'] * len(categories),
            'prediction': ['So it is 2.\nAnswer: (B)'] * len(categories),
        })

        res = module.PuzzleVQA_acc(data)

        overall = res['category'].index('overall')
        self.assertEqual(res['tot'][overall], len(categories))
        self.assertEqual(res['hit'][overall], len(categories))


if __name__ == '__main__':
    unittest.main()
