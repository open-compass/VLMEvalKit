import importlib.util
import unittest

import pandas as pd


def _load_dataset_subset():
    spec = importlib.util.spec_from_file_location(
        'dataset_subset',
        'vlmeval/utils/dataset_subset.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeVideoDataset:

    def __init__(self):
        self.data = pd.DataFrame({
            'index': [10, 20, 30],
            'video': ['b.mp4', 'a.mp4', 'b.mp4'],
        })
        self.videos = ['a.mp4', 'b.mp4']


class TestDatasetSubset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.subset = _load_dataset_subset()

    def test_parse_data_indices_normalizes_integer_and_string_indices(self):
        parsed = self.subset.parse_data_indices(
            '{"MMBench_DEV_EN": [1203, "sample-2"]}'
        )
        self.assertEqual(parsed, {'MMBench_DEV_EN': ['1203', 'sample-2']})

    def test_parse_data_indices_rejects_invalid_or_duplicate_values(self):
        invalid_values = [
            '[]',
            '{"MME": []}',
            '{"MME": [true]}',
            '{"MME": [1, "1"]}',
        ]
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.subset.parse_data_indices(value)

    def test_validate_data_indices_rejects_unselected_dataset(self):
        with self.assertRaisesRegex(ValueError, 'not selected'):
            self.subset.validate_data_indices({'MME': ['1']}, ['MMBench_DEV_EN'])

    def test_filter_frame_matches_index_types_and_preserves_dataset_order(self):
        frame = pd.DataFrame({'index': [3, 1, 2], 'question': ['c', 'a', 'b']})
        filtered = self.subset.filter_frame_by_indices(frame, ['2', '3'])
        self.assertEqual(filtered['index'].tolist(), [3, 2])
        self.assertEqual(filtered['question'].tolist(), ['c', 'b'])

    def test_filter_frame_reports_missing_indices(self):
        frame = pd.DataFrame({'index': [1, 2]})
        with self.assertRaisesRegex(ValueError, '3'):
            self.subset.filter_frame_by_indices(frame, ['2', '3'])

    def test_filter_frame_can_keep_available_incomplete_predictions(self):
        frame = pd.DataFrame({'index': [1, 4], 'prediction': ['a', 'd']})
        filtered = self.subset.filter_frame_by_indices(frame, ['1', '2'], require_all=False)
        self.assertEqual(filtered['index'].tolist(), [1])

    def test_subset_dataset_refreshes_video_list(self):
        dataset = FakeVideoDataset()
        original_size, selected_size = self.subset.subset_dataset(dataset, ['10', '30'])
        self.assertEqual((original_size, selected_size), (3, 2))
        self.assertEqual(dataset.data['index'].tolist(), [10, 30])
        self.assertEqual(dataset.videos, ['b.mp4'])


if __name__ == '__main__':
    unittest.main()
