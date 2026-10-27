import unittest

from vlmeval.dataset.OmniDocBench.omnidocbench import OmniDocBench, parse_omnidocbench_annotation


class OmniDocBenchTests(unittest.TestCase):

    def test_v1_0_json_annotation(self):
        value = '{"page_info": {"image_path": "page.png"}, "layout_dets": [], "extra": {}}'
        parsed = parse_omnidocbench_annotation(value)
        self.assertEqual(parsed['page_info']['image_path'], 'page.png')

    def test_v1_5_python_literal_annotation(self):
        value = "{'page_info': {'image_path': 'page.png'}, 'layout_dets': [], 'extra': {}}"
        parsed = parse_omnidocbench_annotation(value)
        self.assertEqual(parsed['page_info']['image_path'], 'page.png')

    def test_non_dictionary_annotation_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_omnidocbench_annotation("['not', 'an', 'annotation']")

    def test_literal_fallback_cannot_execute_code(self):
        with self.assertRaises(ValueError):
            parse_omnidocbench_annotation("__import__('os').system('false')")

    def test_default_dataset_is_pinned_v1_5(self):
        self.assertIn(
            '9702d4ba9a0d30dc5e76789707650c9c54cb0b3b',
            OmniDocBench.DATASET_URL['OmniDocBench'],
        )
        self.assertEqual(
            OmniDocBench.DATASET_MD5['OmniDocBench'],
            '995f1af5b4e24ad0a6417cbff708b3fc',
        )
        self.assertIn('OmniDocBench_v1_0', OmniDocBench.supported_datasets())


if __name__ == '__main__':
    unittest.main()
