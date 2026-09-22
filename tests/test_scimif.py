import json
import unittest

from PIL import Image

from vlmeval.dataset.scimif import SciMIF
from vlmeval.dataset.utils.scimif_eval import evaluate_record, summarize_results


class TestSciMIF(unittest.TestCase):

    def test_convert_text_only_record(self):
        converted = SciMIF._convert_record(
            {
                'sample_id': 'physics_0',
                'subject': 'physics',
                'edit_question': 'Return a velocity in m/s.',
                'instruction_list': [],
            }, 0)

        self.assertIsNone(converted['image'])
        self.assertEqual(json.loads(converted['image_path']), [])
        self.assertEqual(converted['question'], 'Return a velocity in m/s.')

    def test_convert_image_record(self):
        converted = SciMIF._convert_record(
            {
                'sample_id': 'geography_0',
                'subject': 'geography',
                'edit_question': 'Describe the image.',
                'image': [Image.new('RGB', (2, 2), color='white')],
                'image_path': ['images/geography/example.png'],
                'instruction_list': [],
            }, 0)

        self.assertGreater(len(json.loads(converted['image'])[0]), 64)
        self.assertEqual(json.loads(converted['image_path']), ['geography/example.png'])

    def test_instruction_evaluation_and_summary(self):
        item = {
            'index':
            '0',
            'subject':
            'physics',
            'edit_question':
            'Give the final velocity in m/s.',
            'prediction':
            'The final answer is 3 m/s.',
            'instruction_list': [
                {
                    'instruction_name': 'physics_unit_consistency',
                    'source': 'core_task',
                    'required_parameters': 'm/s',
                },
            ],
        }
        result = evaluate_record(item)
        summary = summarize_results([{**item, **result}])

        self.assertEqual(result['instruction_score'], 1.0)
        self.assertEqual(result['strict_score'], 1.0)
        self.assertEqual(summary[0]['instruction_accuracy'], 1.0)
        self.assertEqual(result['instruction_results'][0]['source'], 'core_task')


if __name__ == '__main__':
    unittest.main()
