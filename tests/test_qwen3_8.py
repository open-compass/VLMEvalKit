import unittest
from unittest.mock import MagicMock, patch

from vlmeval.config import qwen3_8_series, supported_VLM
from vlmeval.vlm import Qwen3VLChat
from vlmeval.vlm.qwen3_vl.model import is_moe_model


class TestQwen3_8(unittest.TestCase):

    def test_series_registration(self):
        self.assertGreater(len(qwen3_8_series), 0)
        expected_models = [
            "Qwen3.8-27B",
            "Qwen3.8-27B-Thinking",
            "Qwen3.8-27B-Instruct",
            "Qwen3.8-27B-FP8",
            "Qwen3.8-2.4T-A95B",
            "Qwen3.8-Flash-Next",
            "Qwen3.8-Flash-Next-FP8",
            "Qwen3.8-27B_api",
            "Qwen3.8-27B_ThinkMode_api",
            "Qwen3.8-27B_InstructMode_api",
            "Qwen3.8-2.4T-A95B_api",
            "Qwen3.8-Flash-Next_api",
            "Qwen3.8-Max",
            "Qwen3.8-27B-API",
            "Qwen3.8-Flash-Next-API",
        ]
        for name in expected_models:
            self.assertIn(name, qwen3_8_series)
            self.assertIn(name, supported_VLM)

    def test_moe_detection(self):
        self.assertTrue(is_moe_model("Qwen/Qwen3.8-2.4T-A95B"))
        self.assertTrue(is_moe_model("Qwen/Qwen3.8-Flash-Next"))
        self.assertTrue(is_moe_model("Qwen/Qwen3.8-Flash-Next-FP8"))
        self.assertFalse(is_moe_model("Qwen/Qwen3.8-27B"))
        self.assertFalse(is_moe_model("Qwen/Qwen3.8-27B-FP8"))

    @patch('transformers.AutoModelForImageTextToText.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('vlmeval.vlm.qwen3_vl.model.get_gpu_memory', return_value=[80000])
    @patch('vlmeval.vlm.qwen3_vl.model.torch.cuda.device_count', return_value=1)
    def test_chat_template_kwargs_and_prompts(self, mock_gpu_count, mock_gpu_mem, mock_proc, mock_model):
        mock_processor_instance = MagicMock()
        mock_proc.return_value = mock_processor_instance

        # Test initialization with thinking disabled
        vlm_model = Qwen3VLChat(
            model_path="Qwen/Qwen3.8-27B",
            enable_thinking=False,
            chat_template_kwargs={"custom_flag": True},
            use_vllm=False
        )
        self.assertEqual(vlm_model.chat_template_kwargs, {"custom_flag": True, "enable_thinking": False})

        # Test prompt building for MCQ
        line = {
            'question': 'What color is the sky?',
            'A': 'Blue',
            'B': 'Green',
            'image': 'test.jpg'
        }
        vlm_model.set_dump_image(lambda item: 'test.jpg')
        prompt_msgs = vlm_model.build_prompt(line, dataset='MMMU_DEV_VAL')
        self.assertEqual(len(prompt_msgs), 2)
        self.assertEqual(prompt_msgs[0]['type'], 'image')
        self.assertEqual(prompt_msgs[0]['value'], 'test.jpg')
        self.assertEqual(prompt_msgs[1]['type'], 'text')
        self.assertIn('What color is the sky?', prompt_msgs[1]['value'])


if __name__ == '__main__':
    unittest.main()
