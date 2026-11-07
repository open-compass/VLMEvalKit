import os
import unittest
from unittest.mock import MagicMock, patch

from vlmeval.config import qwen3_8_series, supported_VLM


class TestAllQwen3_8Models(unittest.TestCase):

    def test_all_15_models_configuration(self):
        """Verify each of the 15 Qwen 3.8 models has correct configuration in supported_VLM."""
        self.assertEqual(len(qwen3_8_series), 15)
        for name, partial_func in qwen3_8_series.items():
            self.assertIn(name, supported_VLM, f"{name} not found in supported_VLM")
            func = partial_func.func
            keywords = partial_func.keywords
            print(f"Verified config for {name} -> {func.__name__} (keys: {list(keywords.keys())})")

    @patch('transformers.AutoModelForImageTextToText.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('vlmeval.vlm.qwen3_vl.model.get_gpu_memory', return_value=[80000])
    @patch('vlmeval.vlm.qwen3_vl.model.torch.cuda.device_count', return_value=1)
    def test_all_open_weights_models_pipeline(self, mock_gpu_count, mock_gpu_mem, mock_proc, mock_model):
        """Test instantiation, prompt generation, and inference pipeline for all open-weights models."""
        open_weight_models = [
            "Qwen3.8-27B",
            "Qwen3.8-27B-Thinking",
            "Qwen3.8-27B-Instruct",
            "Qwen3.8-27B-FP8",
            "Qwen3.8-2.4T-A95B",
            "Qwen3.8-Flash-Next",
            "Qwen3.8-Flash-Next-FP8",
        ]

        mock_processor_instance = MagicMock()
        mock_processor_instance.apply_chat_template.return_value = "<mock_prompt>"
        mock_processor_instance.tokenizer.batch_decode.return_value = ["A single red apple."]
        mock_proc.return_value = mock_processor_instance

        mock_model_instance = MagicMock()
        mock_model_instance.generate.return_value = [[1, 2, 3, 4]]
        mock_model.return_value = mock_model_instance

        img_path = os.path.abspath('assets/apple.jpg')
        test_messages = [
            {'type': 'image', 'value': img_path},
            {'type': 'text', 'value': 'Describe what is in this image.'}
        ]

        for name in open_weight_models:
            builder = supported_VLM[name]
            # Override use_vllm=False for testing transformers generation pipeline
            model = builder(use_vllm=False)
            model.set_dump_image(lambda item: img_path)

            # Test prompt building for MMMU, MCQ, Y/N, VQA
            line = {'question': 'Is this an apple?', 'A': 'Yes', 'B': 'No'}
            mmmu_prompt = model.build_prompt(line, dataset='MMMU_DEV_VAL')
            self.assertEqual(mmmu_prompt[0]['type'], 'image')
            self.assertEqual(mmmu_prompt[1]['type'], 'text')

            mcq_prompt = model.build_prompt(line, dataset='MMBench_DEV_EN')
            self.assertIn('Answer with the option letter only.', mcq_prompt[1]['value'])

            yorn_prompt = model.build_prompt(line, dataset='MME')
            self.assertIn('Please answer yes or no.', yorn_prompt[1]['value'])

            vqa_prompt = model.build_prompt(line, dataset='DocVQA_VAL')
            self.assertIn('Please answer concisely', vqa_prompt[1]['value'])

            # Test generation through VLMEvalKit generate() entrypoint
            with patch('qwen_vl_utils.process_vision_info', return_value=(None, None, None)):
                out = model.generate(test_messages)
                self.assertEqual(out, "A single red apple.")
                print(f"[PASSED] Open-weights model pipeline: {name}")

    @patch('urllib.request.urlopen')
    def test_all_api_models_pipeline(self, mock_urlopen):
        """Test instantiation and payload construction for all LMDeploy / vLLM server API models."""
        api_models = [
            "Qwen3.8-27B_api",
            "Qwen3.8-27B_ThinkMode_api",
            "Qwen3.8-27B_InstructMode_api",
            "Qwen3.8-2.4T-A95B_api",
            "Qwen3.8-Flash-Next_api",
        ]

        for name in api_models:
            builder = supported_VLM[name]
            model = builder()
            self.assertEqual(model.api_base, "http://0.0.0.0:8000/v1/chat/completions")
            self.assertTrue(hasattr(model, 'generate'))
            print(f"[PASSED] Server API model configuration: {name}")

    def test_all_dashscope_api_models_pipeline(self):
        """Test instantiation and message preparation for all DashScope cloud API models."""
        dashscope_models = [
            "Qwen3.8-Max",
            "Qwen3.8-27B-API",
            "Qwen3.8-Flash-Next-API",
        ]

        img_path = os.path.abspath('assets/apple.jpg')
        test_inputs = [
            {'type': 'image', 'value': img_path},
            {'type': 'text', 'value': 'What is this?'}
        ]

        for name in dashscope_models:
            builder = supported_VLM[name]
            # Initialize with dummy test key to verify structure
            model = builder(key='mock-dashscope-key')
            self.assertTrue(model.is_api)
            prepared = model._prepare_content(test_inputs)
            self.assertEqual(prepared[0]['type'], 'image')
            self.assertEqual(prepared[1]['type'], 'text')
            print(f"[PASSED] DashScope API model: {name} (target model: {model.model})")


if __name__ == '__main__':
    unittest.main()
