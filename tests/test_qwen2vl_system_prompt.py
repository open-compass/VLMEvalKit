import copy
import sys
import types
import unittest
from unittest import mock

from vlmeval.vlm.qwen2_vl.model import Qwen2VLChat


class TestQwen2VLSystemPrompt(unittest.TestCase):

    @staticmethod
    def _message(system_prompt='dataset prompt'):
        return [
            {'type': 'text', 'value': system_prompt, 'role': 'system'},
            {'type': 'image', 'value': 'image.jpg'},
            {'type': 'text', 'value': 'question', 'role': 'user'},
        ]

    @staticmethod
    def _model(system_prompt=None):
        model = Qwen2VLChat.__new__(Qwen2VLChat)
        model.system_prompt = system_prompt
        return model

    def test_split_system_prompt_without_mutating_input_or_model(self):
        model = self._model('config prompt')
        message = self._message()
        original = copy.deepcopy(message)

        user_message, system_prompt = model._split_system_prompt(message)

        self.assertEqual(system_prompt, 'config prompt\ndataset prompt')
        self.assertEqual(
            user_message,
            [
                {'type': 'image', 'value': 'image.jpg'},
                {'type': 'text', 'value': 'question'},
            ],
        )
        self.assertEqual(message, original)
        self.assertEqual(model.system_prompt, 'config prompt')

    def test_split_system_prompt_deduplicates_config_prompt(self):
        model = self._model('shared prompt')

        user_message, system_prompt = model._split_system_prompt(
            self._message('shared prompt')
        )

        self.assertEqual(system_prompt, 'shared prompt')
        self.assertEqual(len(user_message), 2)

    def test_split_system_prompt_rejects_non_text_system_message(self):
        model = self._model()

        with self.assertRaisesRegex(AssertionError, 'System messages must be text'):
            model._split_system_prompt(
                [{'type': 'image', 'value': 'image.jpg', 'role': 'system'}]
            )

    def test_build_chat_messages_uses_separate_system_channel(self):
        model = self._model('config prompt')
        message = [{'type': 'text', 'value': 'question'}]
        expected = [
            {'role': 'system', 'content': 'dataset prompt'},
            {'role': 'user', 'content': [{'type': 'text', 'text': 'question'}]},
        ]

        self.assertEqual(
            model._build_chat_messages(
                message, dataset='E3VQA', system_prompt='dataset prompt'
            ),
            expected,
        )
        self.assertEqual(
            model._build_chat_messages(
                message,
                dataset='E3VQA',
                system_prompt='dataset prompt',
                use_vllm=True,
            ),
            expected,
        )

    def test_lmdeploy_messages_use_separate_system_channel(self):
        model = self._model('config prompt')
        lmdeploy = types.ModuleType('lmdeploy')
        lmdeploy_vl = types.ModuleType('lmdeploy.vl')
        lmdeploy_constants = types.ModuleType('lmdeploy.vl.constants')
        lmdeploy_constants.IMAGE_TOKEN = '<IMAGE_TOKEN>'

        with mock.patch.dict(
            sys.modules,
            {
                'lmdeploy': lmdeploy,
                'lmdeploy.vl': lmdeploy_vl,
                'lmdeploy.vl.constants': lmdeploy_constants,
            },
        ):
            messages = model.message_to_lmdeploy(
                [{'type': 'text', 'value': 'question'}],
                system_prompt='dataset prompt',
            )

        self.assertEqual(
            messages,
            [[
                {'role': 'system', 'content': 'dataset prompt'},
                {
                    'role': 'user',
                    'content': [{'type': 'text', 'text': 'question'}],
                },
            ]],
        )

    def test_generate_inner_passes_per_request_prompt_to_all_backends(self):
        expected_message = [
            {'type': 'image', 'value': 'image.jpg'},
            {'type': 'text', 'value': 'question'},
        ]
        backends = [
            ('transformers', False, False, 'generate_inner_transformers'),
            ('lmdeploy', False, True, 'generate_inner_lmdeploy'),
            ('vllm', True, False, 'generate_inner_vllm'),
        ]

        for name, use_vllm, use_lmdeploy, method_name in backends:
            with self.subTest(backend=name):
                model = self._model('config prompt')
                model.use_vllm = use_vllm
                model.use_lmdeploy = use_lmdeploy
                model.generate_inner_transformers = mock.Mock(return_value='transformers')
                model.generate_inner_lmdeploy = mock.Mock(return_value='lmdeploy')
                model.generate_inner_vllm = mock.Mock(return_value='vllm')

                result = model.generate_inner(self._message(), dataset='E3VQA')

                self.assertEqual(result, name)
                getattr(model, method_name).assert_called_once_with(
                    expected_message,
                    dataset='E3VQA',
                    system_prompt='config prompt\ndataset prompt',
                )


if __name__ == '__main__':
    unittest.main()
