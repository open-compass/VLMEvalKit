import contextlib
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import pandas as _pandas
from PIL import Image


class FakeBaseModel:
    pass


class FakeInputIds:
    shape = (1, 2)

    def to(self, *args, **kwargs):
        return self


class FakeInputs(dict):

    def __init__(self):
        super().__init__(input_ids=FakeInputIds())

    def to(self, *args, **kwargs):
        return self


class FakeModel:
    device = 'cuda'

    def __init__(self, model_name=None):
        if model_name is not None:
            self.language_model = types.SimpleNamespace(name_or_path=model_name)

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


def _base_modules():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    vlm = types.ModuleType('vlmeval.vlm')
    vlm.__path__ = ['vlmeval/vlm']
    base = types.ModuleType('vlmeval.vlm.base')
    base.BaseModel = FakeBaseModel

    dataset = types.ModuleType('vlmeval.dataset')
    dataset.DATASET_TYPE = lambda name: 'VQA'
    smp = types.ModuleType('vlmeval.smp')
    smp.cn_string = lambda value: False
    smp.encode_image_file_to_base64 = lambda path: 'encoded-image'
    smp.get_cache_path = lambda *args, **kwargs: '/tmp/model'
    smp.listinstr = lambda needles, value: any(needle in value for needle in needles)

    torch = types.ModuleType('torch')
    torch.bfloat16 = 'bfloat16'
    torch.float16 = 'float16'
    torch.inference_mode = contextlib.nullcontext
    torch.cuda = types.SimpleNamespace(device_count=lambda: 1, empty_cache=lambda: None)

    huggingface_hub = types.ModuleType('huggingface_hub')
    huggingface_hub.snapshot_download = mock.Mock()
    return {
        'vlmeval': vlmeval,
        'vlmeval.vlm': vlm,
        'vlmeval.vlm.base': base,
        'vlmeval.dataset': dataset,
        'vlmeval.smp': smp,
        'torch': torch,
        'huggingface_hub': huggingface_hub,
        'pandas': _pandas,
    }


def _load_vlm_module(module_name):
    modules = _base_modules()
    full_name = f'vlmeval.vlm.{module_name}'
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(full_name, f'vlmeval/vlm/{module_name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        sys.modules.pop(full_name, None)
        return module


class FakeGemmaProcessor:

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        if kwargs.get('tokenize') is False:
            return 'gemma-prompt'
        return FakeInputs()

    def decode(self, *args, **kwargs):
        return 'gemma-answer'


class FakeTokenizer:
    eos_token_id = 2

    def convert_tokens_to_ids(self, token):
        return 3


class FakeMantisProcessor:

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def __call__(self, prompt, images, **kwargs):
        self.prompt = prompt
        self.images = images
        return FakeInputs()

    def decode(self, *args, **kwargs):
        return 'mantis-answer<|eot_id|>'


class FakeMantisIdeficsProcessor(FakeMantisProcessor):

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'mantis-idefics-prompt'


class FakeConversation:
    roles = ('USER', 'ASSISTANT')
    last_messages = None

    def __init__(self):
        self.messages = []

    def copy(self):
        return FakeConversation()

    def append_message(self, role, content):
        self.messages.append([role, content])

    def get_prompt(self):
        FakeConversation.last_messages = [list(item) for item in self.messages]
        return 'mantis-prompt'


class TextChunk:

    def __init__(self, text):
        self.text = text


class ImageURLChunk:

    def __init__(self, image_url):
        self.image_url = image_url


class BaseMessage:

    def __init__(self, content):
        self.content = content


class UserMessage(BaseMessage):
    role = 'user'


class AssistantMessage(BaseMessage):
    role = 'assistant'


class SystemMessage(BaseMessage):
    role = 'system'


class ChatCompletionRequest:

    def __init__(self, messages):
        self.messages = messages


class FakeMistralTokenizer:
    instruct_tokenizer = types.SimpleNamespace(
        tokenizer=types.SimpleNamespace(eos_id=2)
    )

    def encode_chat_completion(self, request):
        self.request = request
        return types.SimpleNamespace(images=['image'], tokens=[1, 2])

    def decode(self, tokens):
        return 'pixtral-answer'


class SamplingParams:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeVLLM:

    def generate(self, request, sampling_params):
        self.request = request
        self.sampling_params = sampling_params
        return [types.SimpleNamespace(
            outputs=[types.SimpleNamespace(text='gemma-vllm-answer')]
        )]


def _mistral_modules(generate):
    common = types.ModuleType('mistral_common')
    protocol = types.ModuleType('mistral_common.protocol')
    instruct = types.ModuleType('mistral_common.protocol.instruct')
    messages = types.ModuleType('mistral_common.protocol.instruct.messages')
    messages.AssistantMessage = AssistantMessage
    messages.ImageURLChunk = ImageURLChunk
    messages.SystemMessage = SystemMessage
    messages.TextChunk = TextChunk
    messages.UserMessage = UserMessage
    request = types.ModuleType('mistral_common.protocol.instruct.request')
    request.ChatCompletionRequest = ChatCompletionRequest
    inference = types.ModuleType('mistral_inference')
    generate_module = types.ModuleType('mistral_inference.generate')
    generate_module.generate = generate
    return {
        'mistral_common': common,
        'mistral_common.protocol': protocol,
        'mistral_common.protocol.instruct': instruct,
        'mistral_common.protocol.instruct.messages': messages,
        'mistral_common.protocol.instruct.request': request,
        'mistral_inference': inference,
        'mistral_inference.generate': generate_module,
    }


class TestAdditionalChatInner(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tempdir.name) / 'image.png')
        Image.new('RGB', (2, 2), color='white').save(self.image_path)
        self.messages = [
            {'role': 'user', 'content': [
                {'type': 'image', 'value': self.image_path},
                {'type': 'text', 'value': 'Describe this.'},
            ]},
            {'role': 'assistant', 'content': [
                {'type': 'text', 'value': 'A white square.'},
            ]},
            {'role': 'user', 'content': [
                {'type': 'text', 'value': 'What color?'},
            ]},
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    def test_gemma3_chat_inner_preserves_multimodal_turns(self):
        module = _load_vlm_module('gemma')
        model = module.Gemma3.__new__(module.Gemma3)
        model.use_vllm = False
        model.system_prompt = 'Be helpful.'
        model.processor = FakeGemmaProcessor()
        model.model = FakeModel()
        model.device = 'cuda'
        model.kwargs = {'max_new_tokens': 16}

        response = model.chat_inner(self.messages)

        self.assertEqual(response, 'gemma-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['system', 'user', 'assistant', 'user'],
        )
        self.assertEqual(
            model.processor.messages[1]['content'][0],
            {'type': 'image', 'url': self.image_path},
        )
        self.assertEqual(
            model.processor.messages[-1]['content'],
            [{'type': 'text', 'text': 'What color?'}],
        )

    def test_gemma3_single_turn_generation_still_uses_user_message(self):
        module = _load_vlm_module('gemma')
        model = module.Gemma3.__new__(module.Gemma3)
        model.use_vllm = False
        model.system_prompt = None
        model.processor = FakeGemmaProcessor()
        model.model = FakeModel()
        model.device = 'cuda'
        model.kwargs = {'max_new_tokens': 16}

        response = model.generate_inner(self.messages[0]['content'])

        self.assertEqual(response, 'gemma-answer')
        self.assertEqual([turn['role'] for turn in model.processor.messages], ['user'])

    def test_gemma3_vllm_chat_inner_preserves_history_and_images(self):
        module = _load_vlm_module('gemma')
        model = module.Gemma3.__new__(module.Gemma3)
        model.use_vllm = True
        model.system_prompt = 'Be helpful.'
        model.processor = FakeGemmaProcessor()
        model.llm = FakeVLLM()
        model.limit_mm_per_prompt = 24
        model.kwargs = {'max_new_tokens': 16}

        import base64
        image_bytes = Path(self.image_path).read_bytes()
        model.encode_image = lambda path: base64.b64encode(image_bytes).decode()
        vllm = types.ModuleType('vllm')
        vllm.SamplingParams = SamplingParams

        with mock.patch.dict(sys.modules, {'vllm': vllm}):
            response = model.chat_inner(self.messages)

        self.assertEqual(response, 'gemma-vllm-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['system', 'user', 'assistant', 'user'],
        )
        self.assertEqual(len(model.llm.request['multi_modal_data']['image']), 1)

    def test_mantis_chat_inner_builds_native_conversation_history(self):
        module = _load_vlm_module('mantis')
        model = module.Mantis.__new__(module.Mantis)
        model._is_idefics = False
        model.processor = FakeMantisProcessor()
        model.model = FakeModel('Mantis-Llama-3')
        model.conv_templates = {'llama_3': FakeConversation()}
        model.default_conv = FakeConversation()
        model.kwargs = {'max_new_tokens': 16}

        response = model.chat_inner(self.messages)

        self.assertEqual(response, 'mantis-answer')
        self.assertEqual(FakeConversation.last_messages, [
            ['USER', '<image>\nDescribe this.'],
            ['ASSISTANT', 'A white square.'],
            ['USER', 'What color?'],
            ['ASSISTANT', ''],
        ])
        self.assertEqual(len(model.processor.images), 1)

    def test_mantis_single_turn_generation_still_works(self):
        module = _load_vlm_module('mantis')
        model = module.Mantis.__new__(module.Mantis)
        model._is_idefics = False
        model.processor = FakeMantisProcessor()
        model.model = FakeModel('Mantis-Llama-3')
        model.conv_templates = {'llama_3': FakeConversation()}
        model.default_conv = FakeConversation()
        model.kwargs = {'max_new_tokens': 16}

        response = model.generate_inner(self.messages[0]['content'])

        self.assertEqual(response, 'mantis-answer')
        self.assertEqual(len(model.processor.images), 1)

    def test_mantis_idefics_chat_inner_uses_native_chat_template(self):
        module = _load_vlm_module('mantis')
        model = module.Mantis.__new__(module.Mantis)
        model._is_idefics = True
        model.processor = FakeMantisIdeficsProcessor()
        model.model = FakeModel()
        model.kwargs = {'max_new_tokens': 16}

        response = model.chat_inner(self.messages)

        self.assertEqual(response, 'mantis-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['user', 'assistant', 'user'],
        )
        self.assertEqual(model.processor.prompt, 'mantis-idefics-prompt')
        self.assertEqual(len(model.processor.images), 1)

    def test_pixtral_chat_inner_uses_mistral_multi_turn_request(self):
        module = _load_vlm_module('pixtral')
        calls = []

        def generate(*args, **kwargs):
            calls.append((args, kwargs))
            return [[3]], None

        model = module.Pixtral.__new__(module.Pixtral)
        model.tokenizer = FakeMistralTokenizer()
        model.model = object()
        model.max_tokens = 16

        with mock.patch.dict(sys.modules, _mistral_modules(generate)):
            response = model.chat_inner(self.messages)

        request = model.tokenizer.request
        self.assertEqual(response, 'pixtral-answer')
        self.assertEqual([turn.role for turn in request.messages], ['user', 'assistant', 'user'])
        self.assertIsInstance(request.messages[0].content[0], ImageURLChunk)
        self.assertEqual(request.messages[1].content[0].text, 'A white square.')
        self.assertEqual(len(calls), 1)

    def test_pixtral_single_turn_generation_still_works(self):
        module = _load_vlm_module('pixtral')

        def generate(*args, **kwargs):
            return [[3]], None

        model = module.Pixtral.__new__(module.Pixtral)
        model.tokenizer = FakeMistralTokenizer()
        model.model = object()
        model.max_tokens = 16

        with mock.patch.dict(sys.modules, _mistral_modules(generate)):
            response = model.generate_inner(self.messages[0]['content'])

        self.assertEqual(response, 'pixtral-answer')
        self.assertEqual([turn.role for turn in model.tokenizer.request.messages], ['user'])


if __name__ == '__main__':
    unittest.main()
