import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


class FakeBaseModel:
    pass


class FakeInputIds:
    shape = (1, 2)


class FakeInputs(dict):

    def __init__(self):
        super().__init__(input_ids=FakeInputIds())

    def to(self, *args, **kwargs):
        return self


class FakeTensor:

    def __getitem__(self, key):
        return self


class FakeLlamaProcessor:

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        if kwargs.get('tokenize') is False:
            return 'llama-prompt'
        return FakeInputs()

    def batch_decode(self, *args, **kwargs):
        return ['llama4-answer<|eot|>']


class FakeLlamaModel:
    device = 'cuda'

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeTensor()


class FakeVisionProcessor:

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'vision-prompt'

    def __call__(self, images, text, **kwargs):
        self.images = images
        self.text = text
        return FakeInputs()

    def decode(self, *args, **kwargs):
        return 'vision-answer<|eot_id|>'


class FakeVisionModel:

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


class FakeQwenInputIds(list):
    pass


class FakeQwenInputs(dict):

    def __init__(self):
        self.input_ids = [FakeQwenInputIds([1, 2])]
        super().__init__(input_ids=self.input_ids)

    def to(self, *args, **kwargs):
        return self


class FakeQwenTokenizer:

    def batch_decode(self, *args, **kwargs):
        return ['qwen-answer']


class FakeQwenProcessor:

    def __init__(self):
        self.tokenizer = FakeQwenTokenizer()

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'qwen-prompt'

    def __call__(self, **kwargs):
        self.processor_kwargs = kwargs
        return FakeQwenInputs()


class FakeQwenModel:
    device = 'cuda'
    dtype = 'float16'

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [[1, 2, 3]]


class SamplingParams:

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeVLLM:

    def generate(self, request, sampling_params):
        self.request = request
        self.sampling_params = sampling_params
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='vllm-answer')])]


class FakeGenerationConfig:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.random_seed = 'unset'


class FakeLMDeployPipeline:

    def __call__(self, messages, gen_config):
        self.messages = messages
        self.gen_config = gen_config
        return [types.SimpleNamespace(text='lmdeploy-answer')]


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
    smp.get_gpu_memory = lambda: [16]
    smp.listinstr = lambda needles, value: any(needle.lower() in value.lower() for needle in needles)

    pandas = types.ModuleType('pandas')
    pandas.isna = lambda value: False
    torch = types.ModuleType('torch')
    torch.bfloat16 = 'bfloat16'
    torch.cuda = types.SimpleNamespace(device_count=lambda: 1, empty_cache=lambda: None)
    return {
        'vlmeval': vlmeval,
        'vlmeval.vlm': vlm,
        'vlmeval.vlm.base': base,
        'vlmeval.dataset': dataset,
        'vlmeval.smp': smp,
        'pandas': pandas,
        'torch': torch,
    }


def _load_vlm_module(module_name, relative_path):
    modules = _base_modules()
    full_name = f'vlmeval.vlm.{module_name}'
    if module_name == 'qwen3_vl.model':
        package = types.ModuleType('vlmeval.vlm.qwen3_vl')
        package.__path__ = ['vlmeval/vlm/qwen3_vl']
        prompt = types.ModuleType('vlmeval.vlm.qwen3_vl.prompt')

        class Qwen3VLPromptMixin:

            def __init__(self, **kwargs):
                pass

        prompt.Qwen3VLPromptMixin = Qwen3VLPromptMixin
        modules['vlmeval.vlm.qwen3_vl'] = package
        modules['vlmeval.vlm.qwen3_vl.prompt'] = prompt

    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(full_name, relative_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        sys.modules.pop(full_name, None)
        return module


class TestLlamaAndQwenChatInner(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tempdir.name) / 'image.png')
        self.second_image_path = str(Path(self.tempdir.name) / 'second.png')
        Image.new('RGB', (2, 2), color='white').save(self.image_path)
        Image.new('RGB', (2, 2), color='black').save(self.second_image_path)
        self.messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'value': self.image_path},
                    {'type': 'text', 'value': 'Describe this.'},
                ],
            },
            {
                'role': 'assistant',
                'content': [{'type': 'text', 'value': 'A white square.'}],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'value': 'Compare it with this.'},
                    {'type': 'image', 'value': self.second_image_path},
                ],
            },
        ]

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _llama4(module):
        model = module.llama4.__new__(module.llama4)
        model.use_vllm = False
        model.use_lmdeploy = False
        model.system_prompt = 'Be precise.'
        model.processor = FakeLlamaProcessor()
        model.model = FakeLlamaModel()
        model.generate_kwargs = {
            'max_new_tokens': 16,
            'top_p': 0.8,
            'top_k': 10,
            'temperature': 0.1,
            'repetition_penalty': 1.0,
        }
        model.limit_mm_per_prompt = 10
        return model

    def test_llama4_transformers_preserves_history(self):
        module = _load_vlm_module('llama4', 'vlmeval/vlm/llama4.py')
        model = self._llama4(module)

        response = model.chat_inner(self.messages)

        self.assertEqual(response, 'llama4-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['system', 'user', 'assistant', 'user'],
        )
        self.assertEqual(
            model.processor.messages[-1]['content'][0],
            {'type': 'text', 'text': 'Compare it with this.'},
        )
        self.assertEqual(model.processor.messages[-1]['content'][1]['type'], 'image')
        self.assertEqual(model.model.generate_kwargs['max_new_tokens'], 8192)

    def test_llama4_single_turn_generation_still_uses_user_message(self):
        module = _load_vlm_module('llama4', 'vlmeval/vlm/llama4.py')
        model = self._llama4(module)
        model.system_prompt = None

        response = model.generate_inner(self.messages[0]['content'])

        self.assertEqual(response, 'llama4-answer')
        self.assertEqual([turn['role'] for turn in model.processor.messages], ['user'])

    def test_llama4_vllm_preserves_history_and_images(self):
        module = _load_vlm_module('llama4', 'vlmeval/vlm/llama4.py')
        model = self._llama4(module)
        model.use_vllm = True
        model.llm = FakeVLLM()
        vllm = types.ModuleType('vllm')
        vllm.SamplingParams = SamplingParams

        with mock.patch.dict(sys.modules, {'vllm': vllm}):
            response = model.chat_inner(self.messages)

        self.assertEqual(response, 'vllm-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['system', 'user', 'assistant', 'user'],
        )
        images = model.llm.request['multi_modal_data']['image']
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 0))

    def test_llama4_lmdeploy_preserves_history(self):
        module = _load_vlm_module('llama4', 'vlmeval/vlm/llama4.py')
        model = self._llama4(module)
        model.use_lmdeploy = True
        model.model = FakeLMDeployPipeline()
        model.message_to_lmdeploy = lambda content, system_prompt=None: [[{
            'role': 'user',
            'content': [item['value'] for item in content],
        }]]
        lmdeploy = types.ModuleType('lmdeploy')
        lmdeploy.GenerationConfig = FakeGenerationConfig

        with mock.patch.dict(sys.modules, {'lmdeploy': lmdeploy}):
            response = model.chat_inner(self.messages)

        self.assertEqual(response, 'lmdeploy-answer')
        history = model.model.messages[0]
        self.assertEqual([turn['role'] for turn in history], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(history[2]['content'], 'A white square.')

    def test_llama_vision_preserves_history_and_single_turn(self):
        module = _load_vlm_module('llama_vision', 'vlmeval/vlm/llama_vision.py')
        model = module.llama_vision.__new__(module.llama_vision)
        model.processor = FakeVisionProcessor()
        model.model = FakeVisionModel()
        model.device = 'cuda'
        model.kwargs = {'max_new_tokens': 16}
        model.model_name = 'meta-llama/Llama-3.2-11B-Vision-Instruct'

        response = model.chat_inner(self.messages)

        self.assertEqual(response, 'vision-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['user', 'assistant', 'user'],
        )
        self.assertEqual(len(model.processor.images), 2)
        self.assertEqual(model.processor.images[0].mode, 'RGB')
        self.assertEqual(model.processor.images[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(model.processor.images[1].getpixel((0, 0)), (0, 0, 0))

        response = model.generate_inner(self.messages[0]['content'])
        self.assertEqual(response, 'vision-answer')
        self.assertEqual([turn['role'] for turn in model.processor.messages], ['user'])

    @staticmethod
    def _qwen3(module):
        model = module.Qwen3VLChat.__new__(module.Qwen3VLChat)
        model.model_path = 'Qwen/Qwen3-VL-8B-Instruct'
        model.system_prompt = 'Be precise.'
        model.processor = FakeQwenProcessor()
        model.model = FakeQwenModel()
        model.use_vllm = False
        model.verbose = False
        model.post_process = False
        model.min_pixels = None
        model.max_pixels = None
        model.total_pixels = None
        model.fps = 2
        model.nframe = 128
        model.FRAME_FACTOR = 2
        model.use_audio_in_video = False
        model.generate_kwargs = {'max_new_tokens': 16}
        model.temperature = 0.1
        model.max_new_tokens = 16
        model.top_p = 0.8
        model.top_k = 20
        model.repetition_penalty = 1.0
        model.presence_penalty = 1.5
        return model

    @staticmethod
    def _qwen_modules():
        qwen_utils = types.ModuleType('qwen_vl_utils')

        def process_vision_info(messages, **kwargs):
            images = [
                item['image']
                for turn in messages
                if isinstance(turn['content'], list)
                for item in turn['content']
                if item['type'] == 'image'
            ]
            return images, None, {}

        qwen_utils.process_vision_info = process_vision_info
        return {'qwen_vl_utils': qwen_utils}

    def test_qwen3_transformers_preserves_history(self):
        module = _load_vlm_module('qwen3_vl.model', 'vlmeval/vlm/qwen3_vl/model.py')
        model = self._qwen3(module)

        with mock.patch.dict(sys.modules, self._qwen_modules()):
            response = model.chat_inner(self.messages)

        self.assertEqual(response, 'qwen-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['system', 'user', 'assistant', 'user'],
        )
        self.assertTrue(model.processor.messages[1]['content'][0]['image'].startswith('file://'))
        self.assertEqual(
            model.processor.processor_kwargs['images'],
            [f'file://{self.image_path}', f'file://{self.second_image_path}'],
        )

    def test_qwen3_vllm_and_single_turn_routing(self):
        module = _load_vlm_module('qwen3_vl.model', 'vlmeval/vlm/qwen3_vl/model.py')
        model = self._qwen3(module)
        model.use_vllm = True
        model.llm = FakeVLLM()
        vllm = types.ModuleType('vllm')
        vllm.SamplingParams = SamplingParams
        modules = self._qwen_modules()
        modules['vllm'] = vllm

        with mock.patch.dict(sys.modules, modules):
            response = model.chat_inner(self.messages)

        self.assertEqual(response, 'vllm-answer')
        self.assertEqual(
            [turn['role'] for turn in model.processor.messages],
            ['system', 'user', 'assistant', 'user'],
        )
        self.assertEqual(
            model.llm.request[0]['multi_modal_data']['image'],
            [f'file://{self.image_path}', f'file://{self.second_image_path}'],
        )

        model.use_vllm = False
        model.system_prompt = None
        with mock.patch.dict(sys.modules, self._qwen_modules()):
            response = model.generate_inner(self.messages[0]['content'])
        self.assertEqual(response, 'qwen-answer')
        self.assertEqual([turn['role'] for turn in model.processor.messages], ['user'])

    def test_adapters_reject_non_text_assistant_content(self):
        invalid = [
            self.messages[0],
            {
                'role': 'assistant',
                'content': [{'type': 'image', 'value': self.image_path}],
            },
            self.messages[2],
        ]

        llama4_module = _load_vlm_module('llama4', 'vlmeval/vlm/llama4.py')
        with self.assertRaisesRegex(ValueError, 'images in user turns'):
            self._llama4(llama4_module).chat_inner(invalid)

        vision_module = _load_vlm_module('llama_vision', 'vlmeval/vlm/llama_vision.py')
        vision = vision_module.llama_vision.__new__(vision_module.llama_vision)
        with self.assertRaisesRegex(ValueError, 'images in user turns'):
            vision._prepare_chat(invalid)

        qwen_module = _load_vlm_module('qwen3_vl.model', 'vlmeval/vlm/qwen3_vl/model.py')
        with self.assertRaisesRegex(ValueError, 'text in assistant turns'):
            self._qwen3(qwen_module).chat_inner(invalid)


if __name__ == '__main__':
    unittest.main()
