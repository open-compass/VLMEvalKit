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


class FakeInputs(dict):

    def __init__(self):
        super().__init__(input_ids=FakeInputIds(), pixel_values=FakeTensor())
        self.input_ids = [[1, 2]]

    def to(self, *args, **kwargs):
        return self


class FakeInputIds:
    shape = (1, 2)

    def size(self, dim):
        return self.shape[dim]

    def to(self, *args, **kwargs):
        return self


class FakeTensor:

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def __getitem__(self, key):
        return self


class FakeGenerated:

    def __iter__(self):
        return iter([[1, 2, 3]])

    def __getitem__(self, key):
        return FakeTensor()


def _base_modules():
    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = ['vlmeval']
    vlm = types.ModuleType('vlmeval.vlm')
    vlm.__path__ = ['vlmeval/vlm']
    base = types.ModuleType('vlmeval.vlm.base')
    base.BaseModel = FakeBaseModel
    return {
        'vlmeval': vlmeval,
        'vlmeval.vlm': vlm,
        'vlmeval.vlm.base': base,
    }


def _load_vlm_module(module_name, extra_modules=None):
    modules = _base_modules()
    modules.update(extra_modules or {})
    full_name = f'vlmeval.vlm.{module_name}'
    with mock.patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(full_name, f'vlmeval/vlm/{module_name}.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        sys.modules.pop(full_name, None)
        return module


class FakeKimiProcessor:

    def __init__(self):
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'kimi-prompt'

    def __call__(self, *, images, text, **kwargs):
        self.images = images
        self.text = text
        return FakeInputs()

    def batch_decode(self, *args, **kwargs):
        return ['kimi-answer']


class FakeAriaProcessor:

    def __init__(self):
        self.messages = None
        self.images = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return 'aria-prompt'

    def __call__(self, *, text, images=None, **kwargs):
        self.images = images
        return FakeInputs()


class FakePhiProcessor:

    def __init__(self):
        self.prompt = None
        self.images = None

    def __call__(self, *, text, images, **kwargs):
        self.prompt = text
        self.images = images
        return FakeInputs()

    def batch_decode(self, *args, **kwargs):
        return ['phi-answer']


class FakeModel:
    device = 'cuda'
    dtype = 'bfloat16'

    def generate(self, **kwargs):
        return FakeGenerated()


class FakeTokenizer:

    def decode(self, *args, **kwargs):
        return 'aria-answer<|im_end|>'


class TestVLMChatInner(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.tempdir.name) / 'image.png')
        Image.new('RGB', (2, 2), color='white').save(self.image_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_kimi_chat_inner_preserves_turns_and_images(self):
        transformers = types.ModuleType('transformers')
        transformers.AutoModelForCausalLM = object
        transformers.AutoProcessor = object
        module = _load_vlm_module('kimi_vl', {'transformers': transformers})

        model = module.KimiVL.__new__(module.KimiVL)
        model.processor = FakeKimiProcessor()
        model.model = FakeModel()
        model.temperature = 0.0
        model.max_tokens = 32
        model.extract_summary = False

        response = model.chat_inner([
            {'role': 'user', 'content': [
                {'type': 'image', 'value': self.image_path},
                {'type': 'text', 'value': 'Describe this.'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': 'A white square.'}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': 'What color?'}]},
        ])

        self.assertEqual(response, 'kimi-answer')
        self.assertEqual([turn['role'] for turn in model.processor.messages], ['user', 'assistant', 'user'])
        self.assertEqual(len(model.processor.images), 1)
        self.assertEqual(model.processor.messages[-1]['content'], [{'type': 'text', 'text': 'What color?'}])

    def test_aria_chat_inner_uses_native_chat_template(self):
        dataset = types.ModuleType('vlmeval.dataset')
        dataset.DATASET_MODALITY = lambda name: 'IMAGE'
        dataset.DATASET_TYPE = lambda name: 'VQA'
        smp = types.ModuleType('vlmeval.smp')
        smp.listinstr = lambda needles, value: any(needle in value for needle in needles)
        torch = types.ModuleType('torch')
        torch.bfloat16 = 'bfloat16'
        torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
        module = _load_vlm_module('aria', {
            'torch': torch,
            'vlmeval.dataset': dataset,
            'vlmeval.smp': smp,
        })

        model = module.Aria.__new__(module.Aria)
        model.processor = FakeAriaProcessor()
        model.tokenizer = FakeTokenizer()
        model.model = FakeModel()
        model.kwargs = {'max_new_tokens': 16}

        response = model.chat_inner([
            {'role': 'user', 'content': [
                {'type': 'image', 'value': self.image_path},
                {'type': 'text', 'value': 'Describe this.'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': 'A white square.'}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': 'What color?'}]},
        ], dataset=None)

        self.assertEqual(response, 'aria-answer')
        self.assertEqual([turn['role'] for turn in model.processor.messages], ['user', 'assistant', 'user'])
        self.assertEqual(len(model.processor.images), 1)
        self.assertEqual(model.processor.messages[0]['content'][0], {'type': 'image'})

    def test_phi4_chat_inner_numbers_images_across_turns(self):
        module = _load_vlm_module('phi4_multimodal')
        model = module.Phi4Multimodal.__new__(module.Phi4Multimodal)
        model.processor = FakePhiProcessor()
        model.model = FakeModel()
        model.generation_config = object()

        response = model.chat_inner([
            {'role': 'user', 'content': [
                {'type': 'image', 'value': self.image_path},
                {'type': 'text', 'value': 'Describe this.'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': 'A white square.'}]},
            {'role': 'user', 'content': [{'type': 'text', 'value': 'What color?'}]},
        ])

        self.assertEqual(response, 'phi-answer')
        self.assertEqual(len(model.processor.images), 1)
        self.assertEqual(
            model.processor.prompt,
            '<|user|><|image_1|>Describe this.<|end|>'
            '<|assistant|>A white square.<|end|>'
            '<|user|>What color?<|end|><|assistant|>',
        )


if __name__ == '__main__':
    unittest.main()
