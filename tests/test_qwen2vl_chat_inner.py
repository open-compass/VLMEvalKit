import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

MODEL_FILE = Path(__file__).resolve().parents[1] / 'vlmeval' / 'vlm' / 'qwen2_vl' / 'model.py'


class FakeBaseModel:

    def __init__(self, **kwargs):
        pass


def _stub_modules():
    """Light stand-ins for the heavy imports of qwen2_vl/model.py, so the adapter loads without GPUs or weights."""
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: None, device_count=lambda: 0)
    hub = types.ModuleType('huggingface_hub')
    hub.snapshot_download = lambda **kwargs: None
    transformers = types.ModuleType('transformers')
    transformers.StoppingCriteria = object

    vlmeval = types.ModuleType('vlmeval')
    vlmeval.__path__ = []
    dataset = types.ModuleType('vlmeval.dataset')
    dataset.DATASET_MODALITY = lambda dataset, **kwargs: 'IMAGE'
    smp = types.ModuleType('vlmeval.smp')
    smp.get_cache_path = lambda *args, **kwargs: None
    smp.get_gpu_memory = lambda: []
    smp.listinstr = lambda lst, s: any(item in s for item in lst)
    vlm = types.ModuleType('vlmeval.vlm')
    vlm.__path__ = []
    base = types.ModuleType('vlmeval.vlm.base')
    base.BaseModel = FakeBaseModel
    package = types.ModuleType('vlmeval.vlm.qwen2_vl')
    package.__path__ = [str(MODEL_FILE.parent)]
    prompt = types.ModuleType('vlmeval.vlm.qwen2_vl.prompt')
    prompt.Qwen2VLPromptMixin = type('Qwen2VLPromptMixin', (), {})
    return {
        'torch': torch,
        'huggingface_hub': hub,
        'transformers': transformers,
        'vlmeval': vlmeval,
        'vlmeval.dataset': dataset,
        'vlmeval.smp': smp,
        'vlmeval.vlm': vlm,
        'vlmeval.vlm.base': base,
        'vlmeval.vlm.qwen2_vl': package,
        'vlmeval.vlm.qwen2_vl.prompt': prompt,
    }


def load_model_module():
    name = 'vlmeval.vlm.qwen2_vl.model'
    with mock.patch.dict(sys.modules, _stub_modules()):
        spec = importlib.util.spec_from_file_location(name, MODEL_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


class FakeInputs(dict):

    def __init__(self):
        super().__init__(input_ids=[[1, 2]])
        self.input_ids = [[1, 2]]

    def to(self, device):
        return self


class FakeProcessor:

    def __init__(self):
        self.tokenizer = types.SimpleNamespace(batch_decode=lambda ids, **kwargs: ['the answer'])

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_input = messages
        return 'prompt'

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return FakeInputs()


class FakeModel:

    def generate(self, **kwargs):
        return [[1, 2, 3]]


def fake_qwen_vl_utils(calls):
    module = types.ModuleType('qwen_vl_utils')

    def process_vision_info(messages):
        calls.append(messages)
        # the transformers path passes a batch of conversations, the vLLM path a single one
        conversations = messages if messages and isinstance(messages[0], list) else [messages]
        images = [item['image'] for conversation in conversations for msg in conversation
                  if isinstance(msg['content'], list) for item in msg['content'] if item['type'] == 'image']
        return images, None

    module.process_vision_info = process_vision_info
    return module


class TestQwen2VLChatInner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.module = load_model_module()

    def make_model(self, use_vllm=False, use_lmdeploy=False, system_prompt=None):
        model = self.module.Qwen2VLChat.__new__(self.module.Qwen2VLChat)
        model.model_path = 'Qwen/Qwen2-VL-2B-Instruct'
        model.min_pixels = model.max_pixels = model.total_pixels = None
        model.fps, model.nframe = 2, 128
        model.system_prompt = system_prompt
        model.verbose = model.post_process = False
        model.use_vllm, model.use_lmdeploy = use_vllm, use_lmdeploy
        model.limit_mm_per_prompt = self.module.VLLM_MAX_IMAGE_INPUT_NUM
        model.max_new_tokens = 16
        model.generate_kwargs = {}
        model.processor = FakeProcessor()
        model.model = FakeModel()
        return model

    def conversation(self):
        return [
            {'role': 'user', 'content': [
                {'type': 'image', 'value': 'file:///images/cat.png'},
                {'type': 'text', 'value': 'What animal is this?'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'value': 'A cat.'}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'value': 'file:///images/dog.png'},
                {'type': 'text', 'value': 'And this one?'},
            ]},
        ]

    def test_transformers_keeps_history_with_images_on_user_turns(self):
        model = self.make_model()
        calls = []
        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_qwen_vl_utils(calls)}):
            answer = model.chat_inner(self.conversation(), dataset='MMDU')

        self.assertEqual(answer, 'the answer')
        expected = [
            {'role': 'user', 'content': [
                {'type': 'image', 'image': 'file:///images/cat.png'},
                {'type': 'text', 'text': 'What animal is this?'},
            ]},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': 'A cat.'}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': 'file:///images/dog.png'},
                {'type': 'text', 'text': 'And this one?'},
            ]},
        ]
        self.assertEqual(model.processor.chat_template_input, [expected])
        self.assertEqual(calls, [[expected]])
        self.assertEqual(model.processor.call_kwargs['images'], ['file:///images/cat.png', 'file:///images/dog.png'])

    def test_system_prompt_and_system_turn_are_merged_first(self):
        model = self.make_model(system_prompt='Be brief.')
        conversation = [{'role': 'system', 'content': [{'type': 'text', 'value': 'Answer in English.'}]}]
        conversation += self.conversation()

        messages = model._build_multi_turn_messages(conversation)

        self.assertEqual(messages[0], {'role': 'system', 'content': 'Be brief.\nAnswer in English.'})
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])

    def test_vllm_receives_the_whole_history(self):
        model = self.make_model(use_vllm=True)
        model.llm = mock.Mock()
        model.llm.generate.return_value = [types.SimpleNamespace(outputs=[types.SimpleNamespace(text='vllm answer')])]
        vllm = types.ModuleType('vllm')
        vllm.SamplingParams = lambda **kwargs: kwargs
        calls = []
        with mock.patch.dict(sys.modules, {'vllm': vllm, 'qwen_vl_utils': fake_qwen_vl_utils(calls)}):
            answer = model.chat_inner(self.conversation(), dataset='MMDU')

        self.assertEqual(answer, 'vllm answer')
        roles = [m['role'] for m in model.processor.chat_template_input]
        self.assertEqual(roles, ['user', 'assistant', 'user'])
        request = model.llm.generate.call_args.args[0]
        self.assertEqual(request['multi_modal_data']['image'], ['file:///images/cat.png', 'file:///images/dog.png'])

    def test_single_turn_generation_is_unchanged(self):
        model = self.make_model()
        calls = []
        with mock.patch.dict(sys.modules, {'qwen_vl_utils': fake_qwen_vl_utils(calls)}):
            answer = model.generate_inner([{'type': 'text', 'value': 'Hello'}], dataset='MMBench')

        self.assertEqual(answer, 'the answer')
        self.assertEqual(model.processor.chat_template_input,
                         [[{'role': 'user', 'content': [{'type': 'text', 'text': 'Hello'}]}]])

    def test_assistant_turn_must_be_text(self):
        model = self.make_model()
        conversation = self.conversation()
        conversation[1]['content'].append({'type': 'image', 'value': 'file:///images/cat.png'})

        with self.assertRaises(ValueError):
            model.chat_inner(conversation)

    def test_lmdeploy_is_not_supported(self):
        model = self.make_model(use_lmdeploy=True)

        with self.assertRaises(NotImplementedError):
            model.chat_inner(self.conversation())


if __name__ == '__main__':
    unittest.main()
