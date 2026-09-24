import importlib.util
import sys
import unittest
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


def load_vqtoken_module():
    """Load the optional wrapper without importing every VLMEvalKit backend."""
    package_names = ['vlmeval', 'vlmeval.vlm', 'vlmeval.vlm.llava']
    packages = {}
    for package_name in package_names:
        package = ModuleType(package_name)
        package.__path__ = []
        packages[package_name] = package

    llava_module = ModuleType('vlmeval.vlm.llava.llava')

    class FakeLLaVAOneVision:
        pass

    llava_module.LLaVA_OneVision = FakeLLaVAOneVision
    packages['vlmeval.vlm.llava.llava'] = llava_module

    module_name = 'vlmeval.vlm.llava.vqtoken'
    module_path = Path(__file__).parents[1] / 'vlmeval' / 'vlm' / 'llava' / 'vqtoken.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, packages):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


def load_llava_module():
    """Load the OneVision parent with lightweight VLMEvalKit import stubs."""
    packages = {}
    for package_name in ['vlmeval', 'vlmeval.vlm', 'vlmeval.vlm.llava']:
        package = ModuleType(package_name)
        package.__path__ = []
        packages[package_name] = package

    dataset_module = ModuleType('vlmeval.dataset')
    dataset_module.DATASET_MODALITY = (
        lambda dataset: 'VIDEO' if dataset == 'video-dataset' else 'IMAGE')
    dataset_module.DATASET_TYPE = lambda dataset: 'VQA'
    packages['vlmeval.dataset'] = dataset_module

    smp_module = ModuleType('vlmeval.smp')
    smp_module.cn_string = lambda value: False
    smp_module.encode_image_to_base64 = lambda image: ''
    smp_module.splitlen = lambda value: 2
    packages['vlmeval.smp'] = smp_module

    base_module = ModuleType('vlmeval.vlm.base')

    class FakeBaseModel:
        pass

    base_module.BaseModel = FakeBaseModel
    packages['vlmeval.vlm.base'] = base_module

    packages['numpy'] = ModuleType('numpy')
    packages['pandas'] = ModuleType('pandas')
    torch_module = ModuleType('torch')
    torch_module.ones_like = lambda value: value
    packages['torch'] = torch_module
    pil_module = ModuleType('PIL')
    pil_module.Image = SimpleNamespace()
    packages['PIL'] = pil_module

    module_name = 'vlmeval.vlm.llava.llava'
    module_path = Path(__file__).parents[1] / 'vlmeval' / 'vlm' / 'llava' / 'llava.py'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, packages):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module


class FakeTensor:

    def half(self):
        return self

    def cuda(self):
        return self

    def unsqueeze(self, dim):
        return self


class FakeVideoFrames:

    shape = (4, 8, 10, 3)

    def __len__(self):
        return self.shape[0]


class FakeConversation:

    roles = ('user', 'assistant')
    sep = '<sep>'
    sep2 = '<sep2>'
    sep_style = 'single'

    def __init__(self):
        self.messages = []

    def append_message(self, role, content):
        self.messages.append((role, content))

    def get_prompt(self):
        return '\n'.join(content or '' for _, content in self.messages)


class TestVQToken(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.vqtoken = load_vqtoken_module()

    def test_wrapper_passes_public_learned_attention_config(self):
        runtime = SimpleNamespace(
            VQTOKEN_CAPABILITIES={
                'modes': ('attention', 'centroids'),
                'selection_methods': ('fixed', 'elbow', 'silhouette'),
            })
        parent_class = self.vqtoken.LLaVA_OneVision
        with mock.patch.object(
                self.vqtoken, 'import_module', return_value=runtime), mock.patch.object(
                    parent_class, '__init__', return_value=None) as parent_init:
            model = self.vqtoken.LLaVA_OneVision_VQToken(
                vqtoken_selection_method='elbow',
                vqtoken_min_clusters=12,
                vqtoken_max_clusters=32,
                max_frames_num=12,
            )

        kwargs = parent_init.call_args.kwargs
        self.assertEqual(kwargs['model_path'], 'haichaozhang/VQ-Token-llava-ov-0.5b')
        self.assertEqual(kwargs['model_name'], 'llava_qwen')
        self.assertEqual(
            model._get_model_overwrite_config(),
            {
                'use_vqtoken': True,
                'vqtoken_mode': 'attention',
                'vqtoken_selection_method': 'elbow',
                'vqtoken_min_clusters': 12,
                'vqtoken_max_clusters': 32,
                'use_embedded_vision': True,
                'mm_spatial_pool_stride': 2,
                'mm_spatial_pool_mode': 'bilinear',
            },
        )
        self.assertEqual(model.nframe, 12)
        self.assertFalse(model.force_sample)

    def test_wrapper_forces_uniform_sampling_without_timestamp_prompt(self):
        runtime = SimpleNamespace(
            VQTOKEN_CAPABILITIES={
                'modes': ('attention', 'centroids'),
                'selection_methods': ('fixed', 'elbow', 'silhouette'),
            })
        parent_class = self.vqtoken.LLaVA_OneVision
        with mock.patch.object(
                self.vqtoken, 'import_module', return_value=runtime), mock.patch.object(
                    parent_class, '__init__', return_value=None):
            model = self.vqtoken.LLaVA_OneVision_VQToken()

        with mock.patch.object(
                parent_class, 'load_video', create=True,
                return_value=('frames', 'times', 1.0)) as parent_load:
            result = model.load_video('demo.mp4', 32, fps=1, force_sample=False)

        self.assertEqual(result, ('frames', 'times', 1.0))
        parent_load.assert_called_once_with('demo.mp4', 32, 1, True)

    def test_invalid_or_non_public_configuration_is_rejected(self):
        invalid = [
            ('random', 12, 32),
            ('fixed', 0, 32),
            ('elbow', 33, 32),
            ('fixed', True, 32),
        ]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.vqtoken.validate_cluster_config(*config)

    def test_fixed_selection_uses_max_clusters_as_k(self):
        self.vqtoken.validate_cluster_config('fixed', 12, 8)

    def test_custom_checkpoint_embedded_vision_is_opt_in(self):
        runtime = SimpleNamespace(VQTOKEN_CAPABILITIES={
            'modes': ('centroids', ),
            'selection_methods': ('fixed', 'elbow', 'silhouette'),
        })
        parent_class = self.vqtoken.LLaVA_OneVision
        with mock.patch.object(
                self.vqtoken, 'import_module', return_value=runtime), mock.patch.object(
                    parent_class, '__init__', return_value=None):
            inferred = self.vqtoken.LLaVA_OneVision_VQToken(
                model_path='local/custom', vqtoken_mode='centroids')
            explicit = self.vqtoken.LLaVA_OneVision_VQToken(
                model_path='local/custom', vqtoken_mode='centroids', use_embedded_vision=True)

        self.assertFalse(inferred._get_model_overwrite_config()['use_embedded_vision'])
        self.assertTrue(explicit._get_model_overwrite_config()['use_embedded_vision'])

    def test_local_checkpoint_embedded_vision_is_detected(self):
        detector = mock.MagicMock(return_value=True)
        runtime = SimpleNamespace(
            VQTOKEN_CAPABILITIES={
                'modes': ('centroids', ),
                'selection_methods': ('fixed', 'elbow', 'silhouette'),
            },
            has_embedded_vision_weights=detector,
        )
        parent_class = self.vqtoken.LLaVA_OneVision
        with mock.patch.object(
                self.vqtoken, 'import_module', return_value=runtime), mock.patch.object(
                    parent_class, '__init__', return_value=None):
            model = self.vqtoken.LLaVA_OneVision_VQToken(
                model_path='local/checkpoint', vqtoken_mode='centroids')

        detector.assert_called_once_with('local/checkpoint')
        self.assertTrue(model._get_model_overwrite_config()['use_embedded_vision'])

    def test_invalid_frame_count_is_rejected(self):
        runtime = SimpleNamespace(VQTOKEN_CAPABILITIES={
            'modes': ('centroids', ),
            'selection_methods': ('fixed', 'elbow', 'silhouette'),
        })
        with mock.patch.object(
                self.vqtoken, 'import_module',
                return_value=runtime), self.assertRaisesRegex(ValueError, 'max_frames_num'):
            self.vqtoken.LLaVA_OneVision_VQToken(max_frames_num=0)

    def test_missing_runtime_has_actionable_error(self):
        with mock.patch.object(
                self.vqtoken, 'import_module',
                side_effect=ImportError), self.assertRaisesRegex(ImportError,
                                                                 'Hai-chao-Zhang/VQToken'):
            self.vqtoken.require_vqtoken_runtime()

    def test_old_runtime_is_rejected(self):
        runtime = SimpleNamespace(VQTOKEN_CAPABILITIES={'modes': (), 'selection_methods': ()})
        with mock.patch.object(
                self.vqtoken, 'import_module',
                return_value=runtime), self.assertRaisesRegex(ImportError, 'too old'):
            self.vqtoken.require_vqtoken_runtime()

    def test_centroid_only_runtime_is_rejected_for_default_attention(self):
        runtime = SimpleNamespace(VQTOKEN_CAPABILITIES={
            'modes': ('centroids', ),
            'selection_methods': ('fixed', 'elbow', 'silhouette'),
        })
        with mock.patch.object(
                self.vqtoken, 'import_module',
                return_value=runtime), self.assertRaisesRegex(ImportError, 'too old'):
            self.vqtoken.require_vqtoken_runtime()
        with mock.patch.object(self.vqtoken, 'import_module', return_value=runtime):
            self.assertIs(self.vqtoken.require_vqtoken_runtime('centroids'), runtime)

    def test_attention_frame_budget_is_validated_before_loading(self):
        parent_class = self.vqtoken.LLaVA_OneVision
        invalid = [
            {
                'max_frames_num': 33,
                'vqtoken_max_clusters': 32
            },
            {
                'max_frames_num': 13,
                'vqtoken_selection_method': 'silhouette',
                'vqtoken_min_clusters': 12,
                'vqtoken_max_clusters': 32,
            },
        ]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), mock.patch.object(
                    self.vqtoken, 'import_module') as runtime_import, mock.patch.object(
                        parent_class, '__init__',
                        return_value=None) as parent_init, self.assertRaisesRegex(
                            ValueError, 'max_frames_num'):
                self.vqtoken.LLaVA_OneVision_VQToken(**kwargs)
            runtime_import.assert_not_called()
            parent_init.assert_not_called()

    def test_unknown_checkpoint_cannot_silently_use_random_attention(self):
        detector = mock.MagicMock(return_value=False)
        runtime = SimpleNamespace(
            VQTOKEN_CAPABILITIES={
                'modes': ('attention', 'centroids'),
                'selection_methods': ('fixed', 'elbow', 'silhouette'),
            },
            has_released_vq_attention_weights=detector,
        )
        parent_class = self.vqtoken.LLaVA_OneVision
        with mock.patch.object(
                self.vqtoken, 'import_module', return_value=runtime), mock.patch.object(
                    parent_class, '__init__',
                    return_value=None) as parent_init, self.assertRaisesRegex(
                        ValueError, 'released checkpoint'):
            self.vqtoken.LLaVA_OneVision_VQToken(model_path='local/base-only')

        detector.assert_called_once_with('local/base-only')
        parent_init.assert_not_called()

    def test_verified_local_checkpoint_can_use_attention(self):
        attention_detector = mock.MagicMock(return_value=True)
        runtime = SimpleNamespace(
            VQTOKEN_CAPABILITIES={
                'modes': ('attention', 'centroids'),
                'selection_methods': ('fixed', 'elbow', 'silhouette'),
            },
            has_released_vq_attention_weights=attention_detector,
        )
        parent_class = self.vqtoken.LLaVA_OneVision
        with mock.patch.object(
                self.vqtoken, 'import_module', return_value=runtime), mock.patch.object(
                    parent_class, '__init__', return_value=None):
            model = self.vqtoken.LLaVA_OneVision_VQToken(
                model_path='local/released-snapshot', use_embedded_vision=False)

        attention_detector.assert_called_once_with('local/released-snapshot')
        self.assertEqual(model._get_model_overwrite_config()['vqtoken_mode'], 'attention')

    def test_runtime_dependency_is_pinned_to_attention_compatible_commit(self):
        self.assertIn('0314eb9989a7ea843f31bfe0984113529e3f9140', self.vqtoken.RUNTIME_INSTALL)


class TestOneVisionVideoBoundary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.llava = load_llava_module()

    def make_model(self):
        model = self.llava.LLaVA_OneVision.__new__(self.llava.LLaVA_OneVision)
        frames = FakeVideoFrames()
        model.load_video = mock.MagicMock(return_value=(frames, '0.00s,1.00s,2.00s,3.00s', 4.0))
        model.nframe = 32
        model.force_sample = True
        model.DEFAULT_IMAGE_TOKEN = '<image>'
        model.IMAGE_TOKEN_INDEX = -200
        model.image_processor = SimpleNamespace(
            preprocess=lambda *args, **kwargs: {'pixel_values': FakeTensor()})
        model.conv_template = 'qwen_1_5'
        model.conv_templates = {'qwen_1_5': FakeConversation()}
        model.tokenizer = object()
        model.tokenizer_image_token = mock.MagicMock(return_value=FakeTensor())
        model.KeywordStoppingCriteria = lambda *args, **kwargs: object()
        model.SeparatorStyle = SimpleNamespace(TWO='two')
        model.model = SimpleNamespace(generate=mock.MagicMock(return_value=FakeTensor()))
        model.tokenizer = SimpleNamespace(batch_decode=lambda *args, **kwargs: ['answer'])
        return model

    def test_video_generation_uses_one_multimodal_item(self):
        model = self.make_model()
        message = [
            {
                'type': 'video',
                'value': 'demo.mp4'
            },
            {
                'type': 'text',
                'value': 'What happens?'
            },
        ]

        self.assertEqual(model.generate_inner_video(message), 'answer')

        prompt = model.tokenizer_image_token.call_args.args[0]
        generation = model.model.generate.call_args.kwargs
        self.assertIn('and 4 frames are uniformly sampled', prompt)
        self.assertEqual(len(generation['images']), 1)
        self.assertIs(generation['attention_mask'], model.tokenizer_image_token.return_value)
        self.assertEqual(generation['image_sizes'], [(10, 8)])
        self.assertEqual(generation['modalities'], ['video'])

    def test_video_message_routes_without_dataset_name(self):
        model = self.make_model()
        model.generate_inner_video = mock.MagicMock(return_value='video-answer')
        model.generate_inner_image = mock.MagicMock(return_value='image-answer')

        result = model.generate_inner([{'type': 'video', 'value': 'demo.mp4'}], dataset=None)

        self.assertEqual(result, 'video-answer')
        model.generate_inner_video.assert_called_once()
        model.generate_inner_image.assert_not_called()

    def test_explicit_megabench_video_routes_as_video(self):
        model = self.make_model()
        model.generate_inner_video = mock.MagicMock(return_value='video-answer')
        model.generate_inner_image = mock.MagicMock(return_value='image-answer')

        result = model.generate_inner(
            [{
                'type': 'video',
                'value': 'demo.mp4'
            }], dataset='MEGABench')

        self.assertEqual(result, 'video-answer')
        model.generate_inner_video.assert_called_once()
        model.generate_inner_image.assert_not_called()

    def test_parent_loader_merges_only_subclass_overrides(self):
        conversation_module = ModuleType('llava.conversation')
        conversation_module.SeparatorStyle = SimpleNamespace(TWO='two')
        conversation_module.conv_templates = {'qwen_1_5': FakeConversation()}

        mm_utils_module = ModuleType('llava.mm_utils')
        mm_utils_module.KeywordsStoppingCriteria = lambda *args, **kwargs: object()
        mm_utils_module.get_model_name_from_path = lambda path: 'derived-name'
        mm_utils_module.process_images = lambda *args, **kwargs: None
        mm_utils_module.tokenizer_image_token = lambda *args, **kwargs: FakeTensor()

        fake_model = mock.MagicMock()
        fake_model.config = SimpleNamespace()
        loader = mock.MagicMock(return_value=(object(), fake_model, object(), 4096))
        builder_module = ModuleType('llava.model.builder')
        builder_module.load_pretrained_model = loader

        llava_package = ModuleType('llava')
        llava_package.__path__ = []
        llava_model_package = ModuleType('llava.model')
        llava_model_package.__path__ = []
        modules = {
            'llava': llava_package,
            'llava.conversation': conversation_module,
            'llava.mm_utils': mm_utils_module,
            'llava.model': llava_model_package,
            'llava.model.builder': builder_module,
        }

        class CustomOneVision(self.llava.LLaVA_OneVision):

            def __init__(self):
                self.settings = {'use_vqtoken': True}
                super().__init__(model_path='local/checkpoint', model_name='llava_qwen')

            def _get_model_overwrite_config(self):
                return dict(self.settings)

        with warnings.catch_warnings(), mock.patch.dict(sys.modules, modules):
            self.llava.LLaVA_OneVision(model_path='org/llava-model')
            base_overrides = loader.call_args.kwargs['overwrite_config']
            custom = CustomOneVision()
            custom_overrides = loader.call_args.kwargs['overwrite_config']

        self.assertIsNone(base_overrides)
        self.assertEqual(custom_overrides, {'use_vqtoken': True})
        custom_overrides['use_vqtoken'] = False
        self.assertTrue(custom._get_model_overwrite_config()['use_vqtoken'])


if __name__ == '__main__':
    unittest.main()
