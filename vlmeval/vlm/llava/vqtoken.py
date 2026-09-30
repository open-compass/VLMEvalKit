"""VLMEvalKit adapter for the public learned VQToken release."""

from importlib import import_module

from .llava import LLaVA_OneVision

SUPPORTED_SELECTION_METHODS = frozenset({'fixed', 'elbow', 'silhouette'})
SUPPORTED_VQTOKEN_MODES = frozenset({'attention', 'centroids'})
RELEASED_MODEL_PATH = 'haichaozhang/VQ-Token-llava-ov-0.5b'
RUNTIME_INSTALL = ('llava[runtime] @ git+https://github.com/Hai-chao-Zhang/'
                   'VQToken.git@0314eb9989a7ea843f31bfe0984113529e3f9140')


def validate_cluster_config(selection_method, min_clusters, max_clusters):
    """Validate the public VQToken cluster-budget configuration."""
    if selection_method not in SUPPORTED_SELECTION_METHODS:
        raise ValueError('vqtoken_selection_method must be one of '
                         f'{sorted(SUPPORTED_SELECTION_METHODS)}, got {selection_method!r}')
    if isinstance(min_clusters, bool) or not isinstance(min_clusters, int) or min_clusters < 1:
        raise ValueError('vqtoken_min_clusters must be a positive integer')
    if isinstance(max_clusters, bool) or not isinstance(max_clusters, int) or max_clusters < 1:
        raise ValueError('vqtoken_max_clusters must be a positive integer')
    if selection_method != 'fixed' and max_clusters < min_clusters:
        raise ValueError(
            'adaptive VQToken requires max_clusters to be greater than or equal to min_clusters')


def validate_attention_frame_budget(selection_method, min_clusters, max_clusters, max_frames_num):
    """Fail early when the learned attention path could receive more frames than K."""
    smallest_budget = max_clusters if selection_method == 'fixed' else min_clusters
    if max_frames_num > smallest_budget:
        raise ValueError('learned VQ-Attention requires max_frames_num to be no greater than '
                         f'the smallest possible token budget ({smallest_budget})')


def require_vqtoken_runtime(required_mode='attention'):
    """Load the optional VQToken runtime and verify its public capabilities."""
    if required_mode not in SUPPORTED_VQTOKEN_MODES:
        raise ValueError('vqtoken_mode must be one of '
                         f'{sorted(SUPPORTED_VQTOKEN_MODES)}, got {required_mode!r}')
    try:
        runtime = import_module('VQToken')
    except ImportError as err:
        message = ('VQToken requires its public LLaVA runtime. Install with: '
                   f'pip install "{RUNTIME_INSTALL}"')
        raise ImportError(message) from err

    capabilities = getattr(runtime, 'VQTOKEN_CAPABILITIES', {})
    modes = set(capabilities.get('modes', ()))
    methods = set(capabilities.get('selection_methods', ()))
    if required_mode not in modes or not SUPPORTED_SELECTION_METHODS.issubset(methods):
        raise ImportError('The installed VQToken runtime is too old for VLMEvalKit. '
                          f'Upgrade with: pip install --upgrade "{RUNTIME_INSTALL}"')
    return runtime


class LLaVA_OneVision_VQToken(LLaVA_OneVision):
    """Run learned VQ-Attention with the released OneVision checkpoint."""

    def __init__(
        self,
        model_path=RELEASED_MODEL_PATH,
        vqtoken_mode='attention',
        vqtoken_selection_method='fixed',
        vqtoken_min_clusters=12,
        vqtoken_max_clusters=32,
        max_frames_num=32,
        use_embedded_vision=None,
        **kwargs,
    ):
        validate_cluster_config(
            vqtoken_selection_method,
            vqtoken_min_clusters,
            vqtoken_max_clusters,
        )
        if isinstance(max_frames_num,
                      bool) or not isinstance(max_frames_num, int) or max_frames_num < 1:
            raise ValueError('max_frames_num must be a positive integer')
        if vqtoken_mode not in SUPPORTED_VQTOKEN_MODES:
            raise ValueError('vqtoken_mode must be one of '
                             f'{sorted(SUPPORTED_VQTOKEN_MODES)}, got {vqtoken_mode!r}')
        if vqtoken_mode == 'attention':
            validate_attention_frame_budget(
                vqtoken_selection_method,
                vqtoken_min_clusters,
                vqtoken_max_clusters,
                max_frames_num,
            )
        runtime = require_vqtoken_runtime(vqtoken_mode)
        if vqtoken_mode == 'attention':
            is_released_checkpoint = model_path.rstrip('/') == RELEASED_MODEL_PATH
            detector = getattr(runtime, 'has_released_vq_attention_weights', None)
            if not is_released_checkpoint and callable(detector):
                is_released_checkpoint = detector(model_path)
            if not is_released_checkpoint:
                raise ValueError('learned VQ-Attention requires the released checkpoint '
                                 f'{RELEASED_MODEL_PATH!r} or a local checkpoint with its exact '
                                 'VQ-Attention weights; use vqtoken_mode="centroids" only for '
                                 'an intentional centroid ablation')
        if use_embedded_vision is None:
            use_embedded_vision = model_path.rstrip('/') in {
                RELEASED_MODEL_PATH,
                'lmms-lab/llava-onevision-qwen2-0.5b-ov',
            }
            detector = getattr(runtime, 'has_embedded_vision_weights', None)
            if not use_embedded_vision and callable(detector):
                use_embedded_vision = detector(model_path)

        self._vqtoken_overwrite_config = {
            'use_vqtoken': True,
            'vqtoken_mode': vqtoken_mode,
            'vqtoken_selection_method': vqtoken_selection_method,
            'vqtoken_min_clusters': vqtoken_min_clusters,
            'vqtoken_max_clusters': vqtoken_max_clusters,
            'use_embedded_vision': use_embedded_vision,
            'mm_spatial_pool_stride': 2,
            'mm_spatial_pool_mode': 'bilinear',
        }
        super().__init__(
            model_path=model_path,
            model_name='llava_qwen',
            **kwargs,
        )
        self.nframe = max_frames_num
        # The released evaluator uniformly samples exactly ``nframe`` frames,
        # but does not add LLaVA-Video's timestamp instruction to the prompt.
        self.force_sample = False

    def _get_model_overwrite_config(self):
        """Return a copy so the parent cannot mutate the adapter's settings."""
        return dict(self._vqtoken_overwrite_config)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        """Match the released VQToken evaluator's uniform frame sampling."""
        return super().load_video(video_path, max_frames_num, fps, True)
