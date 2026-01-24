"""Specter Alert VQA Dataset for evaluating PPE compliance detection."""
from ..smp import *
from .video_base import VideoBaseDataset
import re
from typing import Optional, Dict, Any


class SpecterAlertDataset(VideoBaseDataset):
    """
    Specter Alert VQA Dataset for evaluating PPE compliance detection.

    Expected data format:
    - TSV with columns: index, video, context, question, answer, rule_id, rule_name
    - Videos in {root}/video/{video}.mp4

    The dataset evaluates VLM predictions by extracting answers from <answer> tags
    and comparing to ground truth (yes/no).

    Optionally loads a processor_config YAML that maps rule_id -> EventProcessor config
    for per-rule preprocessing.
    """

    TYPE = 'Video-VQA'

    def __init__(self, dataset='SpecterAlert', nframe=8, fps=-1, processor_config: Optional[str] = None):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

        # Load processor config if provided
        self.processor_config_path = processor_config
        self.processor_config = None
        self.rule_processors: Dict[str, Any] = {}
        self.default_processor = None

        if processor_config:
            self._load_processor_config(processor_config)

    def _load_processor_config(self, config_path: str):
        """Load rule -> processor mapping from YAML."""
        import yaml

        if not osp.exists(config_path):
            print(f"Warning: processor config not found: {config_path}")
            return

        with open(config_path) as f:
            self.processor_config = yaml.safe_load(f)

        default_cfg = self.processor_config.get('default', {})
        rules_cfg = self.processor_config.get('rules', {})

        # Build default processor
        if default_cfg:
            self.default_processor = self._build_processor(default_cfg)

        # Build rule-specific processors
        for rule_id, rule_cfg in rules_cfg.items():
            self.rule_processors[str(rule_id)] = self._build_processor(rule_cfg)

        print(f"Loaded processor config: default + {len(self.rule_processors)} rule-specific processors")

    def _build_processor(self, config: dict):
        """Build EventProcessor from config dict.

        Note: prompt_kwargs (context/question) are NOT set here - they come from
        the TSV and are applied by SpecterProcessorWrapper at inference time.
        """
        from specter_prompts.event_processor import (
            EventProcessor, FallbackEventProcessor, BoxCropEventProcessor
        )

        processor_type = config.get('type', 'EventProcessor')
        common_kwargs = {
            'system_prompt': config.get('system_prompt'),
            'short_side': config.get('short_side', 672),
            'jpeg_quality': config.get('jpeg_quality', 80),
            'max_frames': config.get('max_frames', 14),
        }

        if processor_type == 'FallbackEventProcessor':
            return FallbackEventProcessor(
                prompt=config['prompt'],
                **common_kwargs
            )
        elif processor_type == 'EventProcessor':
            return EventProcessor(
                prompt_template=config['prompt_template'],
                prompt_kwargs={},  # Will be set by SpecterProcessorWrapper from TSV
                prompt_post_processor_kwargs=config.get('prompt_post_processor_kwargs', {}),
                **common_kwargs
            )
        elif processor_type == 'BoxCropEventProcessor':
            frame_processor = EventProcessor(
                prompt_template=config['prompt_template'],
                prompt_kwargs={},  # Will be set by SpecterProcessorWrapper from TSV
                prompt_post_processor_kwargs=config.get('prompt_post_processor_kwargs', {}),
                **common_kwargs
            )
            return BoxCropEventProcessor(
                frame_event_processor=frame_processor,
                **config.get('box_crop_kwargs', {})
            )
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")

    def get_processor_for_rule(self, rule_id: str):
        """Get the EventProcessor for a given rule_id."""
        return self.rule_processors.get(str(rule_id), self.default_processor)

    @classmethod
    def supported_datasets(cls):
        return ['SpecterAlert']

    def prepare_dataset(self, dataset_name='SpecterAlert'):
        """
        Load dataset from local path.
        Expects:
        - TSV file at {LMUDataRoot()}/{dataset_name}.tsv
        - Videos at {LMUDataRoot()}/{dataset_name}/video/{video}.mp4
        """
        lmu_root = LMUDataRoot()
        data_file = osp.join(lmu_root, f'{dataset_name}.tsv')
        video_root = osp.join(lmu_root, dataset_name)

        if not osp.exists(data_file):
            raise FileNotFoundError(f"Dataset TSV not found: {data_file}")
        if not osp.exists(video_root):
            raise FileNotFoundError(f"Video directory not found: {video_root}")

        return dict(root=video_root, data_file=data_file)

    def save_video_frames(self, video):
        """Override to handle video/ subdirectory structure."""
        import decord

        # Videos are in video/ subdirectory
        vid_path = osp.join(self.data_root, 'video', str(video) + '.mp4')
        if not osp.exists(vid_path):
            raise FileNotFoundError(f"Video not found: {vid_path}")

        vid = decord.VideoReader(vid_path)

        if self.nframe > 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(str(video))
        else:
            raise ValueError("fps mode not supported, use nframe")

        flag = np.all([osp.exists(p) for p in frame_paths])
        if not flag:
            lock_path = osp.join(self.frame_root, str(video) + '.lock')
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths

    def build_prompt(self, line, video_llm=False):
        """Build prompt for inference."""
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video = line['video']
        question = line['question']
        context = line.get('context', '')
        rule_id = str(line.get('rule_id', 'default')) if 'rule_id' in line else 'default'

        # Get processor for this rule (if processor config was loaded)
        processor = self.get_processor_for_rule(rule_id)

        if video_llm:
            # Direct video input for video-native models
            video_path = osp.join(self.data_root, 'video', str(video) + '.mp4')
            message = [
                dict(type='video', value=video_path),
                dict(type='text', value=question)
            ]
        else:
            # Frame-based input for image-based models
            frame_paths = self.save_video_frames(video)
            message = []
            for frame in frame_paths:
                message.append(dict(type='image', value=frame))
            message.append(dict(type='text', value=question))

        # Include processor, rule_id, context, question for SpecterProcessorWrapper
        if processor is not None:
            message.append(dict(type='processor', value=processor))
        message.append(dict(type='rule_id', value=rule_id))
        message.append(dict(type='context', value=context))
        message.append(dict(type='question', value=question))

        return message

    @staticmethod
    def extract_answer(text):
        """Extract yes/no/unknown from model response."""
        if not text:
            return 'unknown'

        text_lower = text.lower().strip()

        # First try: look for <answer> tags
        match = re.search(r'<answer>\s*(yes|no|unknown)\s*</answer>', text_lower)
        if match:
            return match.group(1)

        # Second try: strip punctuation and check if response is just yes/no
        text_clean = re.sub(r'[^\w\s]', '', text_lower).strip()
        if text_clean in ['yes', 'no', 'unknown']:
            return text_clean

        # Third try: look for yes/no anywhere in a short response (< 20 chars)
        if len(text_clean) < 20:
            if 'yes' in text_clean:
                return 'yes'
            elif 'no' in text_clean:
                return 'no'

        # Fourth try: check last word after stripping punctuation
        words = text_clean.split()
        if words:
            last_word = words[-1]
            if last_word in ['yes', 'no', 'unknown']:
                return last_word

        return 'unknown'

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate predictions against ground truth using simple yes/no matching.

        Extracts answers from <answer> tags in predictions and compares to
        the ground truth answer column. Unknown predictions are treated as "no".
        """
        from ..smp import load, dump
        from ..smp.file import get_intermediate_file_path

        data = load(eval_file)

        # Confusion matrix counts
        tp = 0  # predicted yes, actual yes
        fp = 0  # predicted yes, actual no
        tn = 0  # predicted no, actual no
        fn = 0  # predicted no, actual yes

        for idx, row in data.iterrows():
            pred_text = str(row.get('prediction', ''))
            gt = str(row.get('answer', '')).lower().strip()

            pred = cls.extract_answer(pred_text)

            # Treat "unknown" as "no"
            if pred == 'unknown':
                pred = 'no'

            # Only count rows with valid ground truth (yes/no)
            if gt in ['yes', 'no']:
                if pred == 'yes' and gt == 'yes':
                    tp += 1
                elif pred == 'yes' and gt == 'no':
                    fp += 1
                elif pred == 'no' and gt == 'no':
                    tn += 1
                elif pred == 'no' and gt == 'yes':
                    fn += 1

            # Store extracted prediction back for analysis
            data.at[idx, 'extracted_pred'] = pred
            data.at[idx, 'hit'] = int(pred == gt) if gt in ['yes', 'no'] else -1

        total = tp + fp + tn + fn
        correct = tp + tn
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        score_dict = {
            'accuracy': round(accuracy * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'f1': round(f1 * 100, 2),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total': total
        }

        # Save detailed results
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(score_dict, score_file)

        # Also save the detailed data with extracted predictions
        detail_file = get_intermediate_file_path(eval_file, '_detail')
        dump(data, detail_file)

        return score_dict
