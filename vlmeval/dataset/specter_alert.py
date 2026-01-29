"""Specter Alert VQA Dataset for evaluating safety/compliance detection."""
from ..smp import *
from .video_base import VideoBaseDataset
import re
import glob
import json
from typing import Optional, Dict, Any, List


class SpecterAlertDataset(VideoBaseDataset):
    """
    Specter Alert VQA Dataset for evaluating safety/compliance detection.

    This dataset REQUIRES processor-wrapped VLMs. The dataset provides pre-extracted
    frames along with config_name and metadata. The VLM wrapper (SpecterProcessorVLM)
    is responsible for:
    1. Instantiating the EventProcessor from config_name using Hydra
    2. Fetching detection metadata from S3 if needed (for BoxCropEventProcessor)
    3. Calling processor.process_event() with frames to generate the VLM prompt

    Expected data format:
    - TSV with columns: index, video, context, config_name, original_question, answer, rule_id
    - Frames in {root}/frames/{video}/frame_NNNN.jpg
    - Metadata in {root}/clips/{video}/metadata.json (includes detection_metadata_paths)

    Columns:
    - video: Sample ID (e.g., "galaxy_1043_1769618268000")
    - context: Rule-specific context string
    - config_name: EventProcessor config name (e.g., "man_down_event_processor")
    - original_question: The rendered prompt template_text from VLM inference
    - answer: Ground truth (yes/no)
    - rule_id: Rule UUID

    The dataset evaluates VLM predictions by extracting answers from <answer> tags
    and comparing to ground truth (yes/no).
    """

    TYPE = 'Video-VQA'

    def __init__(self, dataset='SpecterAlert', nframe=8, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['SpecterAlert']

    def prepare_dataset(self, dataset_name='SpecterAlert'):
        """
        Load dataset from local path.
        Expects:
        - TSV file at {LMUDataRoot()}/{dataset_name}.tsv
        - Frames at {LMUDataRoot()}/{dataset_name}/frames/{sample_id}/
        - Metadata at {LMUDataRoot()}/{dataset_name}/clips/{sample_id}/metadata.json
        """
        lmu_root = LMUDataRoot()
        data_file = osp.join(lmu_root, f'{dataset_name}.tsv')
        dataset_root = osp.join(lmu_root, dataset_name)

        if not osp.exists(data_file):
            raise FileNotFoundError(f"Dataset TSV not found: {data_file}")
        if not osp.exists(dataset_root):
            raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

        return dict(root=dataset_root, data_file=data_file)

    def _load_frame_paths(self, sample_id: str) -> List[str]:
        """Load pre-extracted frame paths for a sample."""
        frame_dir = osp.join(self.data_root, 'frames', str(sample_id))
        if not osp.exists(frame_dir):
            raise FileNotFoundError(f"Frames directory not found: {frame_dir}")

        frame_paths = sorted(glob.glob(osp.join(frame_dir, 'frame_*.jpg')))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in: {frame_dir}")

        return frame_paths

    def _load_detection_paths(self, sample_id: str) -> List[str]:
        """Load detection metadata paths from sample metadata."""
        metadata_path = osp.join(self.data_root, 'clips', str(sample_id), 'metadata.json')
        if not osp.exists(metadata_path):
            return []

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
            return metadata.get('detection_metadata_paths', [])
        except (json.JSONDecodeError, IOError):
            return []

    def build_prompt(self, line, video_llm=False):
        """Build prompt for processor-wrapped inference.

        This method builds a message containing pre-extracted frames and metadata
        for SpecterProcessorVLM wrapper. The wrapper is responsible for:
        1. Instantiating EventProcessor from config_name using Hydra
        2. Fetching detections from S3 if BoxCropEventProcessor
        3. Calling processor.process_event() with frames to generate VLM prompt

        The message does NOT include a direct text prompt - prompt formatting is
        handled by the EventProcessor at inference time.

        Args:
            line: Row from TSV or index
            video_llm: Not used (frames are pre-extracted)

        Returns:
            List of message dicts with types:
            - 'image': Frame file paths
            - 'config_name': EventProcessor config name for Hydra instantiation
            - 'context': Context string for prompt_kwargs
            - 'original_question': Rendered prompt (for debugging/fallback)
            - 'rule_id': Rule UUID
            - 'detection_paths': S3 paths to detection metadata JSONs
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        sample_id = line['video']
        context = line.get('context', '')
        config_name = line.get('config_name', '')
        original_question = line.get('original_question', '')
        rule_id = str(line.get('rule_id', 'default')) if 'rule_id' in line else 'default'

        # Load pre-extracted frames
        frame_paths = self._load_frame_paths(sample_id)

        # Load detection metadata paths
        detection_paths = self._load_detection_paths(sample_id)

        # Build message for SpecterProcessorVLM wrapper
        message = [dict(type='image', value=frame) for frame in frame_paths]
        message.append(dict(type='config_name', value=config_name))
        message.append(dict(type='context', value=context))
        message.append(dict(type='original_question', value=original_question))
        message.append(dict(type='rule_id', value=rule_id))
        message.append(dict(type='detection_paths', value=detection_paths))

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
