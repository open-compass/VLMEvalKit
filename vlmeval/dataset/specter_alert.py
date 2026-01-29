"""Specter Alert Dataset for evaluating safety/compliance detection."""
from ..smp import *
import re
import glob
import json
from typing import Optional, Dict, Any, List


class SpecterAlertDataset:
    """
    Specter Alert Dataset for evaluating safety/compliance detection.

    This dataset REQUIRES processor-wrapped VLMs. The dataset provides pre-extracted
    frames along with config_name and metadata. The VLM wrapper (ProcessorCloudVLM)
    is responsible for:
    1. Instantiating the EventProcessor from config_name using Hydra
    2. Loading detection metadata from local files
    3. Calling processor.process_event() with frames to generate the VLM prompt

    Expected data format:
    - TSV with columns: index, video, config_name, original_question, answer, rule_id
    - Frames in {root}/clips/{sample_id}/frames/frame_{timestamp}.jpg
    - Metadata in {root}/clips/{sample_id}/metadata.json
    - Detections in {root}/clips/{sample_id}/detections/frame_{timestamp}.json

    The dataset evaluates VLM predictions by extracting answers from <answer> tags
    and comparing to ground truth (yes/no).
    """

    MODALITY = 'VIDEO'
    TYPE = 'Video-VQA'

    def __init__(self, dataset='SpecterAlert', nframe=8, fps=-1):
        self.dataset_name = dataset
        self.nframe = nframe
        self.fps = fps

        ret = self.prepare_dataset(dataset)
        assert ret is not None

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)

        if 'index' not in self.data:
            self.data['index'] = np.arange(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return dict(self.data.iloc[idx])

    @classmethod
    def supported_datasets(cls):
        return ['SpecterAlert']

    def prepare_dataset(self, dataset_name='SpecterAlert'):
        """
        Load dataset from local path.
        Expects:
        - TSV file at {LMUDataRoot()}/{dataset_name}.tsv
        - Clips at {LMUDataRoot()}/{dataset_name}/clips/{sample_id}/
          - frames/frame_{timestamp}.jpg
          - detections/frame_{timestamp}.json
          - metadata.json
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
        """Load pre-extracted frame paths for a sample.

        Frames are stored as frame_{timestamp}.jpg where timestamp enables
        tracing back to source video time and correlating across sensors.
        """
        frame_dir = osp.join(self.data_root, 'clips', str(sample_id), 'frames')
        if not osp.exists(frame_dir):
            raise FileNotFoundError(f"Frames directory not found: {frame_dir}")

        frame_paths = sorted(glob.glob(osp.join(frame_dir, 'frame_*.jpg')))
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in: {frame_dir}")

        return frame_paths

    def _load_detection_paths(self, sample_id: str) -> List[str]:
        """Load local detection file paths for a sample.

        Detections are stored as frame_{timestamp}.json, matching the
        corresponding frame file (frame_{timestamp}.jpg).
        """
        detections_dir = osp.join(self.data_root, 'clips', str(sample_id), 'detections')
        if not osp.exists(detections_dir):
            return []

        detection_paths = sorted(glob.glob(osp.join(detections_dir, 'frame_*.json')))
        return detection_paths

    def build_prompt(self, line, video_llm=False):
        """Build prompt for processor-wrapped inference.

        This method builds a message containing pre-extracted frames and metadata
        for ProcessorCloudVLM wrapper. The wrapper is responsible for:
        1. Instantiating EventProcessor from config_name using Hydra
        2. Loading detections from local files
        3. Calling processor.process_event() with frames to generate VLM prompt

        Args:
            line: Row from TSV or index
            video_llm: Not used (frames are pre-extracted)

        Returns:
            List of message dicts with types:
            - 'image': Frame file paths
            - 'config_name': EventProcessor config name for Hydra instantiation
            - 'original_question': Fallback prompt text if config_name is empty
            - 'detection_paths': Local paths to detection metadata JSONs
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        sample_id = line['video']
        config_name = line.get('config_name', '')
        original_question = line.get('original_question', '')

        # Load pre-extracted frames
        frame_paths = self._load_frame_paths(sample_id)

        # Load local detection paths
        detection_paths = self._load_detection_paths(sample_id)

        # Build message for ProcessorCloudVLM wrapper
        message = [dict(type='image', value=frame) for frame in frame_paths]
        message.append(dict(type='config_name', value=config_name))
        message.append(dict(type='original_question', value=original_question))
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
