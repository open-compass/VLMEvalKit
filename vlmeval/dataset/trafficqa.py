# flake8: noqa
"""
TrafficQA Dataset Implementation for VLMEvalKit

TrafficQA is a video question answering benchmark for traffic scenes from CVPR 2021.
It features 6 reasoning tasks: U (Basic Understanding), A (Attribution),
F (Event Forecasting), R (Reverse Reasoning), C (Counterfactual Inference), I (Introspection).

*** MANUAL DOWNLOAD REQUIRED ***
Dataset must be downloaded manually from: https://sutdcv.github.io/SUTD-TrafficQA/

Key Implementation Details:
- Uses R2 annotation format with q_type field for task categorization
- Handles variable option positioning (options can be in any positions)
- Answer index always points to valid option in original position
- Video files are in compressed_videos/ directory

Paper: https://arxiv.org/abs/2103.15247
Official Website: https://sutdcv.github.io/SUTD-TrafficQA/
Repository: https://github.com/SUTD-TaCheng/TrafficQA
"""

import os
import json
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'


# Reasoning task type mappings from TrafficQA paper
Q_TYPE_NAMES = {
    'U': 'Basic Understanding',
    'A': 'Attribution',
    'F': 'Event Forecasting',
    'R': 'Reverse Reasoning',
    'C': 'Counterfactual Inference',
    'I': 'Introspection'
}

# Mapping from q_type to answer key format
Q_TYPE_TO_ANSWER_KEY = {
    'U': 'basic_understanding',
    'A': 'attribution',
    'F': 'event_forecasting',
    'R': 'reverse_reasoning',
    'C': 'counterfactual_inference',
    'I': 'introspection'
}


class TrafficQA(VideoBaseDataset):
    """
    TrafficQA Dataset Implementation

    *** MANUAL DOWNLOAD REQUIRED ***
    This dataset requires manual download due to licensing restrictions.
    Please download from: https://sutdcv.github.io/SUTD-TrafficQA/

    After downloading, set the TRAFFICQA_DATA_PATH environment variable to point
    to your dataset directory, or ensure it's at DATASET_PATH.

    Args:
        dataset: Dataset name (default: 'TrafficQA')
        split: Dataset split - 'test', 'train', or 'all' (default: 'test')
        nframe: Number of frames to sample (mutually exclusive with fps)
        fps: Frames per second for sampling (mutually exclusive with nframe)

    Dataset Statistics (R2 annotations):
    - Total: 62,533 QA pairs from 10,080 videos
    - Test: 6,075 QAs (recommended for evaluation)
    - Train: 56,459 QAs
    - 99.5% of test QAs come from videos also in training set (by design)
    """

    MD5 = 'a3f2c1d4e5b6a7f8'  # Placeholder - will be set during prepare_dataset

    FRAMES_TMPL_SYS = """
You will receive {} distinct frames sampled from a traffic video.
Based on these frames, answer the following multiple-choice question about the traffic scene.
"""

    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive several frames from a traffic video.
Based on these frames, answer the following multiple-choice question about the traffic scene.
"""

    QUESTION_TMPL = """
Question: {}
{}

Answer with the option letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'Video-MCQ'

    # Local dataset path (can be overridden via environment variable)
    DATASET_PATH = '/storage/disk3/datasets/SUTD-TrafficQA'

    def __init__(self, dataset='TrafficQA', split='test', nframe=0, fps=-1):
        self.split = split
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['TrafficQA']

    def prepare_dataset(self, dataset_name='TrafficQA'):
        """
        Prepare TrafficQA dataset from local JSONL files.

        The dataset should be located at DATASET_PATH with the following structure:
        /storage/disk3/datasets/SUTD-TrafficQA/
        ├── annotations/
        │   ├── R2_all.jsonl (62,533 QA pairs)
        │   ├── R2_train.jsonl (56,459 QA pairs)
        │   └── R2_test.jsonl (6,074 QA pairs - recommended for evaluation)
        └── compressed_videos/
            └── 10,080 .mp4 files

        Returns:
            dict with 'root' (video directory) and 'data_file' (TSV file path)
        """
        # Check for environment variable override
        data_path = os.environ.get('TRAFFICQA_DATA_PATH', self.DATASET_PATH)

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"TrafficQA dataset not found at {data_path}. "
                f"Please set TRAFFICQA_DATA_PATH environment variable or ensure dataset is at {self.DATASET_PATH}. "
                f"Download from: https://sutdcv.github.io/SUTD-TrafficQA/"
            )

        # Determine which annotation file to use based on split
        if self.split == 'test':
            jsonl_file = 'R2_test.jsonl'
        elif self.split == 'train':
            jsonl_file = 'R2_train.jsonl'
        else:  # 'all'
            jsonl_file = 'R2_all.jsonl'

        jsonl_path = os.path.join(data_path, 'annotations', jsonl_file)

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"Annotation file not found: {jsonl_path}. "
                f"Please ensure R2 annotations are downloaded."
            )

        # Convert JSONL to TSV format expected by VLMEvalKit
        tsv_filename = f'{dataset_name}_{self.split}.tsv'
        tsv_path = os.path.join(data_path, tsv_filename)

        # Generate TSV if it doesn't exist or needs update
        if not os.path.exists(tsv_path):
            self._generate_tsv_from_jsonl(jsonl_path, tsv_path, data_path)
        else:
            # Check if TSV is older than JSONL
            jsonl_mtime = os.path.getmtime(jsonl_path)
            tsv_mtime = os.path.getmtime(tsv_path)
            if jsonl_mtime > tsv_mtime:
                self._generate_tsv_from_jsonl(jsonl_path, tsv_path, data_path)

        video_root = os.path.join(data_path, 'compressed_videos')

        if not os.path.exists(video_root):
            raise FileNotFoundError(
                f"Video directory not found: {video_root}. "
                f"Please ensure compressed_videos/ directory exists."
            )

        return dict(root=video_root, data_file=tsv_path)

    def _generate_tsv_from_jsonl(self, jsonl_path, tsv_path, data_path):
        """
        Convert TrafficQA JSONL format to VLMEvalKit TSV format.

        TrafficQA JSONL format (R2):
        [record_id, vid_id, vid_filename, perspective, q_body, q_type,
         option0, option1, option2, option3, answer]

        Note: Each line is a JSON array, not a JSON object.
        Note: Options can be in any positions (variable option positioning).
        Note: Answer index is always valid (points to non-empty option).
        """
        data_rows = []
        video_files = set()

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            # First line is header
            header = json.loads(f.readline().strip())
            header = [h.lower() for h in header]

            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                # Parse JSON array
                record = json.loads(line)

                # Create dict from header
                row_dict = dict(zip(header, record))

                # Extract video filename without extension
                vid_filename = row_dict['vid_filename']
                if vid_filename.endswith('.mp4'):
                    video_name = vid_filename[:-4]
                else:
                    video_name = vid_filename

                # Build video path (relative to video_root)
                video_path_abs = os.path.join(data_path, 'compressed_videos', vid_filename)
                video_path_rel = vid_filename  # Relative path (just filename)

                # Verify video exists
                if not os.path.exists(video_path_abs):
                    print(f"Warning: Video not found: {video_path_abs}")
                    continue

                video_files.add(video_name)

                # Build question with options
                # Handle variable option positioning
                options = []
                option_letters = []
                for opt_idx in range(4):
                    opt_key = f'option{opt_idx}'
                    opt_value = row_dict.get(opt_key, '').strip()
                    options.append(opt_value if opt_value else '')

                # Create option text for prompt
                option_text_parts = []
                for opt_idx in range(4):
                    opt_letter = chr(65 + opt_idx)  # A, B, C, D
                    opt_value = options[opt_idx]
                    if opt_value:
                        option_text_parts.append(f"{opt_letter}. {opt_value}")

                option_text = '\n'.join(option_text_parts)

                # Map answer index to letter
                answer_idx = row_dict['answer']
                answer_letter = chr(65 + answer_idx)  # A, B, C, D

                # Create TSV row
                tsv_row = {
                    'index': line_num - 1,  # 0-based index
                    'video': video_name,
                    'video_path': video_path_rel,  # Just filename (relative to video_root)
                    'question': row_dict['q_body'],
                    'options': option_text,
                    'answer': answer_letter,
                    'q_type': row_dict['q_type'],
                    'record_id': row_dict['record_id'],
                    'vid_id': row_dict['vid_id'],
                    'perspective': row_dict['perspective']
                }

                data_rows.append(tsv_row)

        # Create DataFrame and save as TSV
        df = pd.DataFrame(data_rows)
        df.to_csv(tsv_path, sep='\t', index=False)

        print(f"Generated TSV file: {tsv_path}")
        print(f"Total questions: {len(df)}")
        print(f"Unique videos: {len(video_files)}")

        # Print q_type distribution
        if 'q_type' in df.columns:
            q_type_counts = df['q_type'].value_counts()
            print(f"Question type distribution:")
            for q_type, count in q_type_counts.items():
                q_type_name = Q_TYPE_NAMES.get(q_type, q_type)
                print(f"  {q_type} ({q_type_name}): {count} ({count/len(df)*100:.1f}%)")

    def save_video_frames(self, line):
        """Save video frames for processing."""
        video = line['video']
        vid_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))

        import decord
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # Not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))
        else:
            raise ValueError("Either nframe or fps must be set")

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            lock_path = osp.splitext(vid_path)[0] + '.lock'
            with portalocker.Lock(lock_path, 'w', timeout=30):
                if not np.all([osp.exists(p) for p in frame_paths]):
                    images = [vid[i].asnumpy() for i in indices]
                    images = [Image.fromarray(arr) for arr in images]
                    for im, pth in zip(images, frame_paths):
                        if not osp.exists(pth):
                            im.save(pth)

        return frame_paths

    def build_prompt(self, line, video_llm=False):
        """
        Build prompt for TrafficQA question.

        Args:
            line: Data row (can be int index or dict)
            video_llm: If True, use video path instead of frames

        Returns:
            Message list with frames/video and text prompt
        """
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        if video_llm:
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS_4VIDEO_LLM)]
            video_path = os.path.normpath(os.path.join(self.data_root, line['video_path']))
            message.append(dict(type='video', value=video_path))
        else:
            frame_paths = self.save_video_frames(line)
            message = [dict(type='text', value=self.FRAMES_TMPL_SYS.format(len(frame_paths)))]
            for frame_path in frame_paths:
                message.append(dict(type='image', value=frame_path))

        # Add question and options
        question_prompt = self.QUESTION_TMPL.format(
            line['question'],
            line['options']
        )
        message.append(dict(type='text', value=question_prompt))

        return message

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """
        Evaluate predictions on TrafficQA dataset.

        Computes:
        - Overall accuracy
        - Per-reasoning-task accuracy (U, A, F, R, C, I)

        Args:
            eval_file: Path to evaluation file with predictions
            **judge_kwargs: Additional arguments for judge

        Returns:
            Dictionary with evaluation results
        """
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], \
            'data file should be a supported format (xlsx/json/tsv) file'

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')

            if model == 'exact_matching':
                model = None
            else:
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None

            data = load(eval_file)

            # Process each prediction
            for idx in data['index']:
                ans = str(data.loc[data['index'] == idx, 'answer'].values[0]).strip().upper()
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if FAIL_MSG in pred or pd.isna(pred):
                    data.loc[idx, 'score'] = -1
                else:
                    # Extract answer letter from prediction
                    pred_clean = pred.strip().upper()

                    # Try to extract single letter (A, B, C, D)
                    import re
                    match = re.search(r'[A-D]', pred_clean)
                    if match:
                        pred_letter = match.group(0)
                        data.loc[idx, 'score'] = int(pred_letter == ans)
                    else:
                        # No valid letter found
                        data.loc[idx, 'score'] = 0

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, '
                f'failed to obtain prediction for {len(data) - len(data[~pd.isna(data["prediction"])])} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        # Calculate overall and per-task metrics
        data = load(score_file)

        # Overall accuracy
        valid_data = data[data['score'] != -1]
        overall_acc = valid_data['score'].sum() / len(valid_data) if len(valid_data) > 0 else 0

        results = {
            'overall': {
                'acc': overall_acc,
                'total': len(data),
                'valid': len(valid_data),
                'correct': int(valid_data['score'].sum()) if len(valid_data) > 0 else 0
            }
        }

        # Per-reasoning-task accuracy
        if 'q_type' in data.columns:
            for q_type in Q_TYPE_NAMES.keys():
                q_type_data = data[data['q_type'] == q_type]
                q_type_valid = q_type_data[q_type_data['score'] != -1]

                if len(q_type_valid) > 0:
                    q_type_acc = q_type_valid['score'].sum() / len(q_type_valid)
                    answer_key = Q_TYPE_TO_ANSWER_KEY[q_type]
                    results[answer_key] = {
                        'acc': q_type_acc,
                        'total': len(q_type_data),
                        'valid': len(q_type_valid),
                        'correct': int(q_type_valid['score'].sum())
                    }

        # Save results
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        dump(results, tgt_file)

        # Print summary
        print(f"\nTrafficQA Evaluation Results:")
        print(f"Overall Accuracy: {overall_acc:.4f} ({int(valid_data['score'].sum()) if len(valid_data) > 0 else 0}/{len(valid_data)})")

        if 'q_type' in data.columns:
            print(f"\nPer-Reasoning-Task Accuracy:")
            for q_type in Q_TYPE_NAMES.keys():
                answer_key = Q_TYPE_TO_ANSWER_KEY[q_type]
                if answer_key in results:
                    r = results[answer_key]
                    print(f"  {q_type} ({Q_TYPE_NAMES[q_type]}): {r['acc']:.4f} ({r['correct']}/{r['valid']})")

        return results
