# flake8: noqa
"""VideoEval-Pro dataset integration.

The upstream benchmark runs each sample twice: once as a short-answer question
and once as a multiple-choice question. VLMEvalKit evaluates one prediction
column at a time, so this module exposes matching OpenEnded and MCQ variants
while sharing the upstream parquet metadata and video layout.
"""

from __future__ import annotations
import ast
import os
import os.path as osp
import tarfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

from vlmeval.smp import LMUDataRoot, dump, get_file_extension, get_intermediate_file_path, load
from .utils import DEBUG_MESSAGE, build_judge
from .video_base import VideoBaseDataset

VIDEOEVAL_PRO_REPO = 'TIGER-Lab/VideoEval-Pro'
VIDEOEVAL_PRO_JUDGE_MODEL = 'gpt-4o-0806'
VIDEOEVAL_PRO_FAIL_MSG = 'Failed to obtain answer via API.'

SHORT_ANSWER_SUFFIX = ' Keep the answer short and concise.'
MCQ_INSTRUCTION = (
    'Select the best answer to the following multiple-choice question based on the video. '
    'Respond with only the letter (A, B, C, or D) of the correct option, with no text around it.'
)


def _options_to_list(options) -> list[str]:
    if isinstance(options, np.ndarray):
        options = options.tolist()
    elif isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            options = [options]
    if options is None or (isinstance(options, float) and pd.isna(options)):
        return []
    return [str(option) for option in options]


def short_answer_prompt(question: str) -> str:
    return f'{question}{SHORT_ANSWER_SUFFIX}'


def multiple_choice_prompt(question: str, options) -> str:
    return '\n'.join([MCQ_INSTRUCTION, question, ' '.join(_options_to_list(options))])


def build_official_inference_inputs(item: dict, video_path: str, *, using_frames: bool = False,
                                   frame_paths: list[str] | None = None,
                                   num_frames: int = 32) -> tuple[list[dict], list[dict]]:
    """Return the two upstream input lists, preserving media-first ordering."""
    media_type = 'frames' if using_frames else 'video'
    media_content = frame_paths if using_frames else video_path
    media = {
        'type': media_type,
        'content': media_content,
        'metadata': {
            'video_num_frames': num_frames,
            'video_sample_type': 'rand',
            'img_shortest_edge': 256,
            'img_longest_edge': 480,
            'max_img_seq_len': 16000,
            'do_resize': False,
        },
    }
    textqa = [media, {'type': 'text', 'content': f'<video> {short_answer_prompt(item["question"])}'}]
    mcq = [media, {'type': 'text', 'content': f'<video> '
                 f'{multiple_choice_prompt(item["question"], item.get("options", []))}'}]
    return textqa, mcq


def build_judge_prompt(question: str, target: str, predicted_answer: str) -> str:
    """Build the exact GPT-4o judge prompt used by the official script."""
    return _JUDGE_PROMPT.replace('{question}', question).replace(
        '{target}', target).replace('{predicted_answer}', predicted_answer)


_JUDGE_PROMPT = r'''Your job is to look at a question generated from the video, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]. First, I will give examples of each grade, and then you will grade a new example. The following are examples of CORRECT predicted answers. {FENCE} Question: What is the name of the man's child in the video? Gold target: Malia Obama and Sasha Obama Predicted answer 1: sashaand maliaobama Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001. {FENCE} These predicted answers are all CORRECT because:-They fully contain the important information in the gold target.-They do not contain any information that contradicts the gold target.-Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.-Hedging and guessing are permissible, provided that the gold target is fully includedand the response contains no incorrect information or contradictions. The following are examples of INCORRECT predicted answers. {FENCE} Question: What is the name of the man's child in the video? Gold target: Malia and Sasha Predicted answer 1: Malia. Predicted answer 2: Malia, Sasha, and Susan. Predicted answer 3: Barack Obama does not have any children. Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia. Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children. Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer? Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information. {FENCE} These predicted answers are all INCORRECT because:-A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'mnot sure, i think") are also considered incorrect. The following are examples of NOT_ATTEMPTED predicted answers. {FENCE} Question: What is the name of the man's child in the video? Gold target: Malia and Sasha Predicted answer 1: I don't know. Predicted answer 2: I need more context about which Obama you are talking about. Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children. Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one. {FENCE} These predicted answers are all NOT_ATTEMPTED because:-The important information in the gold target is not included in the answer.-No statements in the answer contradict the gold target.

Also note the following things:-For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". -Predicted answers "120k", "124k", and 115k" are all CORRECT. -Predicted answers "100k" and "113k" are INCORRECT. -Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.-The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.-For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.-Do not punish predicted answers if they omit information that would be clearly inferred from the question.-For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".-Consider the question "What award did A pretrainer'sguide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.-For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.-For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.-Do not punish for typos in people's name if it's clearly the same name. -For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "HyoongWon Choong", "HyungwonChung", or "Hyun Won Chung".

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
{FENCE}
Question:{question}
Goldtarget:{target}
Predictedanswer:{predicted_answer}
{FENCE}
Grade the predicted answer ofthe question as one of: A: CORRECT B: INCORRECT C: NOT_ATTEMPTED Just return the letter "A", "B", or "C", with no text around it.
'''.strip().replace('{FENCE}', chr(96) * 3).replace(
    'Hyun Won Chung".\n\nHere', 'Hyun Won Chung". \n\nHere'
).replace(
    'grade the answer.\n' + chr(96) * 3, 'grade the answer. \n' + chr(96) * 3
).replace(
    'Question:{question}\n', 'Question:{question} \n'
).replace(
    'Goldtarget:{target}\n', 'Goldtarget:{target} \n'
).replace(
    'Predictedanswer:{predicted_answer}\n', 'Predictedanswer:{predicted_answer} \n'
).replace(
    chr(96) * 3 + '\nGrade the predicted',
    chr(96) * 3 + ' \nGrade the predicted'
)


def option_judge(gt: str, response: str) -> bool:
    response = str(response).lower()
    if 'the answer is' in response:
        response = response.split('the answer is')[-1].strip()
    elif 'answer:' in response:
        response = response.split('answer:')[-1].strip()
    elif 'the option is' in response:
        response = response.split('the option is')[-1].strip()
    for char in response:
        if char.isalpha():
            response = char
            break
    return bool(response) and response[0] == str(gt).lower()


def _safe_extract(tar_path: str, target_dir: str) -> None:
    target = Path(target_dir).resolve()
    with tarfile.open(tar_path, 'r:*') as archive:
        members = []
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if destination != target and target not in destination.parents:
                raise ValueError(f'Unsafe path in VideoEval-Pro archive: {member.name}')
            members.append(member)
        archive.extractall(target, members=members)


def _extract_video_archives(dataset_path: str) -> str:
    video_root = osp.join(dataset_path, 'videos')
    if osp.isdir(video_root) and any(Path(video_root).glob('*.mp4')):
        return video_root
    nested_root = osp.join(video_root, 'videos')
    if osp.isdir(nested_root) and any(Path(nested_root).glob('*.mp4')):
        return nested_root
    parts = sorted(Path(dataset_path).glob('videos/videos_part_*.tar.gz'))
    if not parts:
        parts = sorted(Path(dataset_path).glob('videos_part_*.tar.gz'))
    if not parts:
        return video_root
    os.makedirs(video_root, exist_ok=True)
    merged = Path(dataset_path) / 'videos_merged.tar.gz'
    if not merged.exists():
        with merged.open('wb') as out:
            for part in parts:
                with part.open('rb') as source:
                    while chunk := source.read(16 * 1024 * 1024):
                        out.write(chunk)
    _safe_extract(str(merged), video_root)
    if osp.isdir(nested_root) and any(Path(nested_root).glob('*.mp4')):
        return nested_root
    return video_root


class VideoEvalPro(VideoBaseDataset):
    """VideoEval-Pro short-answer split."""

    TYPE = 'Video-VQA'
    DEFAULT_JUDGE_MODEL = VIDEOEVAL_PRO_JUDGE_MODEL
    MD5 = ''
    TASK = 'textqa'

    def __init__(self, dataset='VideoEval-Pro', nframe=32, fps=-1, repo_id=VIDEOEVAL_PRO_REPO):
        if dataset not in self.supported_datasets():
            supported = ', '.join(self.supported_datasets())
            raise ValueError(
                f'{self.__class__.__name__} expects one of [{supported}], got {dataset!r}'
            )
        self.repo_id = repo_id
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['VideoEval-Pro']

    @staticmethod
    def _find_data_file(dataset_path: str, dataset_name: str) -> str:
        candidates = [
            osp.join(dataset_path, 'data', 'test-00000-of-00001.parquet'),
            osp.join(dataset_path, 'test-00000-of-00001.parquet'),
            osp.join(dataset_path, 'data', f'{dataset_name}.parquet'),
            osp.join(dataset_path, 'data', f'{dataset_name}.tsv'),
            osp.join(dataset_path, f'{dataset_name}.parquet'),
            osp.join(dataset_path, f'{dataset_name}.tsv'),
            osp.join(dataset_path, 'data', 'VideoEval-Pro.parquet'),
            osp.join(dataset_path, 'data', 'VideoEval-Pro.tsv'),
            osp.join(dataset_path, 'VideoEval-Pro.parquet'),
            osp.join(dataset_path, 'VideoEval-Pro.tsv'),
        ]
        for path in candidates:
            if osp.exists(path):
                return path
        raise FileNotFoundError(
            f'VideoEval-Pro metadata was not found under {dataset_path}. '
            f'Expected one of: {", ".join(candidates)}'
        )

    @classmethod
    def _find_lmu_dataset_path(cls, dataset_name: str) -> str | None:
        lmu_root = LMUDataRoot()
        candidates = [
            osp.join(lmu_root, 'datasets', 'VideoEval-Pro'),
            osp.join(lmu_root, 'VideoEval-Pro'),
            lmu_root,
        ]
        for dataset_path in candidates:
            if not osp.isdir(dataset_path):
                continue
            try:
                cls._find_data_file(dataset_path, dataset_name)
            except FileNotFoundError:
                continue
            return osp.abspath(dataset_path)
        return None

    def prepare_dataset(self, dataset_name='VideoEval-Pro', repo_id=VIDEOEVAL_PRO_REPO):
        if os.environ.get('VLMEVALKIT_USE_MODELSCOPE') in ['1', 'True']:
            warnings.warn(
                'VideoEval-Pro is hosted on Hugging Face and does not support '
                'downloading from ModelScope; the ModelScope setting is ignored.',
                UserWarning,
                stacklevel=2,
            )
        dataset_path = self._find_lmu_dataset_path(dataset_name)
        if dataset_path is None:
            dataset_path = osp.join(LMUDataRoot(), 'datasets', 'VideoEval-Pro')
            os.makedirs(dataset_path, exist_ok=True)
            dataset_path = snapshot_download(
                repo_id=self.repo_id, repo_type='dataset', local_dir=dataset_path
            )
        data_file = self._find_data_file(dataset_path, dataset_name)
        video_root = _extract_video_archives(dataset_path)
        if not osp.isdir(video_root):
            candidates = [
                osp.join(dataset_path, 'video'),
                osp.join(dataset_path, 'data', 'videos'),
                osp.join(dataset_path, 'data', 'video'),
            ]
            video_root = next((path for path in candidates if osp.isdir(path)), dataset_path)
        return dict(data_file=data_file, root=video_root)

    def _video_path(self, video: str) -> str:
        video = str(video)
        path = video if osp.isabs(video) else osp.join(self.data_root, video)
        if osp.isfile(path):
            return path
        if not osp.splitext(path)[1]:
            for suffix in ('.mp4', '.webm', '.mov', '.mkv'):
                if osp.isfile(path + suffix):
                    return path + suffix
        raise FileNotFoundError(
            f'VideoEval-Pro video not found: {video!r}. Checked {path!r} and common extensions.'
        )

    def save_video_frames(self, video, video_llm=False):
        import decord
        video_path = self._video_path(video)
        reader = decord.VideoReader(video_path, num_threads=1)
        fps = reader.get_avg_fps()
        vlen = len(reader)
        stem = osp.splitext(osp.basename(str(video)))[0]
        if self.fps > 0:
            duration = vlen / fps
            required = max(1, int(duration * self.fps))
            step = fps / self.fps
            indices = [min(vlen - 1, int(i * step)) for i in range(required)]
            frame_paths = self.frame_paths_fps(stem, len(indices))
        else:
            if self.nframe <= 0:
                raise ValueError('VideoEval-Pro requires a positive nframe or fps for frame input')
            step = vlen / (self.nframe + 1)
            indices = [min(vlen - 1, int(i * step)) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(stem)
        if not np.all([osp.exists(path) for path in frame_paths]):
            images = [Image.fromarray(reader[index].asnumpy()) for index in indices]
            for image, path in zip(images, frame_paths):
                if not osp.exists(path):
                    image.save(path)
        return frame_paths

    def _build_vlmeval_message(self, line, video_llm=False):
        video_path = self._video_path(line['video'])
        if self.TASK == 'mcq':
            text = multiple_choice_prompt(line['question'], line.get('options', []))
        else:
            text = short_answer_prompt(line['question'])
        if video_llm:
            media_items = [dict(
                type='video',
                value=video_path,
                metadata={
                    'video_num_frames': self.nframe if self.nframe > 0 else 32,
                    'video_sample_type': 'rand',
                    'img_shortest_edge': 256,
                    'img_longest_edge': 480,
                    'max_img_seq_len': 16000,
                    'do_resize': False,
                },
            )]
        else:
            media_items = [
                dict(type='image', value=path)
                for path in self.save_video_frames(line['video'], video_llm=False)
            ]
        return media_items + [dict(type='text', value=f'<video> {text}')]

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return self._build_vlmeval_message(line, video_llm=video_llm)

    @staticmethod
    def _judge_response(model, prompt: str):
        messages = [dict(type='text', value=prompt)]
        if hasattr(model, 'generate'):
            return model.generate(messages, dataset='VideoEval-Pro')
        return model(messages)

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], (
            'data file should be an supported format (xlsx/json/tsv) file'
        )
        data = load(eval_file)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        scores = []
        model = judge_kwargs.get('model', cls.DEFAULT_JUDGE_MODEL)
        if cls.TASK == 'textqa' and model != 'exact_matching' and isinstance(model, str):
            model = build_judge(**dict(judge_kwargs, model=model))
            if not model.working():
                warnings.warn('OPENAI API is not working properly, using exact matching for evaluation')
                warnings.warn(DEBUG_MESSAGE)
                model = None
        elif cls.TASK == 'textqa' and model == 'exact_matching':
            model = None
        for _, row in data.iterrows():
            prediction = '' if pd.isna(row.get('prediction')) else str(row.get('prediction'))
            if cls.TASK == 'mcq':
                scores.append(int(option_judge(row.get('answer', ''), prediction)))
                continue
            target = '' if pd.isna(row.get('answer_text')) else str(row.get('answer_text'))
            if model is None:
                scores.append(int(prediction.strip().lower() == target.strip().lower()))
            else:
                prompt = build_judge_prompt(str(row.get('question', '')).strip(), target.strip(), prediction.strip())
                result = cls._judge_response(model, prompt)
                scores.append(int(str(result).strip()[:1].upper() == 'A'))
        data['score'] = scores
        score_file = get_intermediate_file_path(eval_file, '_score')
        dump(data, score_file)
        success = int(sum(scores))
        overall = len(scores)
        return {cls.TASK: {'success': success, 'overall': overall,
                           'acc': round(success / overall * 100, 2) if overall else 0.0}}


class VideoEvalPro_MCQ(VideoEvalPro):
    TYPE = 'Video-MCQ'
    TASK = 'mcq'
    DEFAULT_JUDGE_MODEL = 'exact_matching'

    def __init__(self, dataset='VideoEval-Pro-MCQ', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['VideoEval-Pro-MCQ']


class VideoEvalPro_OpenEnded(VideoEvalPro):
    TYPE = 'Video-VQA'
    TASK = 'textqa'

    def __init__(self, dataset='VideoEval-Pro-OpenEnded', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    @classmethod
    def supported_datasets(cls):
        return ['VideoEval-Pro-OpenEnded']


__all__ = [
    'VideoEvalPro_MCQ', 'VideoEvalPro_OpenEnded',
    'build_official_inference_inputs', 'build_judge_prompt',
    'multiple_choice_prompt', 'option_judge', 'short_answer_prompt',
]
