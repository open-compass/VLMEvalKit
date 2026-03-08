import os.path as osp
import pandas as pd
import validators

from ...smp import *
from .gen_base import GenBaseImageDataset
from ..utils.judge_util import build_judge
from ..utils.physics_eval_utils import extract_all_boxed_content


SYS_PROMPT_PERCEPTION = """You are an expert AI assistant that interprets user instructions and generates modified images that mark specific points on an image. Your primary task is to generate images that contain one or more marked points as requested.

Instructions:

Mandatory Image Generation: You must respond by generating an image with the requested point markings. Do not provide plain-text answers or refuse to generate the annotated image.

Strict Task Limitation - Mark Points Only:
Your sole function is to mark points on the image. You must refuse to perform any other image manipulation task such as drawing continuous lines, cropping, rotating, or applying filters.
Your generated image should only include the marked point(s).

Method of Point Marking:
Use your built-in image generation capability to draw a small, visible mark at the target coordinates—such as a small filled circle, a cross, or another minimal, clear indicator. A small round dot is often a good choice.

Coordinate Accuracy:
You must determine the exact coordinates according to the user's request and the provided image size/context.
Do not use placeholder or example coordinates (e.g., (10, 10)) when the request implies a specific location (e.g., “mark the center,” “put a dot in the top-right corner”).

Focus:
Your response must only perform the requested point-marking action.
No additional edits beyond marking the specified points are allowed.
"""  # noqa: E501

SYS_PROMPT_IGI = """You are a helpful assistant.

Solve the following problem step by step, and optionally generate images with auxiliary graphics on the original image (such as annotations, auxiliary lines, regions, points, etc.), in order to engage in interactive mathematical reasoning or visual question answering.
Any generated image will be returned to aid your reasoning and help you arrive at the final answer.

Reasoning & Drawing Auxiliary Graphics (Optional but Encouraged):
    •   You have the capability to directly generate images that include auxiliary graphics (such as segments, circles, rectangles, labels, etc.) to help illustrate your reasoning process.
    •   The generated images will be returned to you for further analysis.
    •   Do NOT write Python code.
All auxiliary graphics must be produced using your own image-generation abilities.
    •   Make sure each generated image clearly reflects the auxiliary elements added to support your reasoning.
"""  # noqa: E501


SYS_PROMPT_IR = """ Solve the following problem step by step, and optionally generate images with auxiliary graphics on the image (such as annotations, auxiliary lines, regions, points, etc.), in order to engage in interactive mathematical reasoning or visual question answering.
Any generated image will be returned to aid your reasoning and help you arrive at the final answer.

Reasoning & Drawing Auxiliary Graphics (Optional but Encouraged):
    •   You have the capability to directly generate modified images that contain auxiliary graphics (such as segments, circles, rectangles, labels, etc.) to help illustrate your reasoning process.
    •   The generated images will be returned to you for further analysis.
    •   Do NOT write Python code.
All visuals should be produced using your own image-generation capabilities.
    •   When you return a generated image, include a brief description of what annotations were added.
    •   At the end, provide clear step-by-step reasoning that leads to the final answer.
"""  # noqa: E501


PROMPT_IMAGE_JUDGE_PERCEPTION = """Task Description: You are an expert visual evaluator. Your task is to determine if the point(s) marked in the [Generated Image] are conceptually correct according to the Instruction, using the [Ground Truth Image] as a reference for the correct concept.

Judging Criteria: The primary goal is to assess if the geometric concept is correctly applied, not to enforce pixel-perfect accuracy.

Consistent (Judgement: 1): The generated image is considered consistent if it meets the following conditions:

Conceptual Correctness: The point(s) are marked on the correct intended features (e.g., marking the correct vertex, the midpoint of the correct line segment, the center of the correct circle, or the intersection of specific lines).

Tolerance for Imperfection: The result is visually and conceptually similar to the [Ground Truth Image]. Minor inaccuracies in the exact pixel position are acceptable as long as the point clearly identifies the correct feature.

Irrelevant Differences: Differences in the point's color, size, or style (e.g., a dot vs. a small circle vs. an 'x') should be ignored.

Inconsistent (Judgement: 0): The generated image is considered inconsistent if:

Conceptual Error: The point(s) are marked on the wrong features (e.g., marking the wrong vertex, the midpoint of the wrong line, or a random location not specified by the instruction).

Missing or Incomplete: The required point(s) are missing or do not follow the core instruction.

No Change: The [Generated Image] is identical to the [Original Image] (no effective modifications were made).

Output Format:

If consistent, output 1.

If inconsistent, output 0.

Output ONLY 0 or 1. Do not provide any explanation.

[Instruction]: {instruction}

There are three images attached in order:
1) Original Image (before applying the instruction)
2) Generated Image (produced by the model)
3) Ground Truth Image (reference for the correct concept)

Please compare all the information above and provide your judgement. Judgement:"""  # noqa: E501


PROMPT_IMAGE_JUDGE_IGI = """Task Description: You are an expert visual evaluator. Your task is to determine if the auxiliary lines or modifications in the [Generated Image] are conceptually correct according to the [Instruction], using the [Ground Truth Image] as a reference for the correct concept.

Judging Criteria: The primary goal is to assess if the geometric concept is correctly applied, not to enforce pixel-perfect accuracy.

Consistent (Judgement: 1): The generated image is considered consistent if it meets the following conditions:

Conceptual Correctness: The modifications correctly follow the geometric/logical concept of the instruction. The key is that the auxiliary lines connect the correct intended features (e.g., connecting the correct vertices, bisecting the correct angle, drawing a perpendicular from the right point).

Tolerance for Imperfection: The result is visually and conceptually similar to the [Ground Truth Image]. Minor inaccuracies in position, angle, or length are acceptable as long as the core intent is preserved.

Irrelevant Differences: Differences in color, line thickness, or line style (e.g., dashed vs. solid) should be ignored.

Inconsistent (Judgement: 0): The generated image is considered inconsistent if:

Conceptual Error: The modifications are conceptually wrong (e.g., connecting the wrong vertices, bisecting the wrong angle, dropping a perpendicular to the wrong line).

Missing or Incomplete: The required modifications are missing, incomplete, or do not follow the core instruction.

No Change: The [Generated Image] is identical to the [Original Image] (no effective modifications were made).

Output Format:

If consistent, output 1.

If inconsistent, output 0.

Output ONLY 0 or 1. Do not provide any explanation.

[Instruction]: {instruction}

There are three images attached in order:
1) Original Image (before applying the instruction)
2) Generated Image (produced by the model)
3) Ground Truth Image (reference for the correct concept)

Please compare all the information above and provide your judgement. Judgement:"""  # noqa: E501


PROMPT_IR_JUDGE = """Below are two answers to a math question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question. Determine whether these two answers are consistent.
Please note that only when the [Model_answer] completely matches the [Standard Answer] means they are consistent. For non-multiple-choice questions, if the meaning is expressed in the same way, it is also considered consistent, for example, 0.5m and 50cm.
If they are consistent, Judgement is 1; if they are different, Judgement is 0.
Be strict.

Output ONLY 0 or 1. Do not provide any explanation.

[Question]: {question}
[Standard Answer]: {gt}
[Model_answer]: {extraction}
Judgement:"""  # noqa: E501


def _first_existing_key(line, keys):
    if isinstance(line, pd.DataFrame):
        for k in keys:
            if k in line.columns:
                return k
        return None
    for k in keys:
        if k in line and (not pd.isna(line[k])) and str(line[k]) != '':
            return k
    return None


def _to_zero_or_one(text):
    if text is None:
        return 0
    s = str(text).strip()
    for ch in s:
        if ch == '0':
            return 0
        if ch == '1':
            return 1
    return 0


def _extract_text_from_prediction(pred):
    if pred is None:
        return ''
    if isinstance(pred, str) and pred[0] == '[' and pred[-1] == ']':
        pred = eval(pred)
    if isinstance(pred, str):
        return pred
    if not isinstance(pred, list) or not pred:
        return ''
    if isinstance(pred[0], list):
        gen = pred[-1]
        texts = [x for x in gen if isinstance(x, str) and not GenBaseImageDataset.is_path_image(x)]
        return '\n'.join(texts)
    texts = [x for x in pred if isinstance(x, str) and not GenBaseImageDataset.is_path_image(x)]
    return '\n'.join(texts)


def _strip_think_tags(text: str) -> str:
    if not isinstance(text, str):
        return ''
    end_token = '</think_never_used_51bce0c785ca2f68081bfa7d91973934>'
    if end_token in text:
        text = text.split(end_token)[-1]
    if '</think>' in text:
        text = text.split('</think>')[-1]
    return text.strip()


class VTBench(GenBaseImageDataset):
    TYPE = 'TI2TI'
    NUM_GENERATIONS = 1
    DEFAULT_JUDGE = 'gpt-4o-mini'

    DATASET_URL = {
        'VTBench': 'https://opencompass.openxlab.space/utils/GenEval/VTBench.tsv',
    }

    DATASET_MD5 = {
        'VTBench': 'f5a47c49217f43c1af07e06ef10c2ac1',
    }

    def __init__(self, dataset='VTBench', **kwargs):
        super().__init__(dataset=dataset, **kwargs)

    def post_build(self, dataset):
        data = self.data.copy()
        data['index'] = data['index'].astype(str)

        perception = data.copy()
        perception['vtbench_task'] = 'perception'
        perception['original_index'] = perception['index']
        perception['index'] = perception['index'] + '_Perception'
        q_key = _first_existing_key(perception, ['question_perception'])
        if q_key is not None:
            perception['question'] = perception[q_key]
        gt_key = _first_existing_key(perception, ['annotation_perception'])
        if gt_key is not None:
            perception['gt_image'] = perception[gt_key]

        igi = data.copy()
        igi['vtbench_task'] = 'igi'
        igi['original_index'] = igi['index']
        igi['index'] = igi['index'] + '_IGI'
        q_key = _first_existing_key(igi, ['question_instruction'])
        if q_key is not None:
            igi['question'] = igi[q_key]
        gt_key = _first_existing_key(igi, ['annotation_instruction'])
        if gt_key is not None:
            igi['gt_image'] = igi[gt_key]

        ir = data.copy()
        ir['vtbench_task'] = 'ir'
        ir['original_index'] = ir['index']
        ir['index'] = ir['index'] + '_IR'
        q_key = _first_existing_key(ir, ['question_reasoning'])
        if q_key is not None:
            ir['question'] = ir[q_key]
        gt_key = _first_existing_key(ir, ['answer_reasoning'])
        if gt_key is not None:
            ir['gt_answer'] = ir[gt_key]

        self.data = pd.concat([perception, igi, ir], ignore_index=True)

    def _get_question(self, line):
        q = str(line.get('question', ''))
        if str(line.get('vtbench_task', '')) == 'ir':
            return q + '\nPlease put your final answer in \\boxed{...}.'
        return q

    def _resolve_gt_image_path(self, idx: str, gt_value):
        if gt_value is None or (isinstance(gt_value, float) and pd.isna(gt_value)):
            return None
        if isinstance(gt_value, str):
            s = gt_value.strip()
            if s.startswith('data:image/'):
                _, pth = parse_file(s, force_local=True)
                return pth
            if validators.url(s):
                mime, pth = parse_file(s, force_local=True)
                if mime is not None and mime.split('/')[0] == 'image':
                    return pth
                return None
            if osp.exists(s):
                return s
            if len(s) > 64:
                base_name = f'{idx}_gt.png'
                return self.dump_image_atomic(s, base_name)
        return None

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line) if not self.meta_only else toliststr(line.get('image_path', []))
        if not isinstance(tgt_path, list):
            tgt_path = [tgt_path]
        task = str(line.get('vtbench_task', ''))
        if task == 'perception':
            sys_prompt = SYS_PROMPT_PERCEPTION
        elif task == 'igi':
            sys_prompt = SYS_PROMPT_IGI
        else:
            sys_prompt = SYS_PROMPT_IR

        question = self._get_question(line)
        msgs = [dict(type='text', value=sys_prompt)]
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
        msgs.append(dict(type='text', value=question))
        return msgs

    @staticmethod
    def _evaluate_sample_ti2i(judge_model, dataset_obj, sample):
        idx = str(sample.get('index', ''))
        pred_img = GenBaseImageDataset.extract_single_image_from_response(sample.get('prediction'))
        if not pred_img:
            return dict(score=None, reason='PREDICTION MISSING OR NOT AN IMAGE.', judge_raw='')

        try:
            orig_paths = dataset_obj.dump_image(sample)
            orig_img = orig_paths[0] if isinstance(orig_paths, list) and orig_paths else orig_paths
        except Exception as e:
            return dict(score=None, reason=f'FAILED TO DUMP ORIGINAL IMAGE: {e}', judge_raw='')

        gt_img = dataset_obj._resolve_gt_image_path(idx, sample.get('gt_image', None))
        if not gt_img:
            return dict(score=None, reason='GROUND TRUTH IMAGE MISSING.', judge_raw='')

        task = str(sample.get('vtbench_task', ''))
        inst = dataset_obj._get_question(sample)
        if task == 'perception':
            prompt = PROMPT_IMAGE_JUDGE_PERCEPTION.format(instruction=inst)
        else:
            prompt = PROMPT_IMAGE_JUDGE_IGI.format(instruction=inst)
        message = [
            dict(type='text', value=prompt),
            dict(type='image', value=orig_img),
            dict(type='image', value=pred_img),
            dict(type='image', value=gt_img),
        ]
        resp = judge_model.generate(message=message, temperature=0.0)
        return dict(score=_to_zero_or_one(resp), reason='', judge_raw=resp)

    @staticmethod
    def _evaluate_sample_ir(judge_model, dataset_obj, sample):
        pred_text = _extract_text_from_prediction(sample.get('prediction'))
        pred_text = _strip_think_tags(pred_text)
        boxed = extract_all_boxed_content(pred_text)
        extraction = boxed[-1] if boxed else None
        if extraction is None:
            return dict(score=None, reason='NO BOXED ANSWER FOUND.', judge_raw='')

        question = str(sample.get('question', ''))
        gt = str(sample.get('gt_answer', ''))
        prompt = PROMPT_IR_JUDGE.format(question=question, gt=gt, extraction=extraction)
        resp = judge_model.generate(message=[dict(type='text', value=prompt)], temperature=0.0, max_tokens=32)
        return dict(score=_to_zero_or_one(resp), reason='', judge_raw=resp)

    def evaluate(self, eval_file, **judge_kwargs):
        judge_name = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 8)
        _ = judge_kwargs.pop('verbose', None)
        _ = judge_kwargs.pop('retry', None)

        tmp_file = get_intermediate_file_path(eval_file, f'_{judge_name}_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, f'_{judge_name}', 'tsv')
        rating_file = get_intermediate_file_path(eval_file, f'_{judge_name}_rating', 'json')

        judge_kwargs['temperature'] = 0.0
        model = build_judge(**judge_kwargs)
        eval_df = load(eval_file)
        eval_df['index'] = eval_df['index'].astype(str)

        if not osp.exists(tgt_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if v.get('score', None) is not None}

            todo_mask = ~eval_df['index'].isin(res.keys())
            data_un = eval_df[todo_mask].reset_index(drop=True)
            lt = len(data_un)
            if lt > 0:
                samples = [data_un.iloc[i] for i in range(lt)]
                indices = [str(x['index']) for x in samples]
                jobs_ti2i = []
                keys_ti2i = []
                jobs_ir = []
                keys_ir = []
                for sample, key in zip(samples, indices):
                    if str(sample.get('vtbench_task', '')) in ['perception', 'igi']:
                        jobs_ti2i.append(dict(judge_model=model, dataset_obj=self, sample=sample))
                        keys_ti2i.append(key)
                    else:
                        jobs_ir.append(dict(judge_model=model, dataset_obj=self, sample=sample))
                        keys_ir.append(key)
                if len(jobs_ti2i):
                    _ = track_progress_rich(
                        self._evaluate_sample_ti2i,
                        jobs_ti2i,
                        keys=keys_ti2i,
                        save=tmp_file,
                        nproc=nproc
                    )
                if len(jobs_ir):
                    _ = track_progress_rich(
                        self._evaluate_sample_ir,
                        jobs_ir,
                        keys=keys_ir,
                        save=tmp_file,
                        nproc=nproc
                    )

            score_map = load(tmp_file) if osp.exists(tmp_file) else {}
            score_map = {**res, **score_map}
            eval_df['score'] = [score_map.get(idx, {}).get('score', None) for idx in eval_df['index']]
            eval_df['judge_raw'] = [score_map.get(idx, {}).get('judge_raw', '') for idx in eval_df['index']]
            eval_df['reason'] = [score_map.get(idx, {}).get('reason', '') for idx in eval_df['index']]
            dump(eval_df, tgt_file)
            final_result = eval_df
        else:
            final_result = load(tgt_file)

        if osp.exists(tmp_file):
            os.remove(tmp_file)

        rating = self.report_metric_by_task(final_result)
        dump(rating, rating_file)
        return rating

    @staticmethod
    def report_metric(final_result):
        res = {}
        scores = pd.to_numeric(final_result['score'], errors='coerce')
        total_samples = int(len(scores))
        failed_mask = scores.isna()
        num_failures = int(failed_mask.sum())
        res['total_samples'] = total_samples
        res['failed_samples'] = num_failures
        res['failure_rate'] = f"{(num_failures / total_samples):.2%}" if total_samples > 0 else '0.00%'
        valid_scores = scores.dropna()
        res['acc'] = float(valid_scores.mean()) if len(valid_scores) else float('nan')
        res['overall'] = float(res['acc'] * 100) if res['acc'] == res['acc'] else float('nan')
        return res

    @classmethod
    def report_metric_by_task(cls, final_result):
        if 'vtbench_task' not in final_result:
            return {'overall': cls.report_metric(final_result)}

        out = {}
        for task in ['perception', 'igi', 'ir']:
            sub = final_result[final_result['vtbench_task'] == task]
            out[task] = cls.report_metric(sub)

        weighted_sum = 0.0
        valid_total = 0
        for task in ['perception', 'igi', 'ir']:
            scores = pd.to_numeric(final_result[final_result['vtbench_task'] == task]['score'], errors='coerce')
            valid_n = int(scores.notna().sum())
            if valid_n and out[task]['acc'] == out[task]['acc']:
                weighted_sum += float(out[task]['acc']) * valid_n
                valid_total += valid_n

        overall = {}
        overall['acc'] = float(weighted_sum / valid_total) if valid_total else float('nan')
        overall['acc_percent'] = float(overall['acc'] * 100) if overall['acc'] == overall['acc'] else float('nan')
        out['weighted_overall'] = overall
        out['overall'] = np.mean([
            0 if pd.isna(out[x]['overall']) else out[x]['overall']
            for x in ['perception', 'igi', 'ir']
        ])
        return out
