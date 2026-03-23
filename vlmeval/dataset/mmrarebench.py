import json
import os
import os.path as osp
import re
import string
import warnings
from collections import Counter

import pandas as pd
from tqdm import tqdm


from .image_base import ImageBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..smp import *
from ..utils import track_progress_rich
from ..smp.file import LMUDataRoot, get_intermediate_file_path


TRACK_CONFIG = {
    'MMRarebench_Diagnosis': {
        'track': 'diagnosis',
        'tsv': 'diagnosis_opened.tsv',
    },
    'MMRarebench_Treatment': {
        'track': 'treatment',
        'tsv': 'treatment_plan.tsv',
    },
    'MMRarebench_Crossmodal': {
        'track': 'crossmodal',
        'tsv': 'crossmodal_comparison.tsv',
    },
    'MMRarebench_Examination': {
        'track': 'examination',
        'tsv': 'examination_workup.tsv',
    },
}

BENCH_ROOT = os.environ.get('MMRAREBENCH_ROOT', 'MMrarebench')


def parse_json_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == '':
        return []
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        return [s.strip() for s in str(val).split(';') if s.strip()]


def normalize_answer(answer: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def compute_f1(gold_list, predicted):
    def _f1(gold, pred):
        gold_tokens = normalize_answer(gold).split()
        pred_tokens = normalize_answer(pred).split()
        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)
    return max(_f1(g, predicted) for g in gold_list)


def load_track_tsv(dataset_name):
    cfg = TRACK_CONFIG[dataset_name]
    root = LMUDataRoot()
    track_dir = osp.join(root, BENCH_ROOT, cfg['track'])
    tsv_path = osp.join(track_dir, cfg['tsv'])

    if not osp.exists(tsv_path):
        raise FileNotFoundError(
            f'MMRarebench TSV not found: {tsv_path}\n'
            f'Please convert CSV to TSV first using scripts/convert_to_tsv.py'
        )

    data = pd.read_csv(tsv_path, sep='\t', encoding='utf-8')

    if 'index' not in data.columns:
        data['index'] = range(len(data))
    data['index'] = [str(x) for x in data['index']]

    return data


class MMRarebenchBase(ImageBaseDataset):
    """Base class for MMRarebench datasets."""

    MODALITY = 'IMAGE'
    DATASET_URL = {
        'MMRarebench_Diagnosis': (
            'https://huggingface.co/datasets/junzhin/MMrarebench/resolve/main/'
            'diagnosis/diagnosis_opened.tsv'
        ),
        'MMRarebench_Treatment': (
            'https://huggingface.co/datasets/junzhin/MMrarebench/resolve/main/'
            'treatment/treatment_plan.tsv'
        ),
        'MMRarebench_Crossmodal': (
            'https://huggingface.co/datasets/junzhin/MMrarebench/resolve/main/'
            'crossmodal/crossmodal_comparison.tsv'
        ),
        'MMRarebench_Examination': (
            'https://huggingface.co/datasets/junzhin/MMrarebench/resolve/main/'
            'examination/examination_workup.tsv'
        ),
    }
    DATASET_MD5 = {}

    def load_data(self, dataset):
        """Load data from HuggingFace URL or local TSV file."""
        url = self.DATASET_URL.get(dataset, None)
        if url:
            # Use parent class method for HuggingFace URL download
            return self.prepare_tsv(url, None)

        # Fallback to local TSV
        data = load_track_tsv(dataset)
        self.data_path = osp.join(
            LMUDataRoot(), BENCH_ROOT,
            TRACK_CONFIG[dataset]['track'],
            TRACK_CONFIG[dataset]['tsv']
        )
        return data

    def dump_image(self, line):
        """Load images from TSV - supports both base64 and file path.

        Priority:
        1. Use 'image' column (base64 encoded) if available
        2. Fallback to 'image_path' column (file paths)
        """
        from ..smp import decode_base64_to_image_file

        image_b64_str = line.get('image', '[]')
        image_b64_list = parse_json_list(image_b64_str)

        if image_b64_list and any(image_b64_list):
            os.makedirs(self.img_root, exist_ok=True)
            tgt_paths = []

            for i, b64_data in enumerate(image_b64_list):
                if not b64_data:
                    continue
                idx = str(line.get('index', i))
                tgt_path = osp.join(self.img_root, f"{idx}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    try:
                        decode_base64_to_image_file(b64_data, tgt_path)
                    except Exception as e:
                        warnings.warn(f'Failed to decode base64 image: {e}')
                        continue
                tgt_paths.append(tgt_path)

            if tgt_paths:
                return tgt_paths

        image_path_str = line.get('image_path', '[]')
        image_list = parse_json_list(image_path_str)

        if not image_list:
            return []

        root = LMUDataRoot()
        track_dir = osp.join(root, BENCH_ROOT, TRACK_CONFIG[self.dataset_name]['track'])

        abs_paths = []
        for rel_path in image_list:
            if osp.isabs(rel_path):
                abs_paths.append(rel_path)
            else:
                abs_paths.append(osp.join(track_dir, rel_path))

        for p in abs_paths:
            if not read_ok(p):
                warnings.warn(f'Image file not found: {p}')

        return abs_paths


class MMRarebenchOpenEndedBase(MMRarebenchBase):
    """Base class for open-ended VQA tasks."""

    TYPE = 'VQA'

    def _build_open_prompt(self, line, task_instruction=''):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = line['question']
        context = str(line.get('context', '')) if pd.notna(line.get('context', '')) else ''

        fig_map_raw = line.get('figure_mapping', '')
        fig_map = {}
        if pd.notna(fig_map_raw) and str(fig_map_raw).strip():
            try:
                fm = json.loads(str(fig_map_raw))
                if isinstance(fm, dict):
                    fig_map = fm
            except (json.JSONDecodeError, TypeError):
                pass

        basename_to_label = {}
        if fig_map:
            for label, rel_path in fig_map.items():
                basename_to_label[osp.basename(rel_path)] = label

        prompt_parts = []
        if context:
            prompt_parts.append(context)
        prompt_parts.append(f'Question: {question}')
        if task_instruction:
            prompt_parts.append(task_instruction)
        prompt = '\n\n'.join(prompt_parts)

        msgs = []
        for p in tgt_path:
            label = basename_to_label.get(osp.basename(p), '')
            if label:
                msgs.append(dict(type='text', value=f'{label}:'))
            msgs.append(dict(type='image', value=p))
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _parse_judge_score(self, response):
        """
        Parse YES/NO responses. Returns YES ratio as score.
        Falls back to numeric parsing for legacy formats.
        """
        import re
        text = str(response).strip()
        text_lower = text.lower()

        yes_count = len(re.findall(r'\byes\b', text_lower))
        no_count = len(re.findall(r'\bno\b', text_lower))
        total = yes_count + no_count
        if total >= 2:
            return yes_count / total

        if total == 1:
            return 1.0 if yes_count == 1 else 0.0

        m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', text)
        if m:
            return min(float(m.group(1)) / float(m.group(2)), 1.0)

        m = re.search(r'(\d+(?:\.\d+)?)\s*%', text)
        if m:
            return min(float(m.group(1)) / 100.0, 1.0)

        m = re.search(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', text_lower)
        if m:
            return float(m.group(1))

        return 0.0

    def _fallback_evaluate(self, eval_file):
        """Substring match fallback evaluation."""
        data = load(eval_file)
        assert 'answer' in data.columns and 'prediction' in data.columns
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        hits = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc='[Judge] Evaluating'):
            pred = row['prediction'].lower()
            ans = row['answer'].lower()
            hits.append(1.0 if ans[:50] in pred else 0.0)

        results = {
            'substring_match': sum(hits) / len(hits) if hits else 0.0,
            'n_total': len(data),
        }
        score_file = get_intermediate_file_path(eval_file, '_fallback_score', 'json')
        dump(results, score_file)
        return results


class MMRarebenchDiagnosis(MMRarebenchOpenEndedBase):
    """Diagnosis Track: Open-ended free-form diagnosis (Judge + F1)."""

    TYPE = 'VQA'
    DATASET_URL = {k: v for k, v in MMRarebenchBase.DATASET_URL.items() if 'Diagnosis' in k}
    DATASET_MD5 = {k: v for k, v in MMRarebenchBase.DATASET_MD5.items() if 'Diagnosis' in k}

    @classmethod
    def supported_datasets(cls):
        return ['MMRarebench_Diagnosis']

    def build_prompt(self, line):
        return self._build_open_prompt(
            line,
            task_instruction=(
                'Based on the clinical presentation and medical images, '
                'provide your diagnosis. Output ONLY the most likely diagnosis '
                'using the standard clinical disease name. '
                'Do not include explanations or reasoning.'
            )
        )

    def evaluate(self, eval_file, **judge_kwargs):
        """Diagnosis evaluation: Judge Score (3-dim YES/NO weighted) + token-level F1."""
        model_name = judge_kwargs.get('model', 'exact_matching')
        if model_name == 'exact_matching':
            warnings.warn(
                f'{self.dataset_name}: Open-ended QA requires Judge Model. '
                f'Please specify --judge model_name. Falling back to F1.'
            )
            return self._eval_f1_only(eval_file)

        judge_model = build_judge(**judge_kwargs)
        if not judge_model.working():
            warnings.warn('Judge API unavailable, falling back to F1.')
            return self._eval_f1_only(eval_file)

        data = load(eval_file)
        assert 'prediction' in data.columns, 'Missing prediction column'
        data['prediction'] = [str(x) for x in data['prediction']]
        meta = self.data

        nproc = judge_kwargs.pop('nproc', 4)
        f1_scores = []
        judge_prompts = []

        for _, row in data.iterrows():
            pred = str(row['prediction']).strip()
            idx = str(row['index'])
            meta_row = meta[meta['index'].astype(str) == idx].iloc[0]

            gold_list = [str(meta_row['answer'])]
            aliases = parse_json_list(meta_row.get('answer_aliases', ''))
            if aliases:
                gold_list.extend([str(a) for a in aliases if str(a).strip()])

            f1_scores.append(compute_f1(gold_list, pred))

            reference = str(meta_row.get('answer', ''))
            prompt = self._build_judge_prompt(row, pred, reference, gold_list)
            judge_prompts.append(prompt)

        def _judge_call(model, prompt):
            try:
                response = model.generate(prompt)
                return dict(score=self._parse_judge_score(response))
            except Exception:
                return dict(score=0.0)

        tups = [dict(model=judge_model, prompt=p) for p in judge_prompts]
        results_list = track_progress_rich(
            _judge_call, tups, nproc=nproc, chunksize=nproc,
            description='[Judge] Diagnosis'
        )
        judge_scores = [r['score'] if isinstance(r, dict) else 0.0 for r in results_list]

        data['f1'] = f1_scores
        data['judge_score'] = judge_scores

        results = {
            'diagnosis_judge_score': np.mean(judge_scores) if judge_scores else 0.0,
            'diagnosis_f1': np.mean(f1_scores) * 100,
            'n_total': len(data),
        }

        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        return results

    def _eval_f1_only(self, eval_file):
        """Fallback when no Judge: compute token-level F1 only."""
        data = load(eval_file)
        data['prediction'] = [str(x) for x in data['prediction']]
        meta = self.data

        f1_scores = []
        for _, row in data.iterrows():
            pred = str(row['prediction']).strip()
            idx = str(row['index'])
            meta_row = meta[meta['index'].astype(str) == idx].iloc[0]

            gold_list = [str(meta_row['answer'])]
            aliases = parse_json_list(meta_row.get('answer_aliases', ''))
            if aliases:
                gold_list.extend([str(a) for a in aliases if str(a).strip()])

            f1_scores.append(compute_f1(gold_list, pred))

        results = {
            'diagnosis_f1': np.mean(f1_scores) * 100,
            'n_total': len(data),
        }
        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        return results

    def _build_judge_prompt(self, row, pred, reference, gold_list):
        """Build Diagnosis Judge Prompt (3-dim YES/NO, focus on diagnostic accuracy)."""
        correct_diagnosis = gold_list[0] if gold_list else reference
        aliases = gold_list[1:] if len(gold_list) > 1 else []

        prompt = (
            'You are an extremely strict medical expert evaluator. '
            'You are evaluating whether a model correctly identified a diagnosis '
            'from clinical presentation and medical images. '
            'The model was asked to output ONLY the disease name. '
            'Apply the STRICTEST possible interpretation. '
            'Answer YES ONLY when the criterion is COMPLETELY satisfied.\n\n'
            f'Reference diagnosis: {correct_diagnosis}\n'
        )
        if aliases:
            prompt += f'Acceptable synonyms/abbreviations: {"; ".join(aliases)}\n'
        prompt += (
            f'\nModel response: {pred}\n\n'
            'IMPORTANT CALIBRATION: Only models that name the EXACT correct diagnosis '
            '(or an accepted synonym) should receive YES on question 1. '
            'Naming a related but different disease, a broader category, or a partial match is NO.\n\n'
            'Answer each question with YES or NO only:\n'
            '1. EXACT DIAGNOSIS MATCH: Does the model response name the correct diagnosis '
            '(or an accepted synonym/abbreviation listed above) as its answer? '
            'The disease name must be an exact or near-exact match — '
            'naming a parent category (e.g., "vasculitis" instead of "granulomatosis with polyangiitis"), '
            'a sibling disease, or a related but distinct condition is NO. (YES/NO)\n'
            '2. CORRECT DISEASE CATEGORY: Even if the exact diagnosis is wrong, '
            'does the response identify the correct organ system AND disease category? '
            'For example, if the answer is "granulomatosis with polyangiitis" and the model says '
            '"microscopic polyangiitis", the category (ANCA-associated vasculitis) is correct. '
            'But saying "pneumonia" for a vasculitis case is NO. (YES/NO)\n'
            '3. DIAGNOSTIC SPECIFICITY: Is the model\'s response a specific disease entity '
            'rather than a vague or overly broad term? '
            'Responses like "infection", "tumor", "inflammatory condition", or "autoimmune disease" '
            'without specifying the exact disease are NO. (YES/NO)\n\n'
            'Reply with ONLY three YES or NO answers, one per line.'
        )
        return prompt

    def _parse_judge_score(self, response):
        """Parse 3-dim YES/NO score, weighted normalization to [0, 1].

        Weights: dim1(exact match)=3, dim2(category correct)=1, dim3(specificity)=1, total=5.
        Cascade: dim1=NO caps total score at 0.2.
        """
        import re
        text = str(response).strip().lower()

        answers = re.findall(r'\b(yes|no)\b', text)
        if len(answers) >= 3:
            dims = [1 if a == 'yes' else 0 for a in answers[:3]]
            if dims[0] == 0:
                return (dims[1] + dims[2]) / 10.0
            return (3 * dims[0] + dims[1] + dims[2]) / 5.0

        m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', text)
        if m:
            return min(float(m.group(1)) / float(m.group(2)), 1.0)

        return 0.0


class MMRarebenchTreatment(MMRarebenchOpenEndedBase):
    """Treatment Track: Rubric scoring + Judge evaluation."""

    TYPE = 'VQA'
    DATASET_URL = {k: v for k, v in MMRarebenchBase.DATASET_URL.items() if 'Treatment' in k}
    DATASET_MD5 = {k: v for k, v in MMRarebenchBase.DATASET_MD5.items() if 'Treatment' in k}

    @classmethod
    def supported_datasets(cls):
        return ['MMRarebench_Treatment']

    def build_prompt(self, line):
        return self._build_open_prompt(
            line,
            task_instruction=(
                'Based on the clinical presentation and medical images provided, '
                'propose a detailed treatment plan. Include specific medications, '
                'procedures, and follow-up recommendations as appropriate.'
            )
        )

    def evaluate(self, eval_file, **judge_kwargs):
        """Treatment Track evaluation: Judge Score (6-dim YES/NO)."""
        model_name = judge_kwargs.get('model', 'exact_matching')
        if model_name == 'exact_matching':
            warnings.warn(
                f'{self.dataset_name}: Open-ended QA requires Judge Model. '
                f'Please specify --judge model_name. Falling back to substring matching.'
            )
            return self._fallback_evaluate(eval_file)

        judge_model = build_judge(**judge_kwargs)
        if not judge_model.working():
            warnings.warn('Judge API unavailable, falling back to substring matching.')
            return self._fallback_evaluate(eval_file)

        data = load(eval_file)
        assert 'prediction' in data.columns, 'Missing prediction column'
        data['prediction'] = [str(x) for x in data['prediction']]

        nproc = judge_kwargs.pop('nproc', 4)
        judge_prompts = []

        for _, row in data.iterrows():
            pred = row['prediction']
            must_include = parse_json_list(row.get('rubric_must_include', ''))
            must_not = parse_json_list(row.get('rubric_must_not', ''))
            reference = str(row.get('answer', ''))
            prompt = self._build_judge_prompt(row, pred, must_include, must_not, reference)
            judge_prompts.append(prompt)

        def _judge_call(model, prompt):
            try:
                response = model.generate(prompt)
                return dict(score=self._parse_judge_score(response))
            except Exception:
                return dict(score=0.0)

        tups = [dict(model=judge_model, prompt=p) for p in judge_prompts]
        results_list = track_progress_rich(
            _judge_call, tups, nproc=nproc, chunksize=nproc,
            description='[Judge] Treatment'
        )
        judge_scores = [r['score'] if isinstance(r, dict) else 0.0 for r in results_list]

        data['judge_score'] = judge_scores

        results = {
            'treatment_judge_score': sum(judge_scores) / len(judge_scores) if judge_scores else 0.0,
            'n_total': len(data),
        }

        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        return results

    def _build_judge_prompt(self, row, pred, must_include, must_not, reference):
        """Build Treatment Track Judge Prompt (Yes/No multi-dim, strictest standard)."""
        prompt = (
            'You are an extremely strict medical expert evaluator conducting a rigorous peer review. '
            'Your standard is that of a senior attending physician reviewing a resident\'s treatment plan. '
            'Apply the STRICTEST possible interpretation of each criterion. '
            'Answer YES ONLY when the criterion is COMPLETELY, EXPLICITLY, and SPECIFICALLY satisfied '
            'with no ambiguity whatsoever. '
            'A merely plausible or partially correct response MUST receive NO. '
            'Generic medical advice that could apply to any patient MUST receive NO. '
            'If information is implied but not explicitly stated, answer NO.\n\n'
            f'Question: {row["question"]}\n\n'
            f'Reference answer: {reference}\n\n'
            f'Model response: {pred}\n\n'
        )
        if must_include:
            prompt += f'Required treatment elements: {"; ".join(must_include)}\n'
        if must_not:
            prompt += f'Safety violations to check: {"; ".join(must_not)}\n'
        prompt += (
            '\nEvaluate with MAXIMUM strictness. Answer each question with YES or NO only:\n'
            '1. Does the response explicitly name the EXACT SAME specific treatments as the reference '
            '(generic categories like "chemotherapy" or "antibiotics" without specifying the '
            'exact regimen do NOT count)? (YES/NO)\n'
            '2. Does the response include the CORRECT specific drug names AND dosages/routes that match the reference '
            '(mentioning a drug class without the specific agent is NO; omitting dosage is NO)? (YES/NO)\n'
            '3. Is the response COMPLETELY free of ALL safety violations, contraindications, drug interactions, '
            'and clinically dangerous suggestions (even one minor safety issue means NO)? (YES/NO)\n'
            '4. Does the response specify CONCRETE follow-up timelines (e.g., "reassess in 2 weeks") and '
            'SPECIFIC monitoring parameters (e.g., "monitor serum creatinine") rather than vague statements '
            'like "regular follow-up" or "monitor labs"? (YES/NO)\n'
            '5. Does the response demonstrate CORRECT understanding of the pathophysiology that directly '
            'justifies the chosen treatment (not just restating the diagnosis)? (YES/NO)\n'
            '6. Is the response detailed and specific enough that a clinician could directly implement it '
            'as a treatment plan WITHOUT needing additional information? (YES/NO)\n\n'
            'Default to NO unless you are CERTAIN the criterion is fully met. '
            'Reply with ONLY six YES or NO answers, one per line.'
        )
        return prompt


class MMRarebenchCrossmodal(MMRarebenchOpenEndedBase):
    """Crossmodal Track: Cross-modal evidence correlation."""

    TYPE = 'VQA'
    DATASET_URL = {k: v for k, v in MMRarebenchBase.DATASET_URL.items() if 'Crossmodal' in k}
    DATASET_MD5 = {k: v for k, v in MMRarebenchBase.DATASET_MD5.items() if 'Crossmodal' in k}

    @classmethod
    def supported_datasets(cls):
        return ['MMRarebench_Crossmodal']

    def build_prompt(self, line):
        return self._build_open_prompt(
            line,
            task_instruction=(
                'Analyze the relationship between the provided medical images and clinical information. '
                'Describe the cross-modal findings, their relationship (confirm, complement, or contrast), '
                'and the clinical significance of this relationship.'
            )
        )

    def evaluate(self, eval_file, **judge_kwargs):
        """Crossmodal Track evaluation: Judge Score (5-dim YES/NO)."""
        model_name = judge_kwargs.get('model', 'exact_matching')
        if model_name == 'exact_matching':
            warnings.warn(
                f'{self.dataset_name}: Open-ended QA requires Judge Model. '
                f'Please specify --judge model_name. Falling back to substring matching.'
            )
            return self._fallback_evaluate(eval_file)

        judge_model = build_judge(**judge_kwargs)
        if not judge_model.working():
            warnings.warn('Judge API unavailable, falling back to substring matching.')
            return self._fallback_evaluate(eval_file)

        data = load(eval_file)
        assert 'prediction' in data.columns, 'Missing prediction column'
        data['prediction'] = [str(x) for x in data['prediction']]

        nproc = judge_kwargs.pop('nproc', 4)
        judge_prompts = []

        for _, row in data.iterrows():
            pred = row['prediction']

            mainline = parse_json_list(row.get('golden_mainline', ''))
            if mainline and isinstance(mainline[0], dict):
                mainline = [item.get('claim', str(item)) for item in mainline]

            reference = str(row.get('answer', ''))
            rel_type = str(row.get('relation_type', '')).lower().strip()
            prompt = self._build_judge_prompt(row, pred, mainline, rel_type, reference)
            judge_prompts.append(prompt)

        def _judge_call(model, prompt):
            try:
                response = model.generate(prompt)
                return dict(score=self._parse_judge_score(response))
            except Exception:
                return dict(score=0.0)

        tups = [dict(model=judge_model, prompt=p) for p in judge_prompts]
        results_list = track_progress_rich(
            _judge_call, tups, nproc=nproc, chunksize=nproc,
            description='[Judge] Crossmodal'
        )
        judge_scores = [r['score'] if isinstance(r, dict) else 0.0 for r in results_list]

        data['judge_score'] = judge_scores

        results = {
            'crossmodal_judge_score': sum(judge_scores) / len(judge_scores) if judge_scores else 0.0,
            'n_total': len(data),
        }

        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        return results

    def _build_judge_prompt(self, row, pred, mainline, rel_type, reference):
        """Build Crossmodal Track Judge Prompt (Yes/No multi-dim, strictest standard)."""
        prompt = (
            'You are an extremely strict medical expert evaluator conducting a rigorous peer review. '
            'Your standard is that of a senior radiologist reviewing cross-modal imaging interpretations. '
            'Apply the STRICTEST possible interpretation of each criterion. '
            'Answer YES ONLY when the criterion is COMPLETELY, EXPLICITLY, and SPECIFICALLY satisfied '
            'with no ambiguity whatsoever. '
            'A merely plausible or partially correct response MUST receive NO. '
            'Generic observations that could apply to any case MUST receive NO. '
            'If information is implied but not explicitly stated, answer NO.\n\n'
            f'Question: {row["question"]}\n\n'
            f'Reference answer: {reference}\n\n'
            f'Model response: {pred}\n\n'
        )
        if mainline:
            prompt += f'Key medical findings: {"; ".join(mainline)}\n'
        if rel_type and rel_type != 'nan':
            prompt += f'Expected cross-modal relationship: {rel_type}\n'
        prompt += (
            '\nEvaluate with MAXIMUM strictness. Answer each question with YES or NO only:\n'
            '1. Does the response correctly identify ALL the SPECIFIC key findings listed above '
            '(missing even one key finding means NO; vague descriptions that do not name the specific '
            'finding means NO)? (YES/NO)\n'
            '2. Does the response EXPLICITLY describe how findings from DIFFERENT modalities relate to each other '
            '(simply listing findings from each modality separately is NO)? (YES/NO)\n'
            '3. Does the response correctly classify the relationship type as EXACTLY matching the expected type '
            '(confirm/complement/contrast) — using the wrong relationship type or being vague means NO? (YES/NO)\n'
            '4. Is the clinical reasoning ENTIRELY specific to this case with ZERO major factual errors '
            '(any incorrect medical claim, even if minor, means NO)? (YES/NO)\n'
            '5. Does the response provide a genuinely INTEGRATED cross-modal synthesis that derives NEW clinical '
            'insight from combining modalities (not just summarizing each modality independently)? (YES/NO)\n\n'
            'Default to NO unless you are CERTAIN the criterion is fully met. '
            'Reply with ONLY five YES or NO answers, one per line.'
        )
        return prompt


class MMRarebenchExamination(MMRarebenchOpenEndedBase):
    """Examination Track: Diagnostic workup evaluation."""

    TYPE = 'VQA'
    DATASET_URL = {k: v for k, v in MMRarebenchBase.DATASET_URL.items() if 'Examination' in k}
    DATASET_MD5 = {k: v for k, v in MMRarebenchBase.DATASET_MD5.items() if 'Examination' in k}

    @classmethod
    def supported_datasets(cls):
        return ['MMRarebench_Examination']

    def build_prompt(self, line):
        """Build prompt without context to prevent answer leakage."""
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = line['question']

        prompt_parts = [
            f'Question: {question}',
            'Based on the clinical presentation and medical images, recommend the appropriate '
            'examination workup. List the key diagnostic tests that should be ordered, '
            'explain the purpose of each test, and identify any red flags that need to be ruled out.',
        ]
        prompt = '\n\n'.join(prompt_parts)

        msgs = [dict(type='image', value=p) for p in tgt_path]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        """Examination Track evaluation: Judge Score (8-dim 0-1-2 scale)."""
        model_name = judge_kwargs.get('model', 'exact_matching')
        if model_name == 'exact_matching':
            warnings.warn(
                f'{self.dataset_name}: Open-ended QA requires Judge Model. '
                f'Please specify --judge model_name. Falling back to substring matching.'
            )
            return self._fallback_evaluate(eval_file)

        judge_model = build_judge(**judge_kwargs)
        if not judge_model.working():
            warnings.warn('Judge API unavailable, falling back to substring matching.')
            return self._fallback_evaluate(eval_file)

        data = load(eval_file)
        assert 'prediction' in data.columns, 'Missing prediction column'
        data['prediction'] = [str(x) for x in data['prediction']]

        nproc = judge_kwargs.pop('nproc', 4)
        judge_prompts = []

        for _, row in data.iterrows():
            pred = row['prediction']

            key_tests = parse_json_list(row.get('key_test_names', ''))
            must_not = parse_json_list(row.get('must_not', ''))
            red_flags = parse_json_list(row.get('red_flags_to_rule_out', ''))
            reference = str(row.get('answer', ''))
            prompt = self._build_judge_prompt(row, pred, key_tests, must_not, red_flags, reference)
            judge_prompts.append(prompt)

        def _judge_call(model, prompt):
            try:
                response = model.generate(prompt)
                return dict(score=self._parse_judge_score(response))
            except Exception:
                return dict(score=0.0)

        tups = [dict(model=judge_model, prompt=p) for p in judge_prompts]
        results_list = track_progress_rich(
            _judge_call, tups, nproc=nproc, chunksize=nproc,
            description='[Judge] Examination'
        )
        judge_scores = [r['score'] if isinstance(r, dict) else 0.0 for r in results_list]

        data['judge_score'] = judge_scores

        results = {
            'examination_judge_score': sum(judge_scores) / len(judge_scores) if judge_scores else 0.0,
            'n_total': len(data),
        }

        score_file = get_intermediate_file_path(eval_file, '_detailed_metrics', 'json')
        dump(results, score_file)
        return results

    def _build_judge_prompt(self, row, pred, key_tests, must_not, red_flags, reference):
        """Build Examination Track Judge Prompt (8-dim 0-1-2 scale)."""
        prompt = (
            'You are a board-certified physician evaluating a diagnostic workup plan. '
            'Rate each dimension on a 0-1-2 scale.\n'
            'CALIBRATION: Score 2 is RARE and requires textbook-perfect performance. '
            'Score 1 is the expected score for a competent response. '
            'Score 0 means clearly inadequate. '
            'A strong response typically averages 0.8-1.0 across dimensions.\n\n'
            f'Question: {row["question"]}\n\n'
            f'Reference answer: {reference}\n\n'
            f'Model response: {pred}\n\n'
        )
        if key_tests:
            prompt += f'Key diagnostic tests expected: {"; ".join(key_tests)}\n'
        if red_flags:
            prompt += f'Red flags that should be addressed: {"; ".join(red_flags)}\n'
        if must_not:
            prompt += f'Tests that should NOT be recommended: {"; ".join(must_not)}\n'
        prompt += (
            '\nRate each dimension (0, 1, or 2):\n'
            '1. KEY TEST COVERAGE:\n'
            '   2 = names >95% of the key tests with exact clinical names AND correct priority order;\n'
            '   1 = names 60-95% of the key tests, or names most but with imprecise terminology;\n'
            '   0 = names <60% of the key tests or recommends mostly irrelevant tests.\n'
            '2. TEST SPECIFICITY:\n'
            '   2 = ALL tests specify exact parameters (body region, contrast agent, sequences/views);\n'
            '   1 = some tests have parameters but others only name the modality;\n'
            '   0 = most tests are named only by modality without clinical details.\n'
            '3. TEST ORDERING & CONDITIONAL LOGIC:\n'
            '   2 = explicitly states first-line vs second-line vs confirmatory with conditional\n'
            '       branching (e.g., "if X is negative, proceed to Y");\n'
            '   1 = mentions some ordering or priority but lacks explicit branching logic;\n'
            '   0 = flat list of tests with no ordering or sequencing.\n'
            '4. EXPECTED FINDINGS:\n'
            '   2 = describes specific expected findings for each major test (e.g., "MRI may show\n'
            '       hyperintense T2 signal in periventricular region suggesting demyelination");\n'
            '   1 = mentions expected findings for some tests but not most;\n'
            '   0 = does not describe what findings to look for in any test.\n'
            '5. QUANTITATIVE CRITERIA:\n'
            '   2 = provides specific thresholds, lab cutoffs, or scoring systems for most tests\n'
            '       (e.g., "ESR >20mm/hr", "lesion >1cm warrants biopsy");\n'
            '   1 = mentions some quantitative elements but most tests lack thresholds;\n'
            '   0 = no quantitative criteria or measurement standards mentioned.\n'
            '6. DIFFERENTIAL NARROWING:\n'
            '   2 = each test explicitly maps to ruling in/out specific differential diagnoses,\n'
            '       forming a coherent diagnostic decision tree;\n'
            '   1 = some tests linked to differentials but the overall logic is incomplete;\n'
            '   0 = tests are not connected to any differential diagnosis reasoning.\n'
            '7. PATIENT-SPECIFIC ADAPTATION:\n'
            '   2 = tests are explicitly adapted to patient demographics, comorbidities, and history\n'
            '       (e.g., adjusting for age, pregnancy, renal function, allergies);\n'
            '   1 = mentions some patient factors but does not adapt test choices accordingly;\n'
            '   0 = generic workup with no patient-specific adaptation.\n'
            '8. TESTING PRECISION:\n'
            '   2 = zero redundant tests AND explicitly justifies why each is necessary;\n'
            '   1 = mostly relevant but includes some redundancy or lacks justification;\n'
            '   0 = shotgun approach with many irrelevant or redundant tests.\n\n'
            'Reply with ONLY eight scores (0, 1, or 2), one per line.'
        )
        return prompt

    def _parse_judge_score(self, response):
        """Parse 8-dim 0-1-2 scores, normalized to [0, 1]."""
        import re
        text = str(response).strip()

        scores = [int(x) for x in re.findall(r'\b([012])\b', text)]
        if len(scores) >= 8:
            final = sum(scores[:8]) / 16.0
            print(f'[DEBUG Exam] scores={scores[:8]} -> {final:.3f} | raw={text[:200]}')
            return final

        print(f'[DEBUG Exam FALLBACK] only {len(scores)} 0-1-2 scores found | raw={text[:300]}')

        text_lower = text.lower()
        yes_count = len(re.findall(r'\byes\b', text_lower))
        no_count = len(re.findall(r'\bno\b', text_lower))
        total = yes_count + no_count
        if total >= 6:
            return yes_count / total

        m = re.search(r'(\d+(?:\.\d+)?)\s*/\s*(\d+)', text)
        if m:
            return min(float(m.group(1)) / float(m.group(2)), 1.0)

        return 0.0
