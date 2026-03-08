import os
import re
import ast
import base64
import pandas as pd
import warnings

from io import BytesIO
from PIL import Image
from collections import OrderedDict
from huggingface_hub import snapshot_download
from concurrent.futures import ThreadPoolExecutor, as_completed

from .image_mcq import ImageMCQDataset
from ..smp.file import load
from ..smp.misc import toliststr, get_cache_path, modelscope_flag_set


class StareBench(ImageMCQDataset):
    """
    STARE.

    Reference:
      Unfolding Spatial Cognition: Evaluating Multimodal Models on Visual Simulations
      https://arxiv.org/abs/2506.04633
    """

    TYPE = 'MCQ'

    STARE_TSV_URL = 'https://opencompass.openxlab.space/utils/Spatial/StareBench.tsv'
    STARE_TSV_MD5 = '7f09c77d1c697ab91d796ea6f9c9629d'

    VARIANTS = ['StareBench', 'StareBench_CoT']

    DATASET_URL = {}
    DATASET_MD5 = {}

    IMG_ZIP_URL = 'https://opencompass.openxlab.space/utils/Spatial/StareBench.zip'
    IMG_ZIP_MD5 = 'a494415db0e078a9444a2ec765598d82'

    for name in VARIANTS:
        DATASET_URL[name] = STARE_TSV_URL
        DATASET_MD5[name] = STARE_TSV_MD5

    def __init__(self, dataset, skip_noimg=True):
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)
        self.use_cot = self.parse_dataset_name(dataset)

    @staticmethod
    def parse_dataset_name(name: str) -> bool:
        if not isinstance(name, str):
            return False

        lower = name.lower()
        return lower.endswith('_cot')

    def prepare_tsv(self, url, file_md5=None, img_zip=IMG_ZIP_URL, img_zip_md5=IMG_ZIP_MD5):
        return super().prepare_tsv(url, file_md5, img_zip, img_zip_md5)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']

        if '<image>' not in question and tgt_path:
            question += '\n<image>'

        # Multi_choice format from STARE codebase
        # Original STARE code requires return in "\\boxed{{}}".
        # We change to <answer></answer>.
        mcq_format = (
            f'{question}\n\n'
            "Answer with the option's letter from the given choices and put the letter in one <answer> answer </answer>"
        )

        if not self.use_cot:
            # Directly (present in code, but not in paper)
            post_prompt = (
                'Please ensure that your output only contains the final answer (a single option letter) '
                'without any additional content (such as intermediate reasoning steps).'
            )
        else:
            # Zero-Shot CoT (STARE codebase default setting)
            post_prompt = 'Please solve the problem step by step.'

        prompt = '\n'.join([mcq_format, post_prompt])

        msgs = self.build_msgs(tgt_path, prompt)
        return msgs

    @staticmethod
    def build_msgs(tgt_path, prompt):
        """
        Interlaced text and pictures.
        """
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]

        parts = prompt.split('<image>')
        segs = []

        for i, part in enumerate(parts):
            part = part.strip()
            if part:
                segs.append(dict(type='text', value=part))
            if i < len(images):
                segs.append(dict(type='image', value=images[i]))

        return [s for s in segs if s['value']]

    # -------- STARE category config --------
    @classmethod
    def task_category(cls, kind: str = 'ordered'):
        """
        Central place to define / query STARE task categories.

        kind:
          - 'mcq'        -> MCQ-only categories
          - 'binary'     -> binary (yes/no) categories
          - 'ordered'    -> full ordered list for per-category / macro
          - 'binary_set' -> set version of binary categories (for fast membership test)
          - 'all'        -> mcq + binary
        """
        mcq = [
            '2d_trans', '2d_trans_vsim',
            '3d_trans', '3d_trans_vsim',
            'temporal',
            'perspective',
        ]
        binary = [
            'folding_nets', 'folding_nets_vsim',
            'tangram_puzzle', 'tangram_puzzle_vsim',
        ]
        ordered = mcq + binary

        if kind == 'mcq':
            return mcq
        elif kind == 'binary':
            return binary
        elif kind == 'ordered':
            return ordered
        elif kind == 'binary_set':
            return set(binary)
        elif kind == 'all':
            return ordered
        else:
            raise ValueError(f'Unknown kind for task_category: {kind}')

    def _task_category(self):
        """Backwards-compat: return ordered categories."""
        return self.task_category('ordered')

    # -------- F1 / macro helpers --------
    @classmethod
    def _parse_option_mapping(cls, question_text: str):
        """
        Parse mapping from option letter to yes/no in the question text.
        e.g. 'A. yes  B. no' -> {'a': 'yes', 'b': 'no'}
        """
        _ZW_RE = re.compile(r'[\u200b\u200c\u200d\ufeff]')
        if not isinstance(question_text, str):
            return {}
        q = _ZW_RE.sub('', question_text)
        pairs = re.findall(r'([A-F])\s*[\.\:：\)\）]?\s*(yes|no)', q, flags=re.IGNORECASE)
        return {k.lower(): v.strip().lower() for k, v in pairs}

    @staticmethod
    def _f1_binary_yn(gt_list, pred_list, pos_label='yes'):
        """
        Compute F1 for a yes/no binary task given GT / Pred label lists.
        """
        if not gt_list or not pred_list or len(gt_list) != len(pred_list):
            return None
        tp = fp = fn = 0
        for g, p in zip(gt_list, pred_list):
            g = (str(g).lower() == pos_label)
            p = (str(p).lower() == pos_label)
            if p and g:
                tp += 1
            elif p and not g:
                fp += 1
            elif (not p) and g:
                fn += 1
        if tp == 0:
            return 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    @classmethod
    def _compute_f1_for_binary_df(cls, df_cat: pd.DataFrame, pos_label='yes'):
        """
        Compute F1 for one binary category DataFrame.

        Requires columns: question, answer, pred_extracted
        Logic:
          1) Parse A/B/... -> yes/no mapping from question
          2) Map answer / pred_extracted to yes/no
          3) Call _f1_binary_yn
        """
        if len(df_cat) == 0:
            return None
        gts, preds = [], []
        for _, r in df_cat.iterrows():
            q = r.get('question', '')
            gl = r.get('answer', None)
            pl = r.get('pred_extracted', None)
            if not isinstance(q, str) or gl is None or pl in (None, False):
                continue
            mp = cls._parse_option_mapping(q)
            glk, plk = str(gl).lower(), str(pl).lower()
            if glk not in mp or plk not in mp:
                continue
            gts.append(mp[glk])
            preds.append(mp[plk])
        if not gts or len(gts) != len(preds):
            return None
        return cls._f1_binary_yn(gts, preds, pos_label=pos_label)

    @staticmethod
    def _mean_ignore_none(vals):
        xs = [v for v in vals if v is not None]
        return None if not xs else float(sum(xs) / len(xs))

    @classmethod
    def _macro_from_scores(cls, scores: dict, force_divisor_6=False):
        """
        scores: dict[category -> score]
        MCQ categories use accuracy, binary categories use F1.

        Macro has 6 terms:
          2D: (2d_trans, 2d_trans_vsim)
          3D: (3d_trans, 3d_trans_vsim)
          cube: (cube_net, cube_net_vsim)
          tangram: (tangram, tangram_vsim)
          temporal
          perspective
        """
        def g(c):
            return scores.get(c)

        term_2d = cls._mean_ignore_none([g('2d_trans'), g('2d_trans_vsim')])
        term_3d = cls._mean_ignore_none([g('3d_trans'), g('3d_trans_vsim')])
        term_fold = cls._mean_ignore_none([g('folding_nets'), g('folding_nets_vsim')])
        term_tang = cls._mean_ignore_none([g('tangram_puzzle'), g('tangram_puzzle_vsim')])
        term_temp = g('temporal')
        term_pers = g('perspective')

        terms = [term_2d, term_3d, term_fold, term_tang, term_temp, term_pers]
        if force_divisor_6:
            usable = [(t if t is not None else 0.0) for t in terms]
            macro = None if all(t is None for t in terms) else float(sum(usable) / 6.0)
        else:
            usable = [t for t in terms if t is not None]
            macro = None if not usable else float(sum(usable) / len(usable))
        return macro

    def evaluate(self, eval_file, **kwargs):
        """
        STAREBench evaluation.

        - MCQ categories (2d/3d/temporal/perspective): accuracy
        - Binary categories (cube_net / tangram × vsim): F1 (yes/no)
        - Overall: accuracy over all questions
        - Macro: 6-term macro over
            (2D, 3D, cube, tangram, temporal, perspective)
          where MCQ uses accuracy, binary uses F1.
        """
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, attach_score_cache
        from .utils.spatial_bench.tools.files import build_eval_paths, get_judge_tag_from_score_fn

        score_fn = build_mcq_score_fn(**kwargs)
        judge_tag = get_judge_tag_from_score_fn(score_fn)

        result_file, xlsx_path, acc_tsv_path = build_eval_paths(eval_file, judge_tag)

        # 1. load raw results
        data = load(eval_file)
        if isinstance(data, list):
            data = pd.DataFrame(data)
        if 'index' in data.columns:
            data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]

        # 2. compute per-sample hit (MCQ)
        attach_score_cache(
            score_fn=score_fn,
            eval_file=eval_file,
            judge_tag=judge_tag,
            key_col='index',
            sub_tag='mcq',
        )
        mcq_scored = score_fn(data.copy())

        # 3. normalize category column
        if 'category_agg' in mcq_scored.columns:
            cat_col = 'category_agg'
        elif 'category' in mcq_scored.columns:
            cat_col = 'category'
        else:
            cat_col = None

        if cat_col is not None:
            mcq_scored['bench_category'] = (
                mcq_scored[cat_col].astype(str).str.strip().str.lower()
            )
        else:
            mcq_scored['bench_category'] = 'all'

        # 4. overall accuracy
        summary = OrderedDict()
        if len(mcq_scored):
            overall_acc = float(mcq_scored['hit'].mean())
        else:
            overall_acc = 0.0

        summary['overall_accuracy'] = overall_acc * 100.0

        # 5. per-category acc / f1 + macro scores
        scores_for_macro = {}
        rows = []
        cat_order = self._task_category()
        binary_set = self.task_category('binary_set')

        for cat in cat_order:
            sub = mcq_scored[mcq_scored['bench_category'] == cat]
            n_c = int(len(sub))
            if n_c > 0:
                acc_c = float(sub['hit'].mean())
            else:
                acc_c = 0.0

            f1_c = None
            if cat in binary_set and n_c > 0:
                f1_c = self._compute_f1_for_binary_df(sub, pos_label='yes')

            # macro: MCQ uses acc, binary uses F1
            scores_for_macro[cat] = (f1_c if cat in binary_set else acc_c)

            summary[f'{cat}_accuracy'] = acc_c * 100.0
            if cat in binary_set and f1_c is not None:
                summary[f'{cat}_f1'] = float(f1_c) * 100.0

            rows.append({
                'category': cat,
                'n': n_c,
                'acc': acc_c,
                'f1': f1_c,
            })

        per_category_df = pd.DataFrame(rows, columns=['category', 'n', 'acc', 'f1'])

        # 6. macro (6-term)
        macro_avg = self._macro_from_scores(scores_for_macro, force_divisor_6=False)
        if macro_avg is not None:
            summary['macro'] = float(macro_avg) * 100.0

        # 7. tabulate
        tab_keys = ', '.join(list(summary.keys()))
        tab_vals = ', '.join(
            [f'{v:.3f}' for v in summary.values() if isinstance(v, (int, float))]
        )
        summary['tabulated_keys'] = tab_keys
        summary['tabulated_results'] = tab_vals

        # 8. save pkl
        try:
            import pickle
            with open(result_file, 'wb') as f:
                pickle.dump(
                    {'mcq_scored': mcq_scored, 'per_category': per_category_df, 'summary': summary},
                    f
                )
            print(f'[save] result saved to {result_file}')
        except Exception as e:
            warnings.warn(f'[save] failed to save result to {result_file}: {e}')

        # 9. save extract_matching xlsx
        try:
            prefer_front = [
                'index', 'bench_category', 'category', 'category_agg',
                'question', 'prediction', 'pred_extracted',
                'answer', 'hit',
            ]
            merged = mcq_scored.copy()
            ordered_cols = (
                [c for c in prefer_front if c in merged.columns]
                + [c for c in merged.columns if c not in prefer_front]
            )
            merged = merged[ordered_cols]

            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='ALL', index=False)
            print(f'[save] extract & matching saved to {xlsx_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save extract xlsx to {xlsx_path}: {e}')

        # 10. save metrics tsv
        try:
            metric_rows = []
            for k, v in summary.items():
                if k in ('tabulated_keys', 'tabulated_results'):
                    continue
                metric_rows.append((k, v))
            acc_df = pd.DataFrame(metric_rows, columns=['metric', 'value'])

            metric_order = ['overall_accuracy', 'macro']
            for c in cat_order:
                metric_order.append(f'{c}_accuracy')
            for c in cat_order:
                if f'{c}_f1' in acc_df['metric'].values:
                    metric_order.append(f'{c}_f1')
            metric_order += [k for k in acc_df['metric'].tolist()
                             if k not in metric_order]

            acc_df = acc_df.set_index('metric').reindex(metric_order).dropna(subset=['value'])
            wide = acc_df.T
            wide.to_csv(acc_tsv_path, sep='\t', index=False, float_format='%.4f')

            print(f'[save] accuracy/F1/macro table saved to {acc_tsv_path}')
        except Exception as e:
            warnings.warn(f'[save] failed to save acc tsv to {acc_tsv_path}: {e}')

        print(f'[StareBench] summary: {summary}')
        return summary
