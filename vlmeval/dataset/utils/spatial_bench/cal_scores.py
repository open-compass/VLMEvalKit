import re
import ast
import math
import numpy as np
import pandas as pd
import warnings

from collections import OrderedDict

from .matching_func import can_match_na, can_match_option


def exact_match(pred: str, target: str) -> float:
    pred = str(pred).strip().lower()
    target = str(target).strip().lower()
    return 1. if pred == target else 0.


def abs_dist_norm(pred: float, target: float) -> float:
    if target == 0.0:
        return abs(pred - target)
    else:
        return abs((pred - target) / target)


def mean_relative_accuracy(
    pred: float,
    target: float,
    start: float = 0.5,
    end: float = 0.95,
    interval: float = 0.05,
) -> float:
    # TODO：check this, should be + 1, but in vsi code this is + 2
    num_pts = int((end - start) / interval + 2)
    conf_intervs = np.linspace(start, end, num_pts)
    err = abs_dist_norm(pred, target)
    ok = (err <= (1 - conf_intervs)).astype(float)
    return float(ok.mean())


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _safe_len_candidates(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, list):
        return len(val)
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return len(parsed)
        except Exception:
            return None
    return None


def _ensure_options_count_row(row, default_choices=4):
    n = None
    if 'candidates' in row:
        n = _safe_len_candidates(row['candidates'])
    elif 'options' in row:
        n = _safe_len_candidates(row['options'])
    return n if (isinstance(n, int) and n > 0) else default_choices


def compute_mcq_score(df: pd.DataFrame) -> pd.DataFrame:
    preds_extracted, acc = [], []
    for _, r in df.iterrows():
        pred_raw = str(r['prediction'])
        gt_raw = str(r['answer']).strip()

        pred = can_match_option(pred_raw)
        gt = can_match_option(gt_raw)

        preds_extracted.append(pred)
        acc.append(exact_match(pred, gt))

    df = df.copy()
    df['pred_extracted'] = preds_extracted
    df['hit'] = acc
    return df


def compute_na_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Mean Relative Accuracy (MRA) for numerical-answer (NA) items,
    following the VSI codebase.

    Definition:
        For prediction ŷ, ground-truth y, and confidence threshold θ,
        the relative accuracy is 1[ |ŷ - y| / |y| < 1 - θ ].
        MRA averages this relative accuracy over θ ∈ {0.50, 0.55, ..., 0.95}.

    Reference:
        Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces. (https://arxiv.org/pdf/2412.14171)  # noqa: E501
    """
    preds_extracted, mra_scores = [], []

    for _, r in df.iterrows():
        pred_num = can_match_na(str(r['prediction']))
        gt_num = to_float(r['answer'])

        preds_extracted.append(pred_num)

        if pred_num is None or gt_num is None or math.isnan(gt_num):
            mra_scores.append(0.0)          # WORST_CASE
        else:
            mra_scores.append(mean_relative_accuracy(pred_num, gt_num, .5, .95, .05))

    df = df.copy()
    df['pred_extracted'] = preds_extracted
    df['MRA:.5:.95:.05'] = mra_scores
    return df


def compute_caa_score(df_all: pd.DataFrame, default_choices: int = 4) -> float:
    """
    Compute Class-Adjusted Accuracy (CAA) for Multiple Choice Questions.

    Definition:
    For each item i with n_i options and correctness indicator X_i ∈ {0, 1},
        CAA = (Σ_i X_i - Σ_i (1 / n_i)) / (N - Σ_i (1 / n_i))
            - N   : total number of evaluated items.
            - X_i : 1 if the prediction for item i is correct, otherwise 0.
            - n_i : number of answer options for item i.

    Reference:
        SITE: Towards Spatial Intelligence Thorough Evaluation. (https://arxiv.org/pdf/2505.05456)

    """
    if len(df_all) == 0:
        return 0.0
    n_list = df_all.apply(lambda r: _ensure_options_count_row(r, default_choices), axis=1)
    xi = df_all['hit'].astype(int)
    N = len(df_all)
    sum_Xi = xi.sum()
    sum_1_ni = (1.0 / n_list).sum()
    denom = N - sum_1_ni
    return float((sum_Xi - sum_1_ni) / denom) if denom != 0 else 0.0


def eval_mcq_core(
    *,
    load_fn,
    eval_file: str,
    score_fn,
    group_col: str | list[str] = 'category',
    order: list[str] | dict[str, list[str]] | None = None,
    dataset_name: str = 'MCQ'
):
    suffix = eval_file.split('.')[-1]
    result_file = eval_file.replace(f'.{suffix}', '_result.pkl')
    base_no_suffix = eval_file[:-(len(suffix) + 1)]
    xlsx_path = f"{base_no_suffix}_extract_matching.xlsx"
    acc_tsv_path = f"{base_no_suffix}_acc.tsv"

    data = load_fn(eval_file)
    if 'index' in data.columns:
        data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]

    mcq_scored = score_fn(data.copy())

    # ---------- group_cols / order_map ----------
    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = list(group_col)

    if isinstance(order, dict) or order is None:
        order_map: dict[str, list[str]] = order or {}
    else:
        order_map = {group_cols[0]: order}

    summary = OrderedDict()
    overall_acc = float(mcq_scored['hit'].mean()) if len(mcq_scored) else 0.0
    summary['overall'] = overall_acc * 100.0

    # ---------- category && tasks ----------
    for gc in group_cols:
        if gc not in mcq_scored.columns:
            continue

        preferred = order_map.get(gc, []) or []
        present = list(mcq_scored[gc].dropna().unique().tolist())
        remain = [c for c in present if c not in preferred]
        cat_order = preferred + remain

        prefix = '' if len(group_cols) == 1 else f'{gc}.'

        for cat in cat_order:
            sub = mcq_scored[mcq_scored[gc] == cat]
            if len(sub):
                acc = float(sub['hit'].mean()) * 100.0
                summary[f'{prefix}{cat}_accuracy'] = acc

    tab_keys = ", ".join(list(summary.keys()))
    tab_vals = ", ".join([f"{v:.3f}" for v in summary.values()])
    summary['tabulated_keys'] = tab_keys
    summary['tabulated_results'] = tab_vals

    # ---------- pkl ----------
    try:
        import pickle
        with open(result_file, 'wb') as f:
            pickle.dump({'mcq_scored': mcq_scored, 'summary': summary}, f)
        print(f"[save] result saved to {result_file}")
    except Exception as e:
        warnings.warn(f"[save] failed to save result to {result_file}: {e}")

    # ---------- extract_matching.xlsx ----------
    try:
        prefer_front = [
            'index', 'question_type',
            group_cols[0] if group_cols else None,
            'prediction', 'pred_extracted', 'answer', 'hit'
        ]
        prefer_front = [c for c in prefer_front if c is not None]

        merged = mcq_scored.copy()
        ordered_cols = [c for c in prefer_front if c in merged.columns] + \
                       [c for c in merged.columns if c not in prefer_front]
        merged = merged[ordered_cols]
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            merged.to_excel(writer, sheet_name="ALL", index=False)
        print(f"[save] extract & matching saved to {xlsx_path}")
    except Exception as e:
        warnings.warn(f"[save] failed to save extract xlsx to {xlsx_path}: {e}")

    # ---------- acc.tsv ----------
    try:
        acc_df = pd.DataFrame(
            [(k, v) for k, v in summary.items()
             if k not in ('tabulated_keys', 'tabulated_results')],
            columns=['metric', 'value']
        )

        metric_order = ['overall']

        for gc in group_cols:
            preferred = order_map.get(gc, []) or []
            prefix = '' if len(group_cols) == 1 else f'{gc}.'
            metric_order += [f'{prefix}{c}_accuracy' for c in preferred]

        metric_order += [k for k in acc_df['metric'].tolist()
                         if k not in metric_order]

        acc_df = acc_df.set_index('metric').reindex(metric_order).dropna(subset=['value'])
        wide = acc_df.T
        wide.to_csv(acc_tsv_path, sep='\t', index=False, float_format='%.4f')

        print(f"[save] accuracy table saved to {acc_tsv_path}")
    except Exception as e:
        warnings.warn(f"[save] failed to save acc tsv to {acc_tsv_path}: {e}")

    print(f"[{dataset_name}] summary: {summary}")
    return summary
