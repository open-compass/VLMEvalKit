"""Adapter that bridges VLMEvalKit's dataframe-based I/O with the official
OmniDocBench v1.6 evaluation pipeline (vendored under ``odb_v16_vendor``).

The official pipeline (``src.core.pipeline.run_config``) expects a ground-truth
JSON file plus a folder of per-page prediction markdown files, and writes its
results to ``./result`` relative to the current working directory. VLMEvalKit
instead stores the per-page GT JSON in the ``answer`` column of the dataset TSV
and the model output in the ``prediction`` column of the result file. This
module reconstructs the folder/JSON layout the official code needs, runs it in
an isolated working directory, and parses the official result back into a
VLMEvalKit-friendly ``pandas.DataFrame``.
"""

import contextlib
import json
import os
import os.path as osp
import shutil
import sys
import tempfile

import pandas as pd

from vlmeval.smp import dump, get_intermediate_file_path, get_logger, load

logger = get_logger('OmniDocBench_v1.6')

VENDOR_DIR = osp.join(osp.dirname(osp.abspath(__file__)), 'odb_v16_vendor')

# Official end-to-end leaderboard columns (see the OmniDocBench v1.6 README).
RESULT_COLUMNS = [
    'Overall', 'TextEdit', 'FormulaCDM', 'TableTEDS', 'TableTEDS-S', 'ReadOrderEdit',
]


@contextlib.contextmanager
def vendor_runtime(work_dir):
    """Expose the vendored official ``src`` package and chdir into ``work_dir``.

    The official pipeline imports its modules through the top-level ``src``
    package and writes outputs to ``./result`` relative to the cwd, so we
    isolate both ``sys.path``/``sys.modules`` and the working directory, and
    restore everything on exit.
    """
    old_cwd = os.getcwd()
    old_sys_path = list(sys.path)
    # Stash any pre-existing `src` modules so the vendored copy never collides
    # with (or leaks into) an unrelated top-level `src` package.
    stashed = {k: v for k, v in list(sys.modules.items()) if k == 'src' or k.startswith('src.')}
    for k in list(stashed):
        del sys.modules[k]
    sys.path.insert(0, VENDOR_DIR)
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_sys_path
        for k in [m for m in list(sys.modules) if m == 'src' or m.startswith('src.')]:
            del sys.modules[k]
        sys.modules.update(stashed)


def cdm_available():
    """Probe whether the CDM rendering toolchain is installed.

    CDM needs a full LaTeX render stack (TeX Live ``pdflatex`` + Ghostscript +
    ImageMagick). It cannot be a hard dependency of VLMEvalKit, so we detect it
    at runtime and gracefully degrade when missing.
    """
    if os.environ.get('OMNIDOCBENCH_DISABLE_CDM', '').lower() in ('1', 'true', 'yes'):
        return False
    if shutil.which('gs') is None:
        return False
    if shutil.which('magick') is None and shutil.which('convert') is None:
        return False
    if shutil.which('pdflatex') is None:
        return False
    return True


def _build_index_to_prediction(eval_data):
    preds = {}
    for _, row in eval_data.iterrows():
        preds[str(row['index'])] = '' if pd.isna(row['prediction']) else str(row['prediction'])
    return preds


def _materialize_inputs(eval_data, gt_df, work_dir):
    """Write the GT JSON file and per-page prediction markdown folder.

    Returns ``(gt_json_path, pred_folder, n_pages)``.
    """
    index_to_pred = _build_index_to_prediction(eval_data)
    pred_folder = osp.join(work_dir, 'pred')
    os.makedirs(pred_folder, exist_ok=True)

    gt_list = []
    load_fail = 0
    for _, row in gt_df.iterrows():
        try:
            page = json.loads(row['answer'])
        except (json.JSONDecodeError, TypeError):
            load_fail += 1
            continue
        gt_list.append(page)

        img_name = osp.basename(page['page_info']['image_path'])
        # Official _resolve_prediction_path looks for `{img_name[:-4]}.md`.
        stem = img_name[:-4] if img_name.lower().endswith(('.jpg', '.png')) else img_name
        prediction = index_to_pred.get(str(row['index']), '')
        with open(osp.join(pred_folder, f'{stem}.md'), 'w', encoding='utf-8') as f:
            f.write(prediction)

    if load_fail:
        logger.warning(f'Failed to parse {load_fail} GT answer rows as JSON; they were skipped.')

    gt_json_path = osp.join(work_dir, 'OmniDocBench_v1.6_gt.json')
    with open(gt_json_path, 'w', encoding='utf-8') as f:
        json.dump(gt_list, f, ensure_ascii=False)
    return gt_json_path, pred_folder, len(gt_list)


def _build_config(gt_json_path, pred_folder, match_method, use_cdm, match_workers):
    display_metrics = ['Edit_dist']
    if use_cdm:
        display_metrics.append('CDM')
    return {
        'end2end_eval': {
            'metrics': {
                'text_block': {'metric': ['Edit_dist']},
                'display_formula': {'metric': display_metrics, 'cdm_workers': match_workers},
                'table': {'metric': ['TEDS', 'Edit_dist'], 'teds_workers': match_workers},
                'reading_order': {'metric': ['Edit_dist']},
            },
            'dataset': {
                'dataset_name': 'end2end_dataset',
                'ground_truth': {'data_path': gt_json_path},
                'prediction': {'data_path': pred_folder},
                'match_method': match_method,
                'match_workers': match_workers,
            },
        }
    }


def _safe_get(mapping, *path):
    cur = mapping
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _to_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


def _summarize(metric_result, use_cdm):
    text_edit = _to_float(_safe_get(metric_result, 'text_block', 'all', 'Edit_dist', 'ALL_page_avg'))
    table_teds = _to_float(_safe_get(metric_result, 'table', 'page', 'TEDS', 'ALL'))
    table_teds_s = _to_float(_safe_get(metric_result, 'table', 'page', 'TEDS_structure_only', 'ALL'))
    reading_order = _to_float(_safe_get(metric_result, 'reading_order', 'all', 'Edit_dist', 'ALL_page_avg'))
    formula_cdm = _to_float(_safe_get(metric_result, 'display_formula', 'page', 'CDM', 'ALL')) if use_cdm else None

    row = {
        'TextEdit': round(text_edit, 4) if text_edit is not None else '-',
        'FormulaCDM': round(formula_cdm * 100, 2) if formula_cdm is not None else '-',
        'TableTEDS': round(table_teds * 100, 2) if table_teds is not None else '-',
        'TableTEDS-S': round(table_teds_s * 100, 2) if table_teds_s is not None else '-',
        'ReadOrderEdit': round(reading_order, 4) if reading_order is not None else '-',
    }

    # Official Overall = ((1 - TextEdit) * 100 + TableTEDS + FormulaCDM) / 3
    if text_edit is not None and table_teds is not None and formula_cdm is not None:
        row['Overall'] = round(((1 - text_edit) * 100 + table_teds * 100 + formula_cdm * 100) / 3, 2)
    else:
        row['Overall'] = '-'
        if text_edit is not None and table_teds is not None:
            # CDM-less reference number so the run is still informative; NOT the
            # official metric. Enable the CDM toolchain for a comparable score.
            row['Overall_no_CDM'] = round(((1 - text_edit) * 100 + table_teds * 100) / 2, 2)
            logger.warning(
                'CDM was not computed (toolchain missing or disabled); the official Overall '
                'cannot be reported. An unofficial CDM-less reference is provided as `Overall_no_CDM`.'
            )
    return row


def evaluate_v16(eval_file, tsv_path, **judge_kwargs):
    """Run the official OmniDocBench v1.6 end-to-end evaluation.

    Args:
        eval_file: Path to the VLMEvalKit prediction file (xlsx/tsv) with an
            ``index`` and ``prediction`` column.
        tsv_path: Path to the dataset TSV whose ``answer`` column holds the
            per-page OmniDocBench GT JSON.
        judge_kwargs: Accepts ``match_method`` (default ``quick_match``),
            ``enable_cdm`` (default: auto-detect), and ``nproc`` (match/CDM/TEDS
            workers).
    """
    match_method = judge_kwargs.get('match_method', os.environ.get('OMNIDOCBENCH_MATCH_METHOD', 'quick_match'))
    match_workers = int(judge_kwargs.get('nproc', judge_kwargs.get('match_workers', 8)) or 8)

    enable_cdm = judge_kwargs.get('enable_cdm', None)
    if enable_cdm is None:
        use_cdm = cdm_available()
    else:
        use_cdm = bool(enable_cdm) and cdm_available()
    if enable_cdm and not use_cdm:
        logger.warning('enable_cdm=True but the CDM toolchain was not detected; CDM will be skipped.')
    if not use_cdm:
        logger.warning(
            'Running OmniDocBench v1.6 without CDM. Install TeX Live + Ghostscript + ImageMagick '
            '(or use the official Docker image) for the official Overall metric.'
        )

    eval_data = load(eval_file)
    gt_df = load(tsv_path)

    work_dir = tempfile.mkdtemp(prefix='omnidocbench_v16_')
    try:
        gt_json_path, pred_folder, n_pages = _materialize_inputs(eval_data, gt_df, work_dir)
        logger.info(f'Prepared {n_pages} pages for OmniDocBench v1.6 evaluation (match={match_method}, cdm={use_cdm}).')

        cfg = _build_config(gt_json_path, pred_folder, match_method, use_cdm, match_workers)
        save_name = osp.basename(pred_folder) + '_' + match_method

        with vendor_runtime(work_dir):
            from src.core.pipeline import run_config
            run_config(cfg)

        result_dir = osp.join(work_dir, 'result')
        metric_result_path = osp.join(result_dir, f'{save_name}_metric_result.json')
        with open(metric_result_path, 'r', encoding='utf-8') as f:
            metric_result = json.load(f)

        # Persist the full official breakdown (per-attribute / per-page) as aux files.
        detailed_file = get_intermediate_file_path(eval_file, '_v16_metric_result', 'json')
        dump(metric_result, detailed_file)
        run_summary_path = osp.join(result_dir, f'{save_name}_run_summary.json')
        if osp.exists(run_summary_path):
            with open(run_summary_path, 'r', encoding='utf-8') as f:
                run_summary = json.load(f)
            dump(run_summary, get_intermediate_file_path(eval_file, '_v16_run_summary', 'json'))

        row = _summarize(metric_result, use_cdm)
        df = pd.DataFrame([row], index=['OmniDocBench_v1.6'])
        ordered = [c for c in RESULT_COLUMNS if c in df.columns] + [c for c in df.columns if c not in RESULT_COLUMNS]
        df = df[ordered]

        score_file = get_intermediate_file_path(eval_file, '_v16_score', 'csv')
        dump(df, score_file)
        logger.info(f'OmniDocBench v1.6 score saved to {score_file}')
        return df
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
