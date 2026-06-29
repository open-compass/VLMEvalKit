import copy
import json
import multiprocessing
import os
import queue
import random
import shutil
import time
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import Levenshtein
import evaluate
import pandas as pd
from tqdm import tqdm

from src.core.preprocess import strip_formula_tags
from src.core.registry import METRIC_REGISTRY
from src.runtime.concurrency import resolve_cdm_workers, resolve_teds_workers
from .cdm.modules.texlive_env import describe_tex_runtime
from .cdm_metric import CDM
from .table_metric import TEDS


def _as_sample_list(samples):
    if isinstance(samples, list):
        return samples
    return samples.samples


def _sample_matches_group(sample, group):
    select_flag = True
    for k, v in group.items():
        for gt_attribute in sample['gt_attribute']:
            if not gt_attribute:
                select_flag = False
            elif gt_attribute[k] != v:
                select_flag = False
    return select_flag


def get_groups(samples, group_info):
    group_samples = defaultdict(list)
    for sample in samples:
        group_samples['all'].append(sample)
        for group in group_info:
            if _sample_matches_group(sample, group):
                group_samples[str(group)].append(sample)
    return group_samples


def _safe_average(scores):
    if len(scores) > 0:
        return sum(scores) / len(scores)
    return 'NaN'


def _format_case_name(sample):
    img_id = str(sample.get('img_id') or '')
    parts = []
    gt_idx = sample.get('gt_idx')
    pred_idx = sample.get('pred_idx')
    if gt_idx not in (None, '', []):
        parts.append(f'gt={gt_idx}')
    if pred_idx not in (None, '', []):
        parts.append(f'pred={pred_idx}')
    if parts:
        return f'{img_id}[{", ".join(parts)}]'
    return img_id


def _build_case_record(sample, reason=None, **extra):
    record = {
        'case_name': _format_case_name(sample),
        'img_id': sample.get('img_id'),
    }
    if sample.get('gt_idx') not in (None, '', []):
        record['gt_idx'] = sample.get('gt_idx')
    if sample.get('pred_idx') not in (None, '', []):
        record['pred_idx'] = sample.get('pred_idx')
    if reason not in (None, ''):
        record['reason'] = str(reason)
    for key, value in extra.items():
        if value not in (None, '', []):
            record[key] = value
    return record


def _sort_case_records(records):
    return sorted(
        records,
        key=lambda item: (
            str(item.get('img_id', '')),
            str(item.get('gt_idx', '')),
            str(item.get('pred_idx', '')),
            str(item.get('reason', '')),
            str(item.get('case_name', '')),
        ),
    )


def _read_float_env(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _resolve_metric_timeout(metric_cfg, env_name, default):
    metric_cfg = metric_cfg or {}
    value = metric_cfg.get('timeout_sec', metric_cfg.get('teds_timeout_sec'))
    if value is None:
        value = os.getenv(env_name, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = float(default)
    if value <= 0:
        return None
    return value


def _resolve_timeout_input_dir():
    return os.getenv('OMNIDOCBENCH_TIMEOUT_INPUT_DIR', os.path.join(os.getcwd(), 'logs', 'timeout_inputs'))


def _dump_timeout_input(prefix, payload):
    try:
        out_dir = _resolve_timeout_input_dir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{prefix}_{uuid.uuid4().hex}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path
    except Exception as exc:
        print(f'[timeout-dump-error] prefix={prefix} error={exc}', flush=True)
        return ''


def _teds_pair_worker(pred, gt, result_queue):
    try:
        score = TEDS(structure_only=False).evaluate(pred, gt)
        score_structure_only = TEDS(structure_only=True).evaluate(pred, gt)
        result_queue.put(('success', (score, score_structure_only)))
    except Exception as exc:
        result_queue.put(('error', f'{type(exc).__name__}: {exc}'))


def _evaluate_teds_pair_with_timeout(pred, gt, timeout_sec):
    if timeout_sec is None:
        score = TEDS(structure_only=False).evaluate(pred, gt)
        score_structure_only = TEDS(structure_only=True).evaluate(pred, gt)
        return score, score_structure_only, None

    result_queue = multiprocessing.Queue()
    result_queue.cancel_join_thread()
    process = multiprocessing.Process(target=_teds_pair_worker, args=(pred, gt, result_queue))
    process.daemon = True
    try:
        process.start()
        status, payload = result_queue.get(timeout=timeout_sec)
        if status == 'success':
            score, score_structure_only = payload
            return score, score_structure_only, None
        return 0.0, 0.0, payload
    except queue.Empty:
        return 0.0, 0.0, f'timeout:{timeout_sec}'
    finally:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        try:
            result_queue.close()
        except Exception:
            pass


def _log_teds_timeout(sample, gt, pred, timeout_sec, reason):
    payload = {
        'stage': 'TEDS',
        'reason': reason,
        'timeout_sec': timeout_sec,
        'img_id': sample.get('img_id'),
        'gt_idx': sample.get('gt_idx'),
        'pred_idx': sample.get('pred_idx'),
        'gt_length': len(gt),
        'pred_length': len(pred),
        'gt_html': gt,
        'pred_html': pred,
    }
    dump_path = _dump_timeout_input('teds_timeout', payload)
    print(
        f'[TEDS-timeout] img_id={sample.get("img_id")} gt_idx={sample.get("gt_idx")} pred_idx={sample.get("pred_idx")} timeout={timeout_sec} reason={reason} path={dump_path} gt_len={len(gt)} pred_len={len(pred)}',
        flush=True,
    )


@METRIC_REGISTRY.register("TEDS")
class call_TEDS():
    def __init__(self, samples, metric_cfg=None):
        self.samples = samples
        self.metric_cfg = metric_cfg or {}
        self.max_workers = resolve_teds_workers(self.metric_cfg)
        self.timeout_sec = _resolve_metric_timeout(self.metric_cfg, 'OMNIDOCBENCH_TEDS_TIMEOUT_SEC', 120)
        self.debug_info = {
            'stage': 'TEDS',
            'workers': self.max_workers,
            'timeout_sec': self.timeout_sec,
            'sample_count': 0,
            'timeout_case_count': 0,
            'timeout_cases': [],
            'error_case_count': 0,
            'error_cases': [],
        }

    def evaluate(self, group_info=[], save_name='default'):
        samples = _as_sample_list(self.samples)
        group_scores = defaultdict(list)
        group_scores_structure_only = defaultdict(list)
        per_table_score = {}

        results = []
        worker_args = [(idx, sample, self.timeout_sec) for idx, sample in enumerate(samples)]
        self.debug_info['sample_count'] = len(worker_args)
        if self.max_workers <= 1 or len(worker_args) <= 1:
            progress_bar = tqdm(worker_args, total=len(worker_args), ascii=True, ncols=140, desc='TEDS')
            results = [_score_single_teds_sample(args) for args in progress_bar]
            progress_bar.close()
        else:
            print(f'[TEDS] workers={self.max_workers} samples={len(worker_args)} timeout={self.timeout_sec}', flush=True)
            progress_bar = tqdm(total=len(worker_args), ascii=True, ncols=140, desc='TEDS')
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {executor.submit(_score_single_teds_sample, args): args[0] for args in worker_args}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        sample = samples[idx]
                        print(f'TEDS score error for table {sample.get("gt_idx")} in {sample.get("img_id")}: {exc}. The score is set to 0.', flush=True)
                        error_record = _build_case_record(sample, reason=f'{type(exc).__name__}: {exc}')
                        results.append({
                            'original_index': idx,
                            'score': 0.0,
                            'score_structure_only': 0.0,
                            'status': 'error',
                            'case_record': error_record,
                        })
                    finally:
                        progress_bar.update(1)
            progress_bar.close()

        results.sort(key=lambda item: item['original_index'])
        timeout_cases = []
        error_cases = []
        for result in results:
            if result.get('status') == 'timeout' and result.get('case_record'):
                timeout_cases.append(result['case_record'])
            elif result.get('status') == 'error' and result.get('case_record'):
                error_cases.append(result['case_record'])
            sample = samples[result['original_index']]
            score = result['score']
            score_structure_only = result['score_structure_only']
            group_scores['all'].append(score)
            group_scores_structure_only['all'].append(score_structure_only)
            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['TEDS'] = score
            sample['metric']['TEDS_structure_only'] = score_structure_only
            per_table_score[sample['img_id'] + '_' + str(sample['gt_idx'])] = {
                'TEDS': score,
                'TEDS_structure_only': score_structure_only,
            }
            for group in group_info:
                if _sample_matches_group(sample, group):
                    group_scores[str(group)].append(score)
                    group_scores_structure_only[str(group)].append(score_structure_only)

        with open(f'./result/{save_name}_per_table_TEDS.json', 'w', encoding='utf-8') as f:
            json.dump(per_table_score, f, indent=4, ensure_ascii=False)

        result = {}
        for group_name, scores in group_scores.items():
            result[group_name] = _safe_average(scores)
            if result[group_name] == 'NaN':
                print(f'Warning: Empyty matched samples for {group_name}.')

        structure_only_result = {}
        for group_name, scores in group_scores_structure_only.items():
            structure_only_result[group_name] = _safe_average(scores)
            if structure_only_result[group_name] == 'NaN':
                print(f'Warning: Empyty matched samples for {group_name}.')

        self.debug_info = {
            'stage': 'TEDS',
            'workers': self.max_workers,
            'timeout_sec': self.timeout_sec,
            'sample_count': len(worker_args),
            'timeout_case_count': len(timeout_cases),
            'timeout_cases': _sort_case_records(timeout_cases),
            'error_case_count': len(error_cases),
            'error_cases': _sort_case_records(error_cases),
        }

        return self.samples, {'TEDS': result, 'TEDS_structure_only': structure_only_result}


@METRIC_REGISTRY.register("BLEU")
class call_BLEU():
    def __init__(self, samples, metric_cfg=None):
        self.samples = samples
        self.metric_cfg = metric_cfg or {}

    def evaluate(self, group_info=[], save_name='default'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1, 1e8))
            bleu_results = bleu.compute(predictions=predictions, references=references)
            result[group_name] = bleu_results["bleu"]

        return self.samples, {'BLEU': result}


@METRIC_REGISTRY.register("METEOR")
class call_METEOR():
    def __init__(self, samples, metric_cfg=None):
        self.samples = samples
        self.metric_cfg = metric_cfg or {}

    def evaluate(self, group_info=[], save_name='default'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            meteor = evaluate.load('meteor', keep_in_memory=True, experiment_id=random.randint(1, 1e8))
            meteor_results = meteor.compute(predictions=predictions, references=references)
            result[group_name] = meteor_results['meteor']

        return self.samples, {'METEOR': result}


@METRIC_REGISTRY.register("Edit_dist")
class call_Edit_dist():
    def __init__(self, samples, metric_cfg=None):
        self.samples = samples
        self.metric_cfg = metric_cfg or {}

    def evaluate(self, group_info=[], save_name='default'):
        samples = self.samples
        for sample in samples:
            img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') else '_'.join(sample['img_id'].split('_')[:-1])
            sample['image_name'] = img_name
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            upper_len = max(len(pred), len(gt))
            sample['upper_len'] = upper_len
            if len(pred) > 0 or len(gt) > 0:
                edit_dist = Levenshtein.distance(pred, gt)
                if not sample.get('metric'):
                    sample['metric'] = {}
                sample['metric']['Edit_dist'] = edit_dist / upper_len
                sample['Edit_num'] = edit_dist

        saved_samples = _as_sample_list(samples)
        if not saved_samples:
            return samples, {'Edit_dist': {'ALL_page_avg': 'NaN'}}

        df = pd.DataFrame(saved_samples)
        up_total_avg = df.groupby("image_name").apply(lambda x: x['Edit_num'].sum() / x['upper_len'].sum())
        all_total_avg = df['Edit_num'].sum() / df['upper_len'].sum()
        per_img_score = up_total_avg.to_dict()
        with open(f'./result/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)

        edit_whole = all_total_avg
        df['ratio'] = df['Edit_num'] / df['upper_len']
        edit_sample_avg = df['ratio'].mean()
        return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean(), 'edit_whole': edit_whole, 'edit_sample_avg': edit_sample_avg}}


def _strip_cdm_math_wrappers(text):
    text = str(text or '').strip()
    text = text.lstrip("$$").rstrip("$$").strip()
    text = text.lstrip("$").rstrip("$").strip()
    if text.startswith('\[') and text.endswith('\]'):
        text = text[2:-2].strip()
    if text.startswith('\(') and text.endswith('\)'):
        text = text[2:-2].strip()
    return strip_formula_tags(text).strip()


def _score_single_teds_sample(args):
    idx, sample, timeout_sec = args
    gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
    pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
    score = 0.0
    score_structure_only = 0.0
    reason = None
    status = 'ok'
    case_record = None
    try:
        score, score_structure_only, reason = _evaluate_teds_pair_with_timeout(pred, gt, timeout_sec)
        if reason and str(reason).startswith('timeout:'):
            _log_teds_timeout(sample, gt, pred, timeout_sec, reason)
            status = 'timeout'
            case_record = _build_case_record(
                sample,
                reason=reason,
                timeout_sec=timeout_sec,
                gt_length=len(gt),
                pred_length=len(pred),
            )
        elif reason:
            print(f'TEDS score error for table {sample.get("gt_idx")} in {sample.get("img_id")}: {reason}. The score is set to 0.', flush=True)
            status = 'error'
            case_record = _build_case_record(sample, reason=reason)
    except Exception as exc:
        print(f'TEDS score error for table {sample.get("gt_idx")} in {sample.get("img_id")}: {exc}. The score is set to 0.', flush=True)
        status = 'error'
        case_record = _build_case_record(sample, reason=f'{type(exc).__name__}: {exc}')
    return {
        'original_index': idx,
        'score': score,
        'score_structure_only': score_structure_only,
        'status': status,
        'case_record': case_record,
    }


def _process_single_cdm_sample(args):
    idx, sample, output_root, group_info = args
    cal_cdm = CDM(output_root=output_root)
    sample_copy = copy.deepcopy(sample)
    sample_copy['img_id_cdm'] = str(idx)
    sample_context = {
        'img_id': sample_copy.get('img_id'),
        'gt_idx': sample_copy.get('gt_idx'),
        'pred_idx': sample_copy.get('pred_idx'),
    }

    gt_cdm = sample_copy.get('gt_cdm', sample_copy['gt'])
    pred_cdm = sample_copy.get('pred_cdm', sample_copy['pred'])
    gt_cdm = _strip_cdm_math_wrappers(gt_cdm)
    pred_cdm = pred_cdm.split("```latex")[-1].split("```")[0]
    pred_cdm = _strip_cdm_math_wrappers(pred_cdm)
    pred_cdm_alt = sample_copy.get('pred_cdm_alt', '')
    if pred_cdm_alt:
        pred_cdm_alt = pred_cdm_alt.split("```latex")[-1].split("```")[0]
        pred_cdm_alt = _strip_cdm_math_wrappers(pred_cdm_alt)
    sample_copy['gt_cdm'] = gt_cdm
    sample_copy['pred_cdm'] = pred_cdm
    if pred_cdm_alt:
        sample_copy['pred_cdm_alt'] = pred_cdm_alt

    cdm_metrics = cal_cdm.evaluate(
        gt_cdm,
        pred_cdm,
        sample_copy['img_id_cdm'],
        sample_context=sample_context,
    )
    cdm_score = cdm_metrics["F1_score"]
    cdm_error = cdm_metrics.get('cdm_eval_error')
    if pred_cdm_alt:
        cdm_metrics_alt = cal_cdm.evaluate(
            gt_cdm,
            pred_cdm_alt,
            sample_copy['img_id_cdm'] + '_alt',
            sample_context=sample_context,
        )
        cdm_score_alt = cdm_metrics_alt["F1_score"]
        if cdm_score_alt > cdm_score:
            cdm_score = cdm_score_alt
            sample_copy['pred_cdm'] = pred_cdm_alt
            cdm_error = cdm_metrics_alt.get('cdm_eval_error')

    if not sample_copy.get('metric'):
        sample_copy['metric'] = {}
    sample_copy['metric']['CDM'] = cdm_score

    matched_groups = []
    for group in group_info:
        if _sample_matches_group(sample_copy, group):
            matched_groups.append(str(group))

    return {
        'sample': sample_copy,
        'cdm_score': cdm_score,
        'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
        'matched_groups': matched_groups,
        'original_index': idx,
        'cdm_error': cdm_error,
    }


@METRIC_REGISTRY.register("CDM")
class call_CDM():
    def __init__(self, samples, metric_cfg=None):
        self.samples = samples
        self.metric_cfg = metric_cfg or {}
        self.max_workers = resolve_cdm_workers(self.metric_cfg)
        self.debug_info = {
            'stage': 'CDM',
            'workers': self.max_workers,
            'sample_count': 0,
            'timeout_case_count': 0,
            'timeout_cases': [],
            'exception_case_count': 0,
            'exception_cases': [],
        }

    def evaluate(self, group_info=[], save_name='default', max_workers=None):
        group_scores = defaultdict(list)
        output_root = f"result/{save_name}/CDM"
        if os.path.isdir(output_root):
            shutil.rmtree(output_root, ignore_errors=True)
        os.makedirs(output_root, exist_ok=True)
        tex_runtime = describe_tex_runtime()
        print(f"[CDM] texlive_root={tex_runtime['texlive_root']}")
        print(f"[CDM] texlive_bin={tex_runtime['texlive_bin_dir']}")
        print(f"[CDM] pdflatex={tex_runtime['pdflatex']}")
        print(f"[CDM] pdflatex_version={tex_runtime['pdflatex_version']}")
        print(f"[CDM] kpsewhich={tex_runtime['kpsewhich']}")
        print(f"[CDM] kpsewhich_texmfroot={tex_runtime['kpsewhich_texmfroot']}")
        print(f"[CDM] texmfcnf={tex_runtime['texmfcnf']}")
        print(f"[CDM] cjk_font={tex_runtime['cjk_font']}")
        print(f"[CDM] cjk_sty={tex_runtime['cjk_sty']}")
        print(f"[CDM] cjk_font_fd={tex_runtime['cjk_font_fd']}")
        original_samples = _as_sample_list(self.samples)
        worker_args = [(idx, sample, output_root, group_info) for idx, sample in enumerate(original_samples)]
        per_sample_score = {}
        cdm_samples = []
        results = []
        resolved_workers = max(1, max_workers or self.max_workers)
        print(f'[CDM] workers={resolved_workers} samples={len(worker_args)}')
        self.debug_info['sample_count'] = len(worker_args)

        if resolved_workers <= 1 or len(worker_args) <= 1:
            progress_bar = tqdm(worker_args, total=len(worker_args), ascii=True, ncols=140, desc='CDM')
            for args in progress_bar:
                idx = args[0]
                try:
                    results.append(_process_single_cdm_sample(args))
                except Exception as exc:
                    print(f'Sample {idx} generated an exception: {exc}')
                    sample_copy = copy.deepcopy(original_samples[idx])
                    sample_copy['img_id_cdm'] = str(idx)
                    if not sample_copy.get('metric'):
                        sample_copy['metric'] = {}
                    sample_copy['metric']['CDM'] = 0.0
                    error_message = f'{type(exc).__name__}: {exc}'
                    results.append({
                        'sample': sample_copy,
                        'cdm_score': 0.0,
                        'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
                        'matched_groups': [],
                        'original_index': idx,
                        'cdm_error': error_message,
                    })
            progress_bar.close()
        else:
            progress_bar = tqdm(total=len(worker_args), ascii=True, ncols=140, desc='CDM')
            with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                future_to_idx = {executor.submit(_process_single_cdm_sample, args): args[0] for args in worker_args}
                for future in as_completed(future_to_idx):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as exc:
                        idx = future_to_idx[future]
                        print(f'Sample {idx} generated an exception: {exc}')
                        sample_copy = copy.deepcopy(original_samples[idx])
                        sample_copy['img_id_cdm'] = str(idx)
                        if not sample_copy.get('metric'):
                            sample_copy['metric'] = {}
                        sample_copy['metric']['CDM'] = 0.0
                        error_message = f'{type(exc).__name__}: {exc}'
                        results.append({
                            'sample': sample_copy,
                            'cdm_score': 0.0,
                            'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
                            'matched_groups': [],
                            'original_index': idx,
                            'cdm_error': error_message,
                        })
                    finally:
                        progress_bar.update(1)
            progress_bar.close()

        results.sort(key=lambda x: x['original_index'])
        exception_cases = []
        for result in results:
            sample = result['sample']
            cdm_score = result['cdm_score']
            sample_key = result['sample_key']
            matched_groups = result['matched_groups']
            if result.get('cdm_error'):
                exception_cases.append(_build_case_record(sample, reason=result['cdm_error']))
            cdm_samples.append(sample)
            per_sample_score[sample_key] = cdm_score
            group_scores['all'].append(cdm_score)
            for group_name in matched_groups:
                group_scores[group_name].append(cdm_score)

        with open(f'./result/{save_name}_per_sample_CDM.json', 'w', encoding='utf-8') as f:
            json.dump(per_sample_score, f, indent=4, ensure_ascii=False)

        with open(f'result/{save_name}_result.json', 'w', encoding='utf-8') as f:
            json.dump(cdm_samples, f, indent=4, ensure_ascii=False)

        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empty matched samples for {group_name}.')

        self.debug_info = {
            'stage': 'CDM',
            'workers': resolved_workers,
            'sample_count': len(worker_args),
            'timeout_case_count': 0,
            'timeout_cases': [],
            'exception_case_count': len(exception_cases),
            'exception_cases': _sort_case_records(exception_cases),
        }

        return cdm_samples, {'CDM': result}
