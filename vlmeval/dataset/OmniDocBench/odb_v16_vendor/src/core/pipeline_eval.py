from src.core.metrics import get_full_labels_results, get_page_split, show_result
from src.core.registry import EVAL_TASK_REGISTRY, METRIC_REGISTRY
from src.runtime.eval_report import (
    build_eval_run_report,
    build_metric_page_denominators,
    format_eval_run_report,
    format_runtime_environment_log,
    format_stage_execution_log,
)
import json
import os


@EVAL_TASK_REGISTRY.register("end2end_eval")
class End2EndEval():
    def __init__(self, dataset, metrics_list, page_info_path, save_name):
        result_all = {}
        page_denominators = {}
        page_info = {}
        if os.path.isdir(page_info_path):
            md_flag = True
        else:
            md_flag = False
        if not md_flag:
            with open(page_info_path, 'r') as f:
                pages = json.load(f)

            for page in pages:
                img_path = os.path.basename(page['page_info']['image_path'])
                page_info[img_path] = page['page_info']['page_attribute']

        os.makedirs("./result", exist_ok=True)

        for element in metrics_list.keys():
            result = {}
            metric_debug = {}
            metric_cfg = metrics_list[element]
            group_info = metric_cfg.get('group', [])
            samples = dataset.samples[element]
            for metric in metric_cfg['metric']:
                metric_cls = METRIC_REGISTRY.get(metric)
                metric_runner = metric_cls(samples, metric_cfg)
                samples, result_s = metric_runner.evaluate(group_info, f"{save_name}_{element}")
                if result_s:
                    result.update(result_s)
                debug_info = getattr(metric_runner, 'debug_info', None)
                if debug_info:
                    metric_debug[metric] = debug_info
            if result:
                print(f'【{element}】')
                show_result(result)
            result_all[element] = {}

            if md_flag:
                group_result = {}
                page_result = {}
            else:
                group_result = get_full_labels_results(samples)
                page_result = get_page_split(
                    samples,
                    page_info,
                    gt_page_names=getattr(dataset, 'gt_pages_by_element', {}).get(element),
                    expected_metrics=metric_cfg.get('metric', []),
                )
            result_all[element] = {
                'all': result,
                'group': group_result,
                'page': page_result,
            }
            if metric_debug:
                result_all[element]['metric_debug'] = metric_debug

            if isinstance(samples, list):
                saved_samples = samples
            else:
                saved_samples = samples.samples
            page_denominators[element] = build_metric_page_denominators(
                saved_samples,
                gt_page_names=getattr(dataset, 'gt_pages_by_element', {}).get(element),
                expected_metrics=metric_cfg.get('metric', []),
            )
            try:
                with open(f'./result/{save_name}_{element}_result.json', 'w', encoding='utf-8') as f:
                    json.dump(saved_samples, f, indent=4, ensure_ascii=False)
            except TypeError as e:
                print(f"JSON 序列化错误: {e}")
                print("请检查 saved_samples 中是否包含非 JSON 可序列化的数据类型")

                def find_non_serializable(data):
                    if isinstance(data, dict):
                        for k, v in data.items():
                            try:
                                json.dumps(v)
                            except TypeError:
                                print(f"键 '{k}' 包含不可序列化的值: {v} (类型: {type(v)})")
                                find_non_serializable(v)
                    elif isinstance(data, (list, tuple)):
                        for i, item in enumerate(data):
                            try:
                                json.dumps(item)
                            except TypeError:
                                print(f"索引 {i} 包含不可序列化的项: {item} (类型: {type(item)})")
                                find_non_serializable(item)

                find_non_serializable(saved_samples)

        match_debug_info = getattr(dataset, 'match_debug_info', None)
        if match_debug_info:
            result_all['match_debug'] = match_debug_info

        with open(f'./result/{save_name}_metric_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_all, f, indent=4, ensure_ascii=False)

        try:
            run_report = build_eval_run_report(save_name, result_all, page_denominators)
            run_report_path = f'./result/{save_name}_run_summary.json'
            runtime_environment_json_path = f'./result/{save_name}_runtime_environment.json'
            runtime_environment_log_path = f'./result/{save_name}_runtime_environment.log'
            stage_execution_json_path = f'./result/{save_name}_stage_execution.json'
            stage_execution_log_path = f'./result/{save_name}_stage_execution.log'
            with open(run_report_path, 'w', encoding='utf-8') as f:
                json.dump(run_report, f, indent=4, ensure_ascii=False)
            with open(runtime_environment_json_path, 'w', encoding='utf-8') as f:
                json.dump(run_report.get('runtime_environment', {}), f, indent=4, ensure_ascii=False)
            with open(runtime_environment_log_path, 'w', encoding='utf-8') as f:
                f.write(format_runtime_environment_log(run_report.get('runtime_environment', {}), save_name=save_name))
            with open(stage_execution_json_path, 'w', encoding='utf-8') as f:
                json.dump(run_report.get('stage_execution', {}), f, indent=4, ensure_ascii=False)
            with open(stage_execution_log_path, 'w', encoding='utf-8') as f:
                f.write(format_stage_execution_log(run_report.get('stage_execution', {}), save_name=save_name))

            print(f'========== RUNTIME_ENVIRONMENT {save_name} ==========')
            print(format_runtime_environment_log(run_report.get('runtime_environment', {}), save_name=save_name))
            print(f'========== END_RUNTIME_ENVIRONMENT {save_name} ==========')
            print(f'========== STAGE_EXECUTION {save_name} ==========')
            print(format_stage_execution_log(run_report.get('stage_execution', {}), save_name=save_name))
            print(f'========== END_STAGE_EXECUTION {save_name} ==========')
            print(f'========== FINAL_EVAL_RUN_REPORT {save_name} ==========')
            print(format_eval_run_report(run_report))
            print(f'========== END_FINAL_EVAL_RUN_REPORT {save_name} ==========')
            print(f'[final-eval-run-report] saved to {run_report_path}')
            print(f'[runtime-environment-json] saved to {runtime_environment_json_path}')
            print(f'[runtime-environment-log] saved to {runtime_environment_log_path}')
            print(f'[stage-execution-json] saved to {stage_execution_json_path}')
            print(f'[stage-execution-log] saved to {stage_execution_log_path}')
        except Exception as exc:
            print(f'[final-eval-run-report] failed to build summary for {save_name}: {type(exc).__name__}: {exc}', flush=True)
