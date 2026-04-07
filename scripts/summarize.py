import argparse
import json
from numbers import Real
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from vlmeval.smp import collect_run_benchmark_report, load_run_status


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, action='append', required=True)
    parser.add_argument('--data', nargs='+', default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.verbose and len(args.work_dir) != 1:
        parser.error('--verbose only supports a single --work-dir')
    if args.data is not None:
        args.data = list(dict.fromkeys(args.data))
    return args


def resolve_run_dir(work_dir: Path) -> Path:
    work_dir = work_dir.resolve()
    if (work_dir / 'status.json').exists():
        return work_dir

    candidates = sorted(
        path for path in work_dir.iterdir()
        if path.is_dir() and (path / 'status.json').exists()
    )
    if not candidates:
        raise FileNotFoundError(f'No status.json found in {work_dir} or its direct child directories.')
    return candidates[-1]


def format_sigfig(value):
    if value is None or isinstance(value, bool):
        return '-'
    if isinstance(value, Real):
        try:
            if pd.isna(value):
                return '-'
        except Exception:
            pass
        return f'{float(value):.4g}'
    return str(value)


def format_fail_rate(failed, total):
    if failed is None or total is None or total <= 0:
        return '-'
    return f'{failed / total * 100:.2f}% ({failed}/{total})'


def format_metric_field(value):
    if value is None:
        return '-'
    if isinstance(value, Real) and not isinstance(value, bool):
        return format_sigfig(value)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def truncate_error_message(value, max_length=120):
    value = value or '-'
    if value != '-' and len(str(value)) > max_length:
        return f'{str(value)[:max_length]}...'
    return value


def iter_primary_metric_rows(row):
    primary_metric = row.get('primary_metric')
    primary_metric_value = row.get('primary_metric_value')

    if isinstance(primary_metric, (list, tuple)):
        if not primary_metric:
            return [(None, None)]
        value_map = primary_metric_value if isinstance(primary_metric_value, dict) else {}
        return [(metric_name, value_map.get(metric_name)) for metric_name in primary_metric]

    return [(primary_metric, primary_metric_value)]


def resolve_model_name(run_dir: Path) -> str:
    run_status = load_run_status(run_dir)
    return str(run_status.get('model_name') or run_dir.parent.name)


def dedupe_column_names(names):
    counts = {}
    deduped_names = []
    for name in names:
        name = str(name)
        counts[name] = counts.get(name, 0) + 1
        deduped_names.append(name if counts[name] == 1 else f'{name}#{counts[name]}')
    return deduped_names


def build_verbose_report_rows(rows):
    report_rows = []
    for row in rows:
        metric_rows = iter_primary_metric_rows(row)
        eval_error = truncate_error_message(row['eval_error'])

        for idx, (primary_metric, primary_metric_value) in enumerate(metric_rows):
            report_rows.append({
                'benchmark': row['benchmark'] if idx == 0 else '',
                'infer_fail_rate': format_fail_rate(row['infer_failed'], row['infer_total']) if idx == 0 else '',
                'judge_fail_rate': format_fail_rate(row['judge_failed'], row['judge_total']) if idx == 0 else '',
                'primary_metric': format_metric_field(primary_metric),
                'primary_metric_value': format_metric_field(primary_metric_value),
                'skip_reason': (row['skip_reason'] or '-') if idx == 0 else '',
                'eval_error': eval_error if idx == 0 else '',
            })
    return report_rows


def build_summary_rows(run_reports, benchmark_order=None):
    model_columns = dedupe_column_names([report['model_name'] for report in run_reports])
    merged_rows = {}
    benchmark_filter = set(benchmark_order) if benchmark_order is not None else None

    for model_column, report in zip(model_columns, run_reports):
        for row in report['rows']:
            benchmark = row['benchmark']
            if benchmark_filter is not None and benchmark not in benchmark_filter:
                continue
            for primary_metric, primary_metric_value in iter_primary_metric_rows(row):
                row_key = (benchmark, format_metric_field(primary_metric))
                if row_key not in merged_rows:
                    merged_rows[row_key] = {
                        'benchmark': benchmark,
                        'primary_metric': format_metric_field(primary_metric),
                    }
                merged_rows[row_key][model_column] = format_metric_field(primary_metric_value)

    benchmark_to_keys = {}
    for row_key in merged_rows:
        benchmark_to_keys.setdefault(row_key[0], []).append(row_key)

    if benchmark_order is None:
        ordered_benchmarks = list(benchmark_to_keys.keys())
    else:
        ordered_benchmarks = [benchmark for benchmark in benchmark_order if benchmark in benchmark_to_keys]

    ordered_columns = ['benchmark', 'primary_metric', *model_columns]
    summary_rows = []
    for benchmark in ordered_benchmarks:
        row_keys = benchmark_to_keys[benchmark]
        has_primary_metric = any(primary_metric != '-' for _, primary_metric in row_keys)
        if has_primary_metric:
            row_keys = [row_key for row_key in row_keys if row_key[1] != '-']
        for row_key in row_keys:
            row = merged_rows[row_key]
            summary_rows.append({column: row.get(column, '-') for column in ordered_columns})
    return summary_rows, ordered_columns


def print_csv_and_table(rows, columns=None):
    if columns is None:
        columns = list(rows[0].keys())
    df = pd.DataFrame(rows, columns=columns).fillna('-')
    print(df.to_csv(index=False, lineterminator='\n'), end='')
    print()
    print(tabulate(df, headers='keys', showindex=False))


def main():
    args = parse_args()
    run_reports = []
    for work_dir in args.work_dir:
        run_dir = resolve_run_dir(work_dir)
        run_reports.append({
            'run_dir': run_dir,
            'model_name': resolve_model_name(run_dir),
            'rows': collect_run_benchmark_report(run_dir),
        })

    if args.verbose:
        rows = run_reports[0]['rows']
        if not rows:
            return
        print_csv_and_table(build_verbose_report_rows(rows))
        return

    summary_rows, columns = build_summary_rows(run_reports, benchmark_order=args.data)
    if not summary_rows:
        return
    print_csv_and_table(summary_rows, columns=columns)


if __name__ == '__main__':
    main()
