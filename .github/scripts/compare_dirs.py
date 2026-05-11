import argparse
import difflib
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CSV and JSON files with the same basename in two directories (top-level only). "
            "For CSV, row order is ignored (rows sorted by all columns). "
            "For JSON, object key order is ignored (keys sorted recursively)."
        )
    )
    parser.add_argument("dir_a", type=str, help="First directory (e.g. baseline)")
    parser.add_argument("dir_b", type=str, help="Second directory (e.g. current run)")
    parser.add_argument(
        "--max-diff",
        type=int,
        default=5,
        help="CSV: max cell-level diffs per file. JSON: max lines of unified diff per file (default: 5)",
    )
    return parser.parse_args()


def list_compare_files(root: Path):
    """Top-level only: *.csv and *.json."""
    out = {}
    for pattern in ("*.csv", "*.json"):
        for p in root.glob(pattern):
            out[p.name] = p
    return out


def normalize_value(value):
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _sort_rows_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    sort_cols = list(df.columns)
    return df.sort_values(by=sort_cols, kind="mergesort", na_position="last").reset_index(
        drop=True
    )


def compare_csv(file_a: Path, file_b: Path, max_diff: int):
    diffs = []
    try:
        df_a = pd.read_csv(file_a, dtype=str)
    except Exception as exc:
        return [f"failed to read left file: {exc}"]

    try:
        df_b = pd.read_csv(file_b, dtype=str)
    except Exception as exc:
        return [f"failed to read right file: {exc}"]

    if tuple(df_a.shape) != tuple(df_b.shape):
        diffs.append(f"shape differs: left={df_a.shape}, right={df_b.shape}")

    cols_a = list(df_a.columns)
    cols_b = list(df_b.columns)
    if cols_a != cols_b:
        diffs.append(f"columns differ: left={cols_a}, right={cols_b}")

    if diffs:
        return diffs

    df_a = _sort_rows_for_compare(df_a)
    df_b = _sort_rows_for_compare(df_b)

    common_rows = len(df_a)
    common_cols = list(df_a.columns)

    printed = 0
    for row_idx in range(common_rows):
        for col in common_cols:
            va = normalize_value(df_a.iloc[row_idx][col])
            vb = normalize_value(df_b.iloc[row_idx][col])
            if va != vb:
                diffs.append(
                    f"sorted_row={row_idx}, col='{col}': left={va!r}, right={vb!r}"
                )
                printed += 1
                if printed >= max_diff:
                    diffs.append(f"... truncated after {max_diff} cell diffs")
                    return diffs
    return diffs


def _normalize_json_keys(obj):
    """Recursively sort dict keys so key order does not affect equality."""
    if isinstance(obj, dict):
        return {k: _normalize_json_keys(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_normalize_json_keys(x) for x in obj]
    return obj


def compare_json(file_a: Path, file_b: Path, max_diff_lines: int):
    try:
        with open(file_a, encoding="utf-8") as f:
            obj_a = json.load(f)
    except Exception as exc:
        return [f"failed to read or parse left JSON: {exc}"]

    try:
        with open(file_b, encoding="utf-8") as f:
            obj_b = json.load(f)
    except Exception as exc:
        return [f"failed to read or parse right JSON: {exc}"]

    na = _normalize_json_keys(obj_a)
    nb = _normalize_json_keys(obj_b)

    if na == nb:
        return []

    lines_a = json.dumps(na, ensure_ascii=False, indent=2).splitlines()
    lines_b = json.dumps(nb, ensure_ascii=False, indent=2).splitlines()
    diff_iter = difflib.unified_diff(
        lines_a,
        lines_b,
        fromfile=str(file_a),
        tofile=str(file_b),
        lineterm="",
    )
    diff_lines = list(diff_iter)
    if not diff_lines:
        return ["JSON values differ but unified diff is empty (unexpected)."]

    cap = max_diff_lines + 12
    out = diff_lines[:cap]
    if len(diff_lines) > cap:
        out.append(
            f"... truncated: {len(diff_lines) - cap} more diff lines (raise --max-diff)"
        )
    return out


def main():
    args = parse_args()
    dir_a = Path(args.dir_a).resolve()
    dir_b = Path(args.dir_b).resolve()

    if not dir_a.is_dir():
        print(f"left directory not found: {dir_a}")
        sys.exit(2)
    if not dir_b.is_dir():
        print(f"right directory not found: {dir_b}")
        sys.exit(2)

    files_a = list_compare_files(dir_a)
    files_b = list_compare_files(dir_b)

    names_a = set(files_a.keys())
    names_b = set(files_b.keys())

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    common = sorted(names_a & names_b)

    has_diff = False

    if only_a:
        has_diff = True
        print("Files only in left directory:")
        for name in only_a:
            print(f"  - {name}")

    if only_b:
        has_diff = True
        print("Files only in right directory:")
        for name in only_b:
            print(f"  - {name}")

    for name in common:
        path_a = files_a[name]
        path_b = files_b[name]
        suffix = path_a.suffix.lower()

        if suffix == ".csv":
            diffs = compare_csv(path_a, path_b, args.max_diff)
        elif suffix == ".json":
            diffs = compare_json(path_a, path_b, args.max_diff)
        else:
            diffs = [f"unsupported file type: {suffix}"]

        if diffs:
            has_diff = True
            print(f"\nDifferences in {name}:")
            for diff in diffs:
                print(f"  {diff}")

    if has_diff:
        print("\nComparison failed.")
        sys.exit(1)

    print("All common CSV and JSON files are identical.")
    sys.exit(0)


if __name__ == "__main__":
    main()
