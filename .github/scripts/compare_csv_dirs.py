import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare CSV files with the same basename in two directories (top-level only). "
            "Row order is ignored; rows are sorted by all columns before comparing."
        )
    )
    parser.add_argument("dir_a", type=str, help="First directory")
    parser.add_argument("dir_b", type=str, help="Second directory")
    parser.add_argument(
        "--max-diff",
        type=int,
        default=5,
        help="Max number of cell-level diffs printed per file (default: 5)",
    )
    return parser.parse_args()


def list_csv_files(root: Path):
    # Compare only CSV files directly under the given directory.
    return {p.name: p for p in root.glob("*.csv")}


def normalize_value(value):
    if pd.isna(value):
        return "<NA>"
    return str(value)


def _sort_rows_for_compare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows so order-independent comparison matches multiset of rows."""
    if df.empty:
        return df.reset_index(drop=True)
    sort_cols = list(df.columns)
    # Stable sort on all columns; NaNs last for reproducibility.
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

    files_a = list_csv_files(dir_a)
    files_b = list_csv_files(dir_b)

    names_a = set(files_a.keys())
    names_b = set(files_b.keys())

    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)
    common = sorted(names_a & names_b)

    has_diff = False

    if only_a:
        has_diff = True
        print("CSV files only in left directory:")
        for name in only_a:
            print(f"  - {name}")

    if only_b:
        has_diff = True
        print("CSV files only in right directory:")
        for name in only_b:
            print(f"  - {name}")

    for rel in common:
        diffs = compare_csv(files_a[rel], files_b[rel], args.max_diff)
        if diffs:
            has_diff = True
            print(f"\nDifferences in {rel}:")
            for diff in diffs:
                print(f"  - {diff}")

    if has_diff:
        print("\nCSV comparison failed.")
        sys.exit(1)

    print("All common CSV files are identical.")
    sys.exit(0)


if __name__ == "__main__":
    main()
