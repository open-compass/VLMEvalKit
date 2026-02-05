import pandas as pd


def create_small(dataset_path):
    print(f"Reading {dataset_path}...")
    # Use chunksize or nrows to avoid memory issues if file is huge, though we just want 12 rows.
    # We must assume dataset_path is in CWD as per previous steps.
    try:
        df = pd.read_csv(dataset_path, sep='\t', nrows=12)
        print(f"Read {len(df)} rows.")
        output = f'{dataset_path.split(".tsv")[0]}Small.tsv'
        df.to_csv(output, sep='\t', index=False)
        print(f"Saved {output} with {len(df)} samples.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    create_small("docvqa.tsv")