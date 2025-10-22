import argparse
import pandas as pd
from typing import Any
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, parse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and evaluate answers using sympy')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file containing model outputs')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file for extracted answers')
    return parser.parse_args()

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['answer']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    except Exception as e:
        raise Exception(f"Error loading CSV file: {str(e)}")

def serialize_sympy_object(obj: Any) -> str:
    """Convert sympy object to string representation."""
    if obj is None:
        return ""
    try:
        if isinstance(obj, (list, tuple)):
            return ", ".join(str(x) if x is not None else "" for x in obj)
        return str(obj)
    except Exception as e:
        return f"Error: {str(e)}"

def process_answers(df: pd.DataFrame) -> pd.DataFrame:
    """Process each answer through the sympy extraction workflow."""
    results = []
    
    # Set up extraction config and get regexes
    extraction_target = (ExprExtractionConfig(), LatexExtractionConfig())
    for _, row in df.iterrows():
        try:
            # Extract answer using regexes
            extracted = parse(row['answer'], extraction_config=extraction_target)
            feedback = None
            extracted_answer = None
            if len(extracted) == 2:
                extracted_answer = extracted[0]
                feedback = extracted[1]
            elif len(extracted) == 1:
                extracted_answer = extracted[0]
            else:
                feedback = "No valid extraction found"
                extracted_answer = None


            
            result = {
                'original_answer': row['answer'],
                'extracted_answer': serialize_sympy_object(extracted_answer),
                'extracted_feedback': feedback,
                'extraction_success': extracted_answer is not None
            }
            
            # Copy any other columns from input
            for col in df.columns:
                if col != 'answer':
                    result[col] = row[col]
                    
            results.append(result)
            
        except Exception as e:
            results.append({
                'original_answer': row['answer'],
                'extracted_answer': '',
                'extraction_success': False,
                'error': str(e)
            })
    
    return pd.DataFrame(results)

def main():
    args = parse_args()
    
    # Load input CSV
    input_df = load_csv_data(args.input_csv)
    
    # Process answers and extract sympy objects
    results_df = process_answers(input_df)
    
    # Save results to output CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()


