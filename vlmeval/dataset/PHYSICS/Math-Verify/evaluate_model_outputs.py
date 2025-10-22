import argparse
import pandas as pd
from typing import Any
from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
import sympy

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and evaluate answers using sympy')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file containing model outputs')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to output CSV file for extracted answers')
    parser.add_argument('--gold_is_latex', action='store_true', help='Use basic latex normalization', default=True)
    return parser.parse_args()

def load_csv_data(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV data."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['answer', 'gold']
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

def compare_answers(extracted: Any, gold: Any) -> bool:
    """Compare extracted answer with gold answer."""
    if extracted is None or gold is None:
        return False
    try:
        # Handle lists/tuples of expressions
        if isinstance(extracted, (list, tuple)) and isinstance(gold, (list, tuple)):
            if len(extracted) != len(gold):
                return False
            return all(sympy.simplify(a - b) == 0 for a, b in zip(extracted, gold))
        
        # Handle single expressions
        return sympy.simplify(extracted - gold) == 0
    except Exception:
        # If comparison fails (e.g. different types), return False
        return False

def process_answers(df: pd.DataFrame, gold_is_latex: bool) -> pd.DataFrame:
    """Process each answer through the sympy extraction workflow and compare with gold using math_verify."""
    results = []
    
    
    correct_count = 0
    total_count = 0
    
    # Create the verification function
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig() if gold_is_latex else ExprExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        aggregation_function=max,
        precision=6
    )
    
    for _, row in df.iterrows():
        extracted_answers = None
        gold_answers = None
        grade = 0
        try:
            # Use the verification function
            grade, extracted_answers = verify_func([row['gold']], [row['answer']])
            
            if extracted_answers is None:
                extracted_answers = None
                gold_answers = None
            else:
                gold_answers = extracted_answers[0]
                extracted_answers = extracted_answers[1]

            total_count += 1
            if grade == 1:
                correct_count += 1
            
            result = {
                'original_answer': row['answer'],
                'gold_answer': row['gold'],
                'extracted_answer': extracted_answers,
                'extracted_gold': gold_answers,
                'is_correct': grade == 1
            }
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'original_answer': row['answer'],
                'gold_answer': row['gold'],
                'extracted_answer': extracted_answers,
                'extracted_gold': gold_answers,
                'is_correct': grade == 1,
                'error': str(e)
            })
    
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"\nEvaluation Results:")
    print(f"Total examples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")
    
    # Add summary stats to the dataframe
    results_df.attrs['accuracy'] = accuracy
    results_df.attrs['total_count'] = total_count
    results_df.attrs['correct_count'] = correct_count
    
    return results_df

def main():
    args = parse_args()
    
    # Load input CSV
    input_df = load_csv_data(args.input_csv)
    
    # Process answers and extract sympy objects
    results_df = process_answers(input_df, args.gold_is_latex)
    
    # Save results to output CSV
    results_df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

if __name__ == "__main__":
    main()


