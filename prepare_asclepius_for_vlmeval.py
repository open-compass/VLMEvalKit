#!/usr/bin/env python3
"""
Script to convert Asclepius benchmark from Excel to TSV format
and organize it for VLMEvalKit integration.

Usage:
    python prepare_asclepius_for_vlmeval.py --excel-path ./bench_data/Asclepius_bench.xlsx \
                                             --image-dir ./bench_data/images \
                                             --output-dir ~/.cache/lmu_datasets
"""

import argparse
import os
import shutil
import pandas as pd
from pathlib import Path


def convert_excel_to_tsv(excel_path, output_dir, dataset_name='Asclepius'):
    """Convert Asclepius Excel file to TSV format"""
    
    print(f"Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)
    
    print(f"Found {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    
    # Required columns
    required_columns = {
        'question_id': 'index',
        'question': 'question',
        'answer': 'answer',
        'image_id': 'image_id'
    }
    
    # Check for required columns
    for col in required_columns.keys():
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in Excel file")
    
    # Prepare data
    data = pd.DataFrame()
    
    for old_col, new_col in required_columns.items():
        if old_col in df.columns:
            data[new_col] = df[old_col]
    
    # Add optional columns
    optional_columns = ['image_id2', 'category', 'split']
    for col in optional_columns:
        if col in df.columns:
            data[col] = df[col]
    
    # Save as TSV
    tsv_path = os.path.join(output_dir, f'{dataset_name}.tsv')
    data.to_csv(tsv_path, sep='\t', index=False)
    
    print(f"✓ Saved TSV to: {tsv_path}")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Samples: {len(data)}")
    
    return tsv_path


def copy_images(source_dir, output_dir, dataset_name='Asclepius'):
    """Copy images to the correct location"""
    
    target_dir = os.path.join(output_dir, 'images', dataset_name)
    
    print(f"\nCopying images from: {source_dir}")
    print(f"To: {target_dir}")
    
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory does not exist: {source_dir}")
        return False
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy all files
    count = 0
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        target_file = os.path.join(target_dir, filename)
        
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_file)
            count += 1
    
    print(f"✓ Copied {count} images")
    return True


def verify_setup(output_dir, dataset_name='Asclepius'):
    """Verify the setup is correct"""
    
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    
    # Check TSV file
    tsv_path = os.path.join(output_dir, f'{dataset_name}.tsv')
    if os.path.exists(tsv_path):
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"✓ TSV file exists: {tsv_path}")
        print(f"  - Samples: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
    else:
        print(f"✗ TSV file not found: {tsv_path}")
    
    # Check images directory
    images_dir = os.path.join(output_dir, 'images', dataset_name)
    if os.path.exists(images_dir):
        image_count = len([f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))])
        print(f"✓ Images directory exists: {images_dir}")
        print(f"  - Image files: {image_count}")
    else:
        print(f"✗ Images directory not found: {images_dir}")
    
    # Test loading with VLMEvalKit
    print(f"\n{'='*60}")
    print("Testing VLMEvalKit Integration")
    print(f"{'='*60}")
    
    try:
        from vlmeval.dataset import build_dataset
        
        dataset = build_dataset(dataset_name)
        print(f"✓ Successfully loaded dataset: {dataset_name}")
        print(f"  - Type: {dataset.TYPE}")
        print(f"  - Modality: {dataset.MODALITY}")
        print(f"  - Samples: {len(dataset)}")
        
        # Try loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n✓ Sample #0:")
            print(f"  - Index: {sample.get('index')}")
            print(f"  - Question: {sample.get('question', '')[:50]}...")
            if sample.get('answer'):
                print(f"  - Answer: {sample.get('answer', '')[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Asclepius benchmark for VLMEvalKit'
    )
    parser.add_argument(
        '--excel-path',
        type=str,
        required=True,
        help='Path to Asclepius Excel file'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        required=True,
        help='Path to directory containing benchmark images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: ~/.cache/lmu_datasets)'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='Asclepius',
        help='Dataset name (default: Asclepius)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip verification step'
    )
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        # Use environment variable if set
        output_dir = os.environ.get('LMU_DATASET_ROOT')
        if output_dir is None:
            output_dir = os.path.expanduser('~/.cache/lmu_datasets')
    else:
        output_dir = os.path.expanduser(args.output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print("Asclepius Benchmark Preparation for VLMEvalKit")
    print(f"{'='*60}\n")
    
    # Convert Excel to TSV
    convert_excel_to_tsv(args.excel_path, output_dir, args.dataset_name)
    
    # Copy images
    copy_images(args.image_dir, output_dir, args.dataset_name)
    
    # Verify setup
    if not args.no_verify:
        verify_setup(output_dir, args.dataset_name)
    
    print(f"\n{'='*60}")
    print("Setup Complete!")
    print(f"{'='*60}")
    print(f"\nYou can now run evaluations with:")
    print(f"  python run.py --data {args.dataset_name} --model gpt-4v --verbose")


if __name__ == '__main__':
    main()
