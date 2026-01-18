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

# Copy the utility functions from vlmeval/smp/vlm.py for image encoding/decoding

import os
import io
import pandas as pd
# import numpy as np
# import string
# from uuid import uuid4
import os.path as osp
import base64
from PIL import Image
# import sys

Image.MAX_IMAGE_PIXELS = 1e9


def resize_image_by_factor(img, factor=1):
    w, h = img.size
    new_w, new_h = int(w * factor), int(h * factor)
    img = img.resize((new_w, new_h))
    return img


def encode_image_to_base64(img, target_size=-1, fmt='JPEG'):
    # if target_size == -1, will not do resizing
    # else, will set the max_size ot (target_size, target_size)
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')
    if target_size > 0:
        img.thumbnail((target_size, target_size))
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=fmt)
    image_data = img_buffer.getvalue()
    ret = base64.b64encode(image_data).decode('utf-8')
    max_size = os.environ.get('VLMEVAL_MAX_IMAGE_SIZE', 1e9)
    min_edge = os.environ.get('VLMEVAL_MIN_IMAGE_EDGE', 1e2)
    max_size = int(max_size)
    min_edge = int(min_edge)

    if min(img.size) < min_edge:
        factor = min_edge / min(img.size)
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    factor = 1
    while len(ret) > max_size:
        factor *= 0.7  # Half Pixels Per Resize, approximately
        image_new = resize_image_by_factor(img, factor)
        img_buffer = io.BytesIO()
        image_new.save(img_buffer, format=fmt)
        image_data = img_buffer.getvalue()
        ret = base64.b64encode(image_data).decode('utf-8')

    if factor < 1:
        new_w, new_h = image_new.size
        print(
            f'Warning: image size is too large and exceeds `VLMEVAL_MAX_IMAGE_SIZE` {max_size}, '
            f'resize to {factor:.2f} of original size: ({new_w}, {new_h})'
        )

    return ret


def encode_image_file_to_base64(image_path, target_size=-1, fmt='JPEG', safe_max_pixels=40_000_000):
    """Load and encode an image with defensive downscaling to avoid OOM."""
    with Image.open(image_path) as image:
        w, h = image.size
        pixels = w * h

        # Decoder-level reduction to keep memory low on huge images
        if pixels > safe_max_pixels and hasattr(image, 'reduce'):
            # Pick reduction so resulting pixels <= safe_max_pixels
            reduce_r = int((pixels / safe_max_pixels) ** 0.5) + 1
            try:
                image = image.reduce(reduce_r)
            except Exception:
                pass  # Fallback to later thumbnail

        # Hint decoder to lower resolution when possible
        if target_size > 0 and hasattr(image, 'draft'):
            try:
                image.draft(image.mode, (target_size, target_size))
            except Exception:
                pass

        return encode_image_to_base64(image, target_size=target_size, fmt=fmt)


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ('RGBA', 'P', 'LA'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def decode_base64_to_image_file(base64_string, image_path, target_size=-1):
    image = decode_base64_to_image(base64_string, target_size=target_size)
    base_dir = osp.dirname(image_path)
    if not osp.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
    image.save(image_path)


def convert_excel_to_tsv(excel_path, output_dir, dataset_name='Asclepius', force=False):
    """Convert Asclepius Excel file to TSV format; skip if it already exists unless forced."""

    tsv_path = os.path.join(output_dir, f'{dataset_name}.tsv')
    if os.path.exists(tsv_path) and not force:
        print(f"TSV already exists, skipping Excel conversion: {tsv_path}")
        return tsv_path

    print(f"Loading Excel file: {excel_path}")
    df = pd.read_excel(excel_path)

    print(f"Found {len(df)} samples")
    print(f"Columns: {list(df.columns)}")

    # Required columns
    required_columns = {
        'question_id': 'index',
        'question': 'question',
        'answer': 'answer',
        'image_id': 'image',
        'image_id2': 'image_2',
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

    # Save as TSV
    data.to_csv(tsv_path, sep='\t', index=False)

    print(f"✓ Saved TSV to: {tsv_path}")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Samples: {len(data)}")

    return tsv_path


def copy_images(source_dir, output_dir, dataset_name='Asclepius', chunk_size=200, target_size=1024, start_from=0):
    """Stream-encode images to base64 and update TSV in small chunks.

    This avoids loading the whole TSV or thousands of base64 strings into memory.
    `target_size` bounds the largest edge when encoding to reduce memory spikes.
    
    Supports resuming from a temporary file if the process exits early.
    
    Args:
        start_from: Index to start encoding from (0-based). Rows before this will be skipped.
    """
    
    tsv_path = os.path.join(output_dir, f'{dataset_name}.tsv')
    temp_path = tsv_path + '.tmp'
    
    print(f"\nEncoding images from: {source_dir}")
    print(f"Updating TSV file: {tsv_path}")
    print(f"Chunk size: {chunk_size} rows")
    print(f"Target encode size: {target_size}")
    if start_from > 0:
        print(f"Starting from index: {start_from}")
    
    if not os.path.exists(source_dir):
        print(f"Warning: Source directory does not exist: {source_dir}")
        return False
    
    if not os.path.exists(tsv_path):
        print(f"Warning: TSV file does not exist: {tsv_path}")
        return False
    
    try:
        # Check for and merge any existing temporary file from a previous run
        if os.path.exists(temp_path):
            print(f"\nFound incomplete temporary file: {temp_path}")
            print("Merging already-encoded images back into main TSV...")
            
            try:
                temp_df = pd.read_csv(temp_path, sep='\t')
                main_df = pd.read_csv(tsv_path, sep='\t')
                
                # Merge encoded columns from temp file back to main file
                # Use index-based merge to ensure we update the right rows
                for col in ['image', 'image_2']:
                    if col in temp_df.columns and col in main_df.columns:
                        # Update rows where temp file has encoded (long) strings
                        encoded_mask = temp_df[col].notna() & (temp_df[col].astype(str).str.len() > 100)
                        if encoded_mask.any():
                            main_df.loc[encoded_mask.index, col] = temp_df.loc[encoded_mask.index, col]
                
                # Save merged data back to main TSV
                main_df.to_csv(tsv_path, sep='\t', index=False)
                print(f"✓ Merged encoded images from temporary file back to main TSV")
                
            except Exception as merge_err:
                print(f"Warning: Could not merge temporary file: {merge_err}")
                print("Proceeding with fresh encoding...")
            
            # Delete the temporary file to start fresh
            os.remove(temp_path)
            print(f"✓ Deleted temporary file: {temp_path}")
    
        # Build a mapping of image names (without extension) to full file paths
        print("Building image file mapping...")
        image_files = {}
        for root, _, files in os.walk(source_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                name_without_ext = os.path.splitext(filename)[0].strip()
                image_files[name_without_ext] = file_path
        
        print(f"Found {len(image_files)} image files in source directory")
        
        # Helper function to determine image format
        def get_image_format(file_path):
            """Determine the PIL format based on file extension"""
            ext = os.path.splitext(file_path)[1].lower()
            format_map = {
                '.jpg': 'JPEG',
                '.jpeg': 'JPEG',
                '.png': 'PNG',
                '.gif': 'GIF',
                '.bmp': 'BMP',
                '.tiff': 'TIFF',
                '.tif': 'TIFF',
                '.webp': 'WEBP'
            }
            return format_map.get(ext, 'JPEG')  # Default to JPEG if unknown
        
        # Process both columns in streaming mode
        encoded_count = 0
        skipped_count = 0
        deleted_count = 0
        missing_images = set()
        failed_images = []
        rows_to_delete = set()  # Track row indices to delete (for .tif files)
        log_path = os.path.join(output_dir, f'{dataset_name}_encode.log')

        temp_path = tsv_path + '.tmp'

        reader = pd.read_csv(tsv_path, sep='\t', chunksize=chunk_size)
        try:
            for chunk_idx, chunk in enumerate(reader):
                try:
                    for col in ['image', 'image_2']:
                        if col not in chunk.columns:
                            continue

                        mask = chunk[col].notna() & (chunk[col].astype(str).str.strip() != '')
                        if mask.sum() == 0:
                            continue

                        for row_idx in chunk[mask].index:
                            # Skip rows before start_from index
                            if row_idx < start_from:
                                skipped_count += 1
                                continue
                            
                            image_name_str = str(chunk.at[row_idx, col]).strip()

                            # Skip if already encoded (resume capability)
                            # Base64 encoded images are much longer than filename strings
                            if len(image_name_str) > 100:
                                encoded_count += 1
                                continue

                            if (encoded_count + 1) % 50 == 0:
                                print(f"  Processed {encoded_count} images...", flush=True)

                            if image_name_str in image_files:
                                image_path = image_files[image_name_str]
                                
                                # Check if image is .tif format - mark entire row for deletion
                                if image_path.lower().endswith('.tif'):
                                    rows_to_delete.add(row_idx)
                                    deleted_count += 1
                                    continue
                                
                                try:
                                    img_format = get_image_format(image_path)
                                    base64_string = encode_image_file_to_base64(
                                        image_path,
                                        fmt=img_format,
                                        target_size=target_size
                                    )
                                    chunk.at[row_idx, col] = base64_string
                                    encoded_count += 1
                                except Exception as e:
                                    error_msg = f"Failed to encode {image_path}: {e}"
                                    print(f"  Warning: {error_msg}", flush=True)
                                    failed_images.append((image_name_str, str(e)))
                                continue
                            if image_name_str.split(".")[0] in image_files:
                                image_path = image_files[image_name_str.split(".")[0]]
                                
                                # Check if image is .tif format - mark entire row for deletion
                                if image_path.lower().endswith('.tif'):
                                    rows_to_delete.add(row_idx)
                                    deleted_count += 1
                                    continue
                                
                                try:
                                    img_format = get_image_format(image_path)
                                    base64_string = encode_image_file_to_base64(
                                        image_path,
                                        fmt=img_format,
                                        target_size=target_size
                                    )
                                    chunk.at[row_idx, col] = base64_string
                                    encoded_count += 1
                                except Exception as e:
                                    error_msg = f"Failed to encode {image_path}: {e}"
                                    print(f"  Warning: {error_msg}", flush=True)
                                    failed_images.append((image_name_str.split(".")[0], str(e)))
                                continue
                            else:
                                missing_images.add(image_name_str)

                    # Append processed chunk to temp file
                    chunk.to_csv(
                        temp_path,
                        sep='\t',
                        index=False,
                        mode='a',
                        header=(chunk_idx == 0)
                    )
                    print(
                        f"  Chunk {chunk_idx} done (rows {chunk.index.min()}-{chunk.index.max()}), encoded so far {encoded_count}.\n Encoded {encoded_count} samples, skipped {skipped_count} samples",
                        flush=True
                    )
                except Exception as chunk_err:
                    msg = f"Chunk {chunk_idx} failed: {chunk_err}"
                    print(f"ERROR: {msg}", flush=True)
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(msg + '\n')
                    import traceback
                    traceback.print_exc()
                    print(f"\nNote: Progress saved in temporary file {temp_path}")
                    print("You can resume by running the script again.")
                    return False
        finally:
            if hasattr(reader, 'close'):
                reader.close()
            del reader

        # Replace main TSV with the newly processed temp file
        os.replace(temp_path, tsv_path)
        
        # Remove rows with .tif images from the TSV file
        if rows_to_delete:
            print(f"\n✓ Removing {deleted_count} rows with .tif images...")
            df = pd.read_csv(tsv_path, sep='\t')
            df = df.drop(list(rows_to_delete))
            df.to_csv(tsv_path, sep='\t', index=False)
            print(f"✓ Deleted {deleted_count} rows with .tif images from TSV")
        
        print(f"\n✓ Encoded {encoded_count} images to base64")
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} rows before index {start_from}")
        if deleted_count > 0:
            print(f"  Deleted {deleted_count} rows containing .tif images")

        if missing_images:
            print(f"\nWarning: {len(missing_images)} images not found in source directory:")
            for img in list(missing_images)[:10]:
                print(f"  - {img}")
            if len(missing_images) > 10:
                print(f"  ... and {len(missing_images) - 10} more")

        if failed_images:
            print(f"\nWarning: {len(failed_images)} images failed to encode:")
            for img_name, error in failed_images[:5]:
                print(f"  - {img_name}: {error}")
            if len(failed_images) > 5:
                print(f"  ... and {len(failed_images) - 5} more")

        return True
        
    except Exception as e:
        print(f"\nERROR: An exception occurred during image processing:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        
        # Check if images are base64 encoded
        for col in ['image', 'image_2']:
            if col in df.columns:
                non_empty = df[col].notna() & (df[col].astype(str).str.strip() != '')
                if non_empty.any():
                    first_value = df[col][non_empty].iloc[0]
                    is_base64 = len(str(first_value)) > 100  # Base64 strings are long
                    print(f"  - Column '{col}': {'Base64 encoded' if is_base64 else 'Not encoded'}")
    else:
        print(f"✗ TSV file not found: {tsv_path}")
    
    return True


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
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=200,
        help='Rows to process per chunk when encoding images (lower to reduce memory)'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=1024,
        help='Max edge size when encoding images (pixels) to limit memory; -1 to disable'
    )
    parser.add_argument(
        '--force-recreate-tsv',
        action='store_true',
        help='Recreate TSV from Excel even if it already exists (otherwise resume)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Index to start encoding from (0-based). Rows before this will be skipped.'
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
    convert_excel_to_tsv(
        args.excel_path,
        output_dir,
        args.dataset_name,
        force=args.force_recreate_tsv,
    )
    
    # Copy images
    success = copy_images(
        args.image_dir,
        output_dir,
        args.dataset_name,
        chunk_size=args.chunk_size,
        target_size=args.target_size,
        start_from=args.start_from,
    )
    
    # Verify setup
    if not args.no_verify:
        verify_setup(output_dir, args.dataset_name)
    
    print(f"\n{'='*60}")
    if success:
        print("Setup Complete!")
    else:
        print("Setup completed with errors - please review warnings above")
    print(f"{'='*60}")
    print(f"\nYou can now run evaluations with:")
    print(f"  python run.py --data {args.dataset_name} --model gpt-4v --verbose")


if __name__ == '__main__':
    main()
