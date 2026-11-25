#!/usr/bin/env python3
"""Build an MMMU_TEST TSV from the official MMMU/MMMU dataset.

By default this uses the latest official revision and ``convert_rgb=False``.
Official PNG bytes are copied verbatim into base64; other formats are encoded
as PNG with ``encode_image_to_base64(..., fmt='PNG', convert_rgb=False)``.
The resulting TSV contains base64 PNG images and ``.png`` image paths, so RGBA
images keep their alpha channel.
"""

import argparse
import ast
import base64
import csv
import glob
import hashlib
import io
import json
import os
import os.path as osp
from datetime import datetime, timezone

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from vlmeval.smp.vlm import encode_image_to_base64


REPO_ID = 'MMMU/MMMU'
OPTION_COLS = list('ABCDEFGHI')
# Match the hosted MMMU_TEST.tsv column order so ImageMCQDataset and the
# official submission exporter can consume the rebuilt file unchanged.
OUTPUT_COLUMNS = [
    'id', 'index', 'question', 'split', 'A', 'B', 'C', 'D', 'answer',
    'topic_difficulty', 'subfield', 'image_type', 'question_type', 'explanation',
    'image', 'image_path', 'E', 'F', 'G', 'H', 'I', 'category', 'l2-category',
]
# Official MMMU config order. Used only when the HuggingFace metadata API is
# unavailable (for example, an offline run with --data-root).
DEFAULT_SUBJECTS = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine',
    'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics',
    'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature',
    'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music',
    'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology',
]
# Coarse two-level taxonomy used by the original VLMEvalKit MMMU_TEST.tsv.
L2_CATEGORY = {
    'Accounting': 'Business',
    'Agriculture': 'Tech & Engineering',
    'Architecture_and_Engineering': 'Tech & Engineering',
    'Art': 'Art & Design',
    'Art_Theory': 'Art & Design',
    'Basic_Medical_Science': 'Health & Medicine',
    'Biology': 'Science',
    'Chemistry': 'Science',
    'Clinical_Medicine': 'Health & Medicine',
    'Computer_Science': 'Tech & Engineering',
    'Design': 'Art & Design',
    'Diagnostics_and_Laboratory_Medicine': 'Health & Medicine',
    'Economics': 'Business',
    'Electronics': 'Tech & Engineering',
    'Energy_and_Power': 'Tech & Engineering',
    'Finance': 'Business',
    'Geography': 'Science',
    'History': 'Humanities & Social Science',
    'Literature': 'Humanities & Social Science',
    'Manage': 'Business',
    'Marketing': 'Business',
    'Materials': 'Tech & Engineering',
    'Math': 'Science',
    'Mechanical_Engineering': 'Tech & Engineering',
    'Music': 'Art & Design',
    'Pharmacy': 'Health & Medicine',
    'Physics': 'Science',
    'Psychology': 'Humanities & Social Science',
    'Public_Health': 'Health & Medicine',
    'Sociology': 'Humanities & Social Science',
}


def parse_list(value):
    """Parse the Python-list strings used by MMMU's options/img_type fields."""
    if isinstance(value, list):
        return value
    if value is None:
        return []
    text = str(value)
    if text == '' or text.lower() == 'nan':
        return []
    return ast.literal_eval(text)


def sanitize_text(value):
    """Restore LaTeX escapes and remove CSV-breaking control characters.

    MMMU's parquet text sometimes stores a LaTeX backslash escape as a real
    control character (for example ``\\right`` becomes CR + ``ight``).  A CR or
    tab left in a pandas-written field can split a TSV row, so restore the
    literal backslash form before serialization.  Real CRLF line breaks are
    normalized to LF and remain ordinary in-field newlines.
    """
    text = str(value).replace('\r\n', '\n')
    for char, escaped in (
        ('\x07', '\\a'), ('\x08', '\\b'), ('\x09', '\\t'), ('\x0b', '\\v'),
        ('\x0c', '\\f'), ('\x0d', '\\r'),
    ):
        text = text.replace(char, escaped)
    return text


def file_md5(path):
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def drop_trailing_empty_options(options):
    """Match the legacy TSV cleanup: drop trailing None/empty placeholders."""
    options = list(options)
    while options and (options[-1] is None or str(options[-1]) == '' or str(options[-1]) == 'None'):
        options.pop()
    return options


def resolve_metadata(revision, token, data_root):
    """Resolve subject order and split counts.

    The dev/validation counts are needed only to reproduce the public TSV's
    global ``index`` numbering; their rows are not emitted.
    """
    from huggingface_hub import HfApi

    try:
        info = HfApi().dataset_info(REPO_ID, revision=revision, token=token)
        configs = info.cardData.get('dataset_info', [])
        order = [cfg['config_name'] for cfg in configs] or list(DEFAULT_SUBJECTS)
        counts = {
            cfg['config_name']: {split['name']: split['num_examples'] for split in cfg['splits']}
            for cfg in configs
        }
        return info.sha, order, counts
    except Exception:
        if not data_root:
            raise
        print('warning: HuggingFace metadata unavailable; using default subject order and 5/30 dev/val counts')
        counts = {subject: {'dev': 5, 'validation': 30} for subject in DEFAULT_SUBJECTS}
        return revision, list(DEFAULT_SUBJECTS), counts


def prepare_data_root(args, resolved_sha):
    """Return a local root containing only the official test shards."""
    if args.data_root:
        return osp.abspath(args.data_root)
    from huggingface_hub import snapshot_download

    os.makedirs(args.work_dir, exist_ok=True)
    return snapshot_download(
        repo_id=REPO_ID,
        repo_type='dataset',
        revision=resolved_sha,
        allow_patterns=['*/test-*.parquet'],
        local_dir=args.work_dir,
        token=args.token,
    )


def subject_paths(root, subject):
    return sorted(glob.glob(osp.join(root, subject, 'test-*.parquet')))


def subject_row_count(root, subject):
    """Row count for a skipped subject, used only to advance global_index."""
    paths = subject_paths(root, subject)
    if not paths:
        raise FileNotFoundError(f'No test parquet found for {subject} under {root}')
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def build_row(official, subject, index, image_format, image_ext, convert_rgb,
              drop_empty_options, keep_answers, sanitize):
    """Convert one official parquet row to the VLMEvalKit TSV schema."""
    options = parse_list(official.get('options'))
    if drop_empty_options:
        options = drop_trailing_empty_options(options)
    if sanitize:
        options = [sanitize_text(value) for value in options]
    if len(options) > len(OPTION_COLS):
        raise ValueError(f'{official["id"]} has {len(options)} options; at most 9 are supported')
    # A-I columns are the VLMEvalKit MCQ representation of the official option list.
    option_values = list(options) + [''] * (len(OPTION_COLS) - len(options))

    # Official MMMU stores up to seven images in image_1 ... image_7.
    image_fields = []
    for idx in range(1, 8):
        field = official.get(f'image_{idx}')
        if field and field.get('bytes'):
            image_fields.append((idx, field['bytes']))
    if not image_fields:
        raise ValueError(f'{official["id"]} has no image')

    images = []
    image_paths = []
    for order, (_, image_bytes) in enumerate(image_fields, 1):
        image = Image.open(io.BytesIO(image_bytes))
        if not convert_rgb and image.format == 'PNG':
            # Preserve the official PNG bytes exactly (including RGBA alpha).
            images.append(base64.b64encode(image_bytes).decode('utf-8'))
        else:
            images.append(encode_image_to_base64(
                image, target_size=-1, fmt=image_format, convert_rgb=convert_rgb))
        # image_path uses the global index, matching the hosted TSV naming.
        image_paths.append(f'{index}_{order}{image_ext}')

    # Keep text CSV-safe and restore LaTeX escapes before writing the row.
    question = sanitize_text(official['question']) if sanitize else official['question']
    explanation = official.get('explanation', '')
    topic_difficulty = official.get('topic_difficulty', '')
    subfield = official.get('subfield', '')
    question_type = official.get('question_type', '')
    answer = official.get('answer', '') if keep_answers else '?'
    if sanitize:
        explanation = sanitize_text(explanation)
        topic_difficulty = sanitize_text(topic_difficulty)
        subfield = sanitize_text(subfield)
        question_type = sanitize_text(question_type)
        answer = sanitize_text(answer)

    row = {
        'id': official['id'],
        'index': index,
        'question': question,
        'split': 'test',
        'answer': answer,
        'topic_difficulty': topic_difficulty,
        'subfield': subfield,
        'image_type': ','.join(parse_list(official.get('img_type'))),
        'question_type': question_type,
        'explanation': explanation,
        'image': images,
        'image_path': image_paths,
        'category': subject,
        'l2-category': L2_CATEGORY.get(subject, ''),
    }
    for col, value in zip(OPTION_COLS, option_values):
        row[col] = value
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='MMMU_TEST_876ce5.tsv', help='Output TSV path.')
    parser.add_argument('--revision', default='main', help='HuggingFace revision (default: latest main).')
    parser.add_argument('--data-root', default=None,
                        help='Local directory containing <subject>/test-*.parquet. If omitted, download.')
    parser.add_argument('--work-dir', default='MMMU_downloads', help='Download cache when --data-root is omitted.')
    parser.add_argument('--subjects', nargs='*', default=None, help='Optional subject subset.')
    parser.add_argument('--token', default=os.environ.get('HF_TOKEN'), help='Optional HuggingFace token.')
    parser.add_argument('--convert-rgb', action='store_true',
                        help='Legacy mode: convert images to RGB/JPEG instead of preserving PNG alpha.')
    parser.add_argument('--keep-answers', action='store_true',
                        help='Keep official answers instead of masking them with "?".')
    parser.add_argument('--keep-none-options', action='store_true',
                        help='Keep trailing None/empty options instead of dropping them.')
    parser.add_argument('--no-sanitize', action='store_true',
                        help='Do not restore control-character escapes such as \\f, \\b, \\r and \\t.')
    args = parser.parse_args()

    # Keep original dimensions; encode_image_to_base64 otherwise upscales tiny images.
    os.environ['VLMEVAL_MIN_IMAGE_EDGE'] = '0'
    csv.field_size_limit(1 << 30)

    resolved_sha, subject_order, counts = resolve_metadata(args.revision, args.token, args.data_root)
    root = prepare_data_root(args, resolved_sha)
    print(f'MMMU source: revision={args.revision}, resolved_sha={resolved_sha}, root={root}', flush=True)
    selected = args.subjects or subject_order
    unknown = sorted(set(selected) - set(subject_order))
    if unknown:
        raise ValueError(f'Unknown subjects: {unknown}')

    image_format = 'JPEG' if args.convert_rgb else 'PNG'
    image_ext = '.jpg' if args.convert_rgb else '.png'
    rows = []
    total_images = 0
    # Recreate the original MMMU concatenation numbering: for each subject,
    # dev(5) -> test -> validation(30). Only test rows are emitted.
    global_index = 1

    for subject in subject_order:
        dev_count = counts.get(subject, {}).get('dev', 5)
        val_count = counts.get(subject, {}).get('validation', 30)
        global_index += dev_count
        subject_rows = 0
        if subject in selected:
            paths = subject_paths(root, subject)
            if not paths:
                raise FileNotFoundError(f'No test parquet found for {subject} under {root}')
            for path in paths:
                table = pq.read_table(path)
                for official in table.to_pylist():
                    rows.append(build_row(
                        official, subject, global_index, image_format, image_ext,
                        args.convert_rgb, not args.keep_none_options,
                        args.keep_answers, not args.no_sanitize))
                    total_images += len(rows[-1]['image'])
                    subject_rows += 1
                    global_index += 1
        else:
            global_index += subject_row_count(root, subject)
        global_index += val_count
        if subject in selected:
            print(
                f'{subject}: rows={subject_rows}, total_rows={len(rows)}, '
                f'total_images={total_images}', flush=True)

    # Fill optional columns and enforce the hosted TSV order for stable loading.
    data = pd.DataFrame(rows)
    for col in OUTPUT_COLUMNS:
        if col not in data:
            data[col] = ''
    data = data[OUTPUT_COLUMNS]
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    # Quote every field, matching the legacy TSV and protecting any remaining
    # tabs/newlines/quotes from naive CSV readers.
    data.to_csv(args.output, sep='\t', index=False, quoting=csv.QUOTE_ALL)

    digest = file_md5(args.output)
    # Sidecar records provenance and the exact generation options for review.
    metadata = {
        'repo_id': REPO_ID,
        'revision': args.revision,
        'resolved_sha': resolved_sha,
        'output': osp.abspath(args.output),
        'rows': len(data),
        'images': total_images,
        'md5': digest,
        'image_format': image_format,
        'image_ext': image_ext,
        'convert_rgb': args.convert_rgb,
        'drop_none_options': not args.keep_none_options,
        'keep_answers': args.keep_answers,
        'sanitize_text': not args.no_sanitize,
        'subjects': selected,
        'generated_at': datetime.now(timezone.utc).isoformat(),
    }
    with open(args.output + '.meta.json', 'w') as fout:
        json.dump(metadata, fout, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
