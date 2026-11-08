#!/usr/bin/env python3
"""Convert MemLens dataset_*.json files to VLMEvalKit TSV format.

Usage:
    python scripts/build_memlens_tsv.py \
        --output_root /path/to/output_tsv_dir

By default, the source JSON files are downloaded from Hugging Face
(`xiyuRenBill/MEMLENS`). Pass --data-root to reuse local dataset_*.json files.
Image paths stored in the TSV are the same relative paths used by MemLens.
"""
import argparse
import csv
import json
import os

csv.field_size_limit(1 << 30)

REPO_ID = 'xiyuRenBill/MEMLENS'
INSTRUCTION_DEFAULT = 'Directly output the answer with no extra output.'

USER_TEMPLATE = (
    'Provide answers based on the given conversation history. '
    'If the question cannot be answered based on the given conversation, '
    'respond with "Insufficient information".\n'
    'Conversation:\n{context}\n\n'
    '{instruction}\n'
    'Question Date: {question_date}\n'
    'Question: {question}\n'
)

DATASET_SPLITS = {
    'MemLens_32K': 'dataset_32k.json',
    'MemLens_64K': 'dataset_64k.json',
    'MemLens_128K': 'dataset_128k.json',
    'MemLens_256K': 'dataset_256k.json',
}


def hf_download(filename, download_dir, token=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            'huggingface_hub is required to download MemLens assets. '
            'Install it with `pip install huggingface_hub`, or pass --data-root.'
        ) from err

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type='dataset',
        local_dir=download_dir,
        token=token,
    )


def prepare_json_file(filename, data_root, download_dir, token=None, skip_existing=False):
    if data_root:
        path = os.path.join(data_root, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing MemLens source JSON: {path}')
        return path

    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, filename)
    if skip_existing and os.path.exists(local_path):
        return local_path
    print(f'downloading {filename} from {REPO_ID} ...')
    return hf_download(filename, download_dir, token=token)


def extract_image_path(img_info):
    """Extract the relative image path using the same keys as MemLens utils.py."""
    if isinstance(img_info, str):
        return img_info
    if not isinstance(img_info, dict):
        return ''
    img_path = (
        img_info.get('file')
        or img_info.get('path')
        or img_info.get('file_path')
        or img_info.get('img_file')
    )
    if isinstance(img_path, list):
        img_path = img_path[0] if img_path else ''
    return img_path or ''


def build_context(item):
    """Flatten haystack_sessions into (context_text, image_path_list).

    context_text has <image> tokens in-place; image_path_list contains the
    corresponding relative paths (under release_images/) in order.
    """
    parts = []
    images = []

    sessions = item.get('haystack_sessions', [])
    dates = item.get('haystack_dates', [])

    for i, session in enumerate(sessions, 1):
        if isinstance(session, dict):
            date_str = session.get('date', 'unknown')
            turns = session.get('session', [])
        else:
            date_str = dates[i - 1] if i - 1 < len(dates) else 'unknown'
            turns = session

        parts.append(f'\n=== Session {i} (Date: {date_str}) ===\n')

        for turn in turns:
            role = '[User]: ' if turn.get('role') == 'user' else '[Assistant]: '
            parts.append(role)

            text = turn.get('content', '')
            turn_images = turn.get('images', [])

            resolved_paths = [p for p in (extract_image_path(img) for img in turn_images) if p]

            if resolved_paths:
                if text.count('<image>') > 0:
                    # <image> tokens already embedded in content — keep in place
                    images.extend(resolved_paths)
                else:
                    # No tokens in text — prepend one token per image
                    for path in resolved_paths:
                        parts.append('<image> ')
                        images.append(path)
            else:
                text = text.replace('<image>', '')

            text = text.strip()
            if text:
                parts.append(text)
            parts.append('\n')

    return ''.join(parts), images


def build_row(item, row_id):
    context, image_list = build_context(item)
    question_text = USER_TEMPLATE.format(
        context=context,
        instruction=INSTRUCTION_DEFAULT,
        question_date=item.get('question_date', 'unknown'),
        question=item.get('question', ''),
    )
    return {
        'index': row_id,
        'question_id': item.get('question_id', ''),
        'question': question_text,
        'answer': item.get('answer', ''),
        'image_path': json.dumps(image_list, ensure_ascii=False),
        'question_type': item.get('question_type', ''),
        'question_date': item.get('question_date', ''),
    }


def convert(input_json, output_tsv, dataset_name):
    with open(input_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    data = raw.get('data', raw) if isinstance(raw, dict) else raw

    fieldnames = ['index', 'question_id', 'question', 'answer',
                  'image_path', 'question_type', 'question_date']

    os.makedirs(os.path.dirname(output_tsv) or '.', exist_ok=True)
    with open(output_tsv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for i, item in enumerate(data):
            writer.writerow(build_row(item, i))

    print(f'[{dataset_name}] wrote {len(data)} rows -> {output_tsv}')


def main():
    parser = argparse.ArgumentParser(description='Build VLMEvalKit TSVs for MemLens.')
    parser.add_argument(
        '--data-root',
        default=None,
        help='Optional directory containing dataset_32k.json etc. '
             'If omitted, files are downloaded from Hugging Face.',
    )
    parser.add_argument(
        '--download-dir',
        default=None,
        help='Where to store downloaded JSON files. Defaults to <output-root>/downloads.',
    )
    parser.add_argument(
        '--output-root',
        required=True,
        help='Directory where MemLens_*.tsv files will be written.',
    )
    parser.add_argument(
        '--token',
        default=os.environ.get('HF_TOKEN'),
        help='Optional Hugging Face token. Defaults to HF_TOKEN.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Reuse downloaded JSON files when present.',
    )
    parser.add_argument('--splits', nargs='+', default=list(DATASET_SPLITS.keys()),
                        choices=list(DATASET_SPLITS.keys()),
                        help='Which splits to convert (default: all four).')
    args = parser.parse_args()

    download_dir = args.download_dir or os.path.join(args.output_root, 'downloads')
    for split in args.splits:
        json_file = DATASET_SPLITS[split]
        input_path = prepare_json_file(
            json_file,
            data_root=args.data_root,
            download_dir=download_dir,
            token=args.token,
            skip_existing=args.skip_existing,
        )
        output_path = os.path.join(args.output_root, f'{split}.tsv')
        convert(input_path, output_path, split)


if __name__ == '__main__':
    main()
