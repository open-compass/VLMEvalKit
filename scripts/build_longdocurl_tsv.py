#!/usr/bin/env python3
"""Build a VLMEvalKit TSV for LongDocURL from the public Hugging Face JSONL."""

import argparse
import json
import os
import os.path as osp

import pandas as pd


REPO_ID = 'dengchao/LongDocURL'
DATA_FILE = 'LongDocURL_public_with_subtask_category.jsonl'


def hf_download(filename, download_dir, token=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            'huggingface_hub is required to download LongDocURL. '
            'Install it with `pip install huggingface_hub`, or pass --jsonl.'
        ) from err

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type='dataset',
        local_dir=download_dir,
        token=token,
    )


def json_dumps(value):
    return json.dumps(value, ensure_ascii=False)


def relative_image_path(path):
    path = str(path)
    marker = '/pdf_pngs/'
    if marker in path:
        return path.split(marker, 1)[1]
    return path.lstrip('/')


def row_from_sample(sample, idx):
    images = [relative_image_path(p) for p in sample.get('images', [])]
    answer = sample.get('answer', '')
    return {
        'index': idx,
        'question_id': sample.get('question_id', ''),
        'question': sample.get('question', ''),
        'answer': json_dumps(answer) if isinstance(answer, list) else answer,
        'image_path': json_dumps(images),
        'doc_no': sample.get('doc_no', ''),
        'total_pages': sample.get('total_pages', ''),
        'start_end_idx': json_dumps(sample.get('start_end_idx', [])),
        'question_type': sample.get('question_type', ''),
        'answer_format': sample.get('answer_format', ''),
        'task_tag': sample.get('task_tag', ''),
        'evidence_pages': json_dumps(sample.get('evidence_pages', [])),
        'evidence_sources': json_dumps(sample.get('evidence_sources', [])),
        'subTask': json_dumps(sample.get('subTask', [])),
        'detailed_evidences': sample.get('detailed_evidences', ''),
        'pdf_path': sample.get('pdf_path', ''),
    }


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Build VLMEvalKit TSV for LongDocURL.')
    parser.add_argument('--jsonl', default=None, help='Optional local LongDocURL JSONL path.')
    parser.add_argument('--download-dir', default=None, help='Directory for downloaded JSONL.')
    parser.add_argument('--output', required=True, help='Output TSV path, e.g. LongDocURL.tsv.')
    parser.add_argument('--token', default=os.environ.get('HF_TOKEN'), help='Optional Hugging Face token.')
    parser.add_argument('--limit', type=int, default=None, help='Optional limit for debugging.')
    args = parser.parse_args()

    jsonl_path = args.jsonl
    if jsonl_path is None:
        download_dir = args.download_dir or osp.join(osp.dirname(osp.abspath(args.output)), 'downloads')
        os.makedirs(download_dir, exist_ok=True)
        print(f'downloading {DATA_FILE} from {REPO_ID} ...')
        jsonl_path = hf_download(DATA_FILE, download_dir, token=args.token)

    samples = load_jsonl(jsonl_path)
    if args.limit is not None:
        samples = samples[:args.limit]
    data = pd.DataFrame([row_from_sample(sample, i) for i, sample in enumerate(samples)])
    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)
    data.to_csv(args.output, sep='\t', index=False)
    print(f'wrote {len(data)} rows -> {args.output}')


if __name__ == '__main__':
    main()
