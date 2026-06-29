"""Build the OmniDocBench v1.6 TSV locally (no HuggingFace download).

VLMEvalKit stores each benchmark as a single TSV. For OmniDocBench v1.6 the
ground truth is the official ``OmniDocBench.json`` (a list of per-page objects)
plus an ``images`` folder. This script converts them into the TSV layout the
toolkit expects:

    index   : row id (int)
    image   : base64-encoded page image (consumed during inference)
    answer  : the full per-page OmniDocBench GT JSON (consumed during evaluation)

By default the TSV is written to ``<VLMEvalKit>/OmniDocBench_v1.6.tsv`` (the repo
top level) -- exactly where the dataset class looks for it -- so after running
this script you can launch evaluation directly with ``--data OmniDocBench_v1.6``.

Usage
-----
    python -m vlmeval.dataset.OmniDocBench.build_v16_tsv \
        --json /path/to/OmniDocBench.json \
        --image-dir /path/to/images

Download the official v1.6 release (JSON + images) from OpenDataLab or
HuggingFace (``opendatalab/OmniDocBench``) beforehand.
"""

import argparse
import csv
import json
import os
import os.path as osp

import pandas as pd

from vlmeval.smp import encode_image_file_to_base64, md5

# VLMEvalKit repo root (this file lives at <root>/vlmeval/dataset/OmniDocBench/build_v16_tsv.py).
VLMEVALKIT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))


def default_v16_tsv_path():
    """Default OmniDocBench_v1.6 TSV location: top level of the VLMEvalKit repo."""
    return osp.join(VLMEVALKIT_ROOT, 'OmniDocBench_v1.6.tsv')


def _resolve_image_file(image_dir, image_path):
    """Locate the page image, tolerating both relative paths and bare names."""
    img_name = osp.basename(image_path)
    candidates = [
        osp.join(image_dir, image_path),  # image_dir + "images/xxx.jpg"
        osp.join(image_dir, img_name),    # image_dir + "xxx.jpg"
    ]
    for cand in candidates:
        if osp.exists(cand):
            return cand
    return None


def build_omnidocbench_v16_tsv(json_path, image_dir, output_path=None, fmt='JPEG'):
    """Convert the official OmniDocBench v1.6 JSON + images into a VLMEvalKit TSV.

    Args:
        json_path: Path to the official ``OmniDocBench.json`` (list of pages).
        image_dir: Directory that contains the page images.
        output_path: Destination TSV path. Defaults to
            ``<VLMEvalKit>/OmniDocBench_v1.6.tsv`` (the repo top level).
        fmt: Image encoding format for the base64 column (``JPEG`` or ``PNG``).

    Returns:
        The path to the generated TSV.
    """
    if output_path is None:
        output_path = default_v16_tsv_path()

    with open(json_path, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    assert isinstance(pages, list), f'Expected a list of pages in {json_path}, got {type(pages)}'

    rows = []
    missing = []
    for i, page in enumerate(pages):
        try:
            image_path = page['page_info']['image_path']
        except (KeyError, TypeError):
            missing.append(f'<page {i}: no page_info.image_path>')
            continue

        img_file = _resolve_image_file(image_dir, image_path)
        if img_file is None:
            missing.append(osp.basename(image_path))
            continue

        rows.append({
            'index': i,
            'image': encode_image_file_to_base64(img_file, fmt=fmt),
            'answer': json.dumps(page, ensure_ascii=False),
        })

    if not rows:
        raise RuntimeError(
            'No pages were converted. Check that --image-dir points to the folder '
            'that holds the OmniDocBench page images.'
        )

    os.makedirs(osp.dirname(osp.abspath(output_path)), exist_ok=True)
    df = pd.DataFrame(rows, columns=['index', 'image', 'answer'])
    # Mirror VLMEvalKit's dump_tsv convention (QUOTE_ALL) so load() reads it back cleanly.
    df.to_csv(output_path, sep='\t', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

    print(f'[OmniDocBench_v1.6] wrote {len(rows)} pages -> {output_path}')
    if missing:
        print(f'[OmniDocBench_v1.6] WARNING: {len(missing)} pages skipped (image not found), e.g. {missing[:5]}')
    file_md5 = md5(output_path)
    print(f'[OmniDocBench_v1.6] MD5: {file_md5}')
    print(
        '[OmniDocBench_v1.6] The dataset class reads this file directly from '
        f'{output_path}; no further configuration is required. '
        'Optionally set DATASET_MD5["OmniDocBench_v1.6"] to the value above to enable integrity checks.'
    )
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the OmniDocBench v1.6 TSV locally.')
    parser.add_argument('--json', required=True, help='Path to the official OmniDocBench.json (list of pages).')
    parser.add_argument('--image-dir', required=True, help='Directory containing the page images.')
    parser.add_argument(
        '--output', default=None,
        help='Output TSV path. Defaults to <VLMEvalKit>/OmniDocBench_v1.6.tsv (repo top level).')
    parser.add_argument('--fmt', default='JPEG', choices=['JPEG', 'PNG'], help='Base64 image encoding format.')
    args = parser.parse_args(argv)
    build_omnidocbench_v16_tsv(args.json, args.image_dir, output_path=args.output, fmt=args.fmt)


if __name__ == '__main__':
    main()
