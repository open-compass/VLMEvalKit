#!/usr/bin/env python3
"""Download and extract LongDocURL PNG images into the VLMEvalKit cache."""

import argparse
import os

from vlmeval.dataset.longdocurl import LongDocURL
from vlmeval.smp import LMUDataRoot


def main():
    parser = argparse.ArgumentParser(description='Prepare LongDocURL image files.')
    parser.add_argument(
        '--image-root',
        default=None,
        help='Target image root. Defaults to LMUData/images/LongDocURL/pdf_pngs.',
    )
    args = parser.parse_args()

    if args.image_root:
        os.environ['LONGDOCURL_IMAGE_ROOT'] = args.image_root
    else:
        os.environ.setdefault(
            'LONGDOCURL_IMAGE_ROOT',
            os.path.join(LMUDataRoot(), 'images', 'LongDocURL', 'pdf_pngs'),
        )

    dataset = LongDocURL('LongDocURL')
    rel_paths = []
    for _, row in dataset.data.iterrows():
        rel_paths.extend(row['image_path'])
    dataset._ensure_images(rel_paths)
    print(f'LongDocURL images are prepared under {dataset._img_root()}')


if __name__ == '__main__':
    main()
