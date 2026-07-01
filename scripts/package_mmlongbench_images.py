import argparse
import os
import os.path as osp
import tarfile


REPO_ID = 'ZhaoweiWang/MMLongBench'
IMAGE_ARCHIVES = [
    '1_vrag_image.tar.gz',
    '2_vh_image.tar.gz',
    '2_mm-niah_image.tar.gz',
    '3_icl_image.tar.gz',
    '4_summ_image.tar.gz',
    '5_docqa_image.tar.gz',
]


def safe_extract(tar, path):
    root = osp.abspath(path)
    for member in tar.getmembers():
        target = osp.abspath(osp.join(root, member.name))
        if not (target == root or target.startswith(root + os.sep)):
            raise RuntimeError(f'Unsafe path in tar archive: {member.name}')
    tar.extractall(root)


def tar_has_top_level_mmlb_image(tar_path):
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            name = member.name.lstrip('./')
            if not name:
                continue
            return name.split('/', 1)[0] == 'mmlb_image'
    return False


def extract_archive(tar_path, output_root):
    has_root = tar_has_top_level_mmlb_image(tar_path)
    extract_root = output_root if has_root else osp.join(output_root, 'mmlb_image')
    os.makedirs(extract_root, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tar:
        safe_extract(tar, extract_root)
    return extract_root


def download_archive(filename, download_dir, token=None):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as err:
        raise ImportError(
            'huggingface_hub is required. Install it with `pip install huggingface_hub`.'
        ) from err

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type='dataset',
        local_dir=download_dir,
        token=token,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Download and extract official MMLongBench image tarballs from Hugging Face.'
    )
    parser.add_argument(
        '--output-root',
        required=True,
        help='Directory that will contain mmlb_image/. For VLMEvalKit, use $LMUData/images.',
    )
    parser.add_argument(
        '--download-dir',
        default=None,
        help='Where to store downloaded tar.gz files. Defaults to <output-root>/downloads.',
    )
    parser.add_argument(
        '--token',
        default=os.environ.get('HF_TOKEN'),
        help='Optional Hugging Face token. Defaults to HF_TOKEN.',
    )
    parser.add_argument(
        '--archive',
        nargs='+',
        default=IMAGE_ARCHIVES,
        choices=IMAGE_ARCHIVES,
        help='Subset of official image archives to download and extract.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Reuse local tar.gz files when present instead of downloading again.',
    )
    args = parser.parse_args()

    output_root = osp.abspath(args.output_root)
    download_dir = osp.abspath(args.download_dir or osp.join(output_root, 'downloads'))
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(download_dir, exist_ok=True)

    for archive in args.archive:
        local_tar = osp.join(download_dir, archive)
        if args.skip_existing and osp.exists(local_tar):
            tar_path = local_tar
            print(f'use existing: {tar_path}')
        else:
            print(f'downloading {archive} from {REPO_ID} ...')
            tar_path = download_archive(archive, download_dir, token=args.token)
            print(f'downloaded: {tar_path}')

        extract_root = extract_archive(tar_path, output_root)
        print(f'extracted {archive} under: {extract_root}')

    print(f'done. mmlb_image root should be: {osp.join(output_root, "mmlb_image")}')


if __name__ == '__main__':
    main()
