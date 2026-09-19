import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_NAMES = {
    'qa': 'OmniMat_QA',
    'cal': 'OmniMat_CAL',
}


def lmu_data_root() -> Path:
    root = os.environ.get('LMUData')
    if root and Path(root).exists():
        return Path(root)
    return Path.home() / 'LMUData'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert OmniMat data to VLMEvalKit TSV files.')
    parser.add_argument(
        '--omnimat-root',
        type=Path,
        default=Path(__file__).resolve().parents[1] / 'omnimat',
        help='Path to the OmniMat root directory.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=lmu_data_root(),
        help='Directory for OmniMat_QA.tsv and OmniMat_CAL.tsv.',
    )
    parser.add_argument('--subset', choices=['qa', 'cal', 'all'], default='all')
    parser.add_argument('--qa-output', type=Path, default=None)
    parser.add_argument('--cal-output', type=Path, default=None)
    return parser.parse_args()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def split_refs(image_ref: Any) -> list[str]:
    if image_ref is None:
        return []
    raw = str(image_ref).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(',') if part.strip()]


def resolve_one(ref: str, image_dir: Path) -> Path | None:
    raw = ref.strip()
    candidates = [image_dir / raw]
    ref_path = Path(raw)
    if not ref_path.suffix:
        candidates.append(image_dir / f'{raw}.png')
        matches = sorted(image_dir.glob(f'{raw}.*'))
        base_matches = [path for path in matches if '(2)' not in path.name]
        candidates.extend(base_matches or matches)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def relative_image_name(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode('utf-8')


def resolve_refs(image_ref: Any, image_dir: Path, path_root: Path) -> tuple[list[Path], list[str], list[str]]:
    paths = []
    image_names = []
    missing = []
    for ref in split_refs(image_ref):
        resolved = resolve_one(ref, image_dir)
        if resolved is None:
            missing.append(ref)
        else:
            paths.append(resolved)
            image_names.append(relative_image_name(resolved, path_root))
    return paths, image_names, missing


def infer_category_name(stem: str, category_id: str, suffix: str) -> str:
    name = stem
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    name = name.replace(f'{category_id}_', '', 1)
    return name.replace('_', ' ')


def tsv_list_value(values: list[str]) -> str:
    if not values:
        return ''
    if len(values) == 1:
        return values[0]
    return json_dumps(values)


def convert_qa(qa_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings = []
    for rubric_file in sorted(qa_root.glob('*/*_QA_rubric.json')):
        category_id = rubric_file.parent.name
        category = infer_category_name(rubric_file.stem, category_id, '_QA_rubric')
        image_dir = rubric_file.parent / 'images'
        data = json.loads(rubric_file.read_text(encoding='utf-8'))
        for item in data:
            image_paths, image_names, missing = resolve_refs(item.get('image_url'), image_dir, qa_root)
            for ref in missing:
                warnings.append(f'QA {category_id}/{item.get("id")}: missing image {ref}')
            images = [encode_image(path) for path in image_paths]
            item_id = str(item.get('id', len(rows) + 1)).zfill(3)
            rows.append({
                'index': f'{category_id}_{item_id}',
                'subset': 'qa',
                'category_id': category_id,
                'category': category,
                'id': item_id,
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'image': tsv_list_value(images),
                'image_path': tsv_list_value(image_names),
                'image_url': item.get('image_url') or '',
                'multimodal': bool(item.get('multimodal', False)),
                'source_type': item.get('source_type', ''),
                'source_name': item.get('source_name', ''),
                'source_location': item.get('source_location', ''),
                'key_points': json_dumps(item.get('key_points', [])),
                'scoring_weights': json_dumps(item.get('scoring_weights', {})),
            })
    return pd.DataFrame(rows), warnings


def convert_cal(cal_root: Path) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    warnings = []
    for source_file in sorted(cal_root.glob('*/*/*_with_final_answers.jsonl')):
        category_id = source_file.parts[-3]
        category = infer_category_name(source_file.stem.replace('_with_final_answers', ''), category_id, '_Cal')
        image_dir = source_file.parent / 'images'
        with source_file.open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                image_paths, image_names, missing = resolve_refs(item.get('image_url'), image_dir, cal_root)
                for ref in missing:
                    warnings.append(f'CAL {category_id}/{item.get("id")}: missing image {ref}')
                images = [encode_image(path) for path in image_paths]
                item_id = str(item.get('id', len(rows) + 1)).zfill(3)
                rows.append({
                    'index': f'{category_id}_{item_id}',
                    'subset': 'cal',
                    'category_id': category_id,
                    'category': category,
                    'id': item_id,
                    'question': item.get('question', ''),
                    'answer': item.get('answer', ''),
                    'image': tsv_list_value(images),
                    'image_path': tsv_list_value(image_names),
                    'image_url': item.get('image_url') or '',
                    'multimodal': bool(item.get('multimodal', False)),
                    'source_type': item.get('source_type', ''),
                    'source_name': item.get('source_name', ''),
                    'source_location': item.get('source_location', ''),
                    'key_points': json_dumps(item.get('key_points', [])),
                    'final_answer_format': json_dumps(item.get('final_answer_format')),
                    'final_answer_list': json_dumps(item.get('final_answer_list', [])),
                })
    return pd.DataFrame(rows), warnings


def dump_tsv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, sep='\t', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)


def main() -> None:
    args = parse_args()
    omnimat_root = args.omnimat_root.resolve()
    output_dir = args.output_dir.resolve()

    tasks = ['qa', 'cal'] if args.subset == 'all' else [args.subset]
    all_warnings = []
    for subset in tasks:
        if subset == 'qa':
            data, warnings = convert_qa(omnimat_root / 'qa')
            output = args.qa_output or output_dir / f'{DATASET_NAMES[subset]}.tsv'
        else:
            data, warnings = convert_cal(omnimat_root / 'cal')
            output = args.cal_output or output_dir / f'{DATASET_NAMES[subset]}.tsv'
        dump_tsv(data, output)
        all_warnings.extend(warnings)
        print(f'wrote {len(data)} rows to {output}')

    for warning in all_warnings:
        print(f'warning: {warning}')


if __name__ == '__main__':
    main()
