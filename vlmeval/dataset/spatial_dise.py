import os
import os.path as osp
import string
import tarfile
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import load, read_ok, toliststr
from .image_mcq import ImageMCQDataset


class SpatialDISE(ImageMCQDataset):
    """
    Spatial-DISE.

    Reference:
      Spatial-DISE: Dataset for 2D/3D spatial reasoning evaluation.
      https://huggingface.co/datasets/TACPS-liv/Spatial-DISE
    """

    TYPE = 'MCQ'
    MODALITY = 'IMAGE'

    REPO_ID = 'TACPS-liv/Spatial-DISE'
    SPLITS = {
        'Spatial-DISE': 'benchmark',
        'Spatial-DISE_MERGE': 'benchmark',
        'Spatial-DISE_BENCH': 'benchmark',
        'Spatial-DISE_BENCH_MERGE': 'benchmark',
        'Spatial-DISE_SEPARATE': 'benchmark',
        'Spatial-DISE_BENCH_SEPARATE': 'benchmark',
        'Spatial-DISE_TEST': 'test',
        'Spatial-DISE_TEST_MERGE': 'test',
        'Spatial-DISE_TEST_SEPARATE': 'test',
        'Spatial-DISE_VAL': 'val',
        'Spatial-DISE_VAL_MERGE': 'val',
        'Spatial-DISE_VAL_SEPARATE': 'val',
        'Spatial-DISE_TRAIN': 'train',
        'Spatial-DISE_TRAIN_MERGE': 'train',
        'Spatial-DISE_TRAIN_SEPARATE': 'train',
    }
    MODES = {
        name: ('separate' if name.endswith('_SEPARATE') else 'merge')
        for name in SPLITS
    }

    CATEGORY_ORDER = [
        '3D Combination',
        '3D Rotation',
        '3D Folding',
        '3D Projection',
        '3D Shape Finding',
        '2D Combination',
        '2D Rotation',
        '2D Folding',
        '2D Shape Finding',
        'Fold and Punch',
    ]
    DIFFICULTY_ORDER = ['easy', 'medium', 'hard']
    DISE_CATEGORY_ORDER = [
        'Intrinsic-Static',
        'Intrinsic-Dynamic',
        'Extrinsic-Static',
        'Extrinsic-Dynamic',
    ]
    MERGE_IMAGE_COLUMNS = [
        ('image', 'merged image'),
    ]
    SEPARATE_IMAGE_COLUMNS = [
        ('question_image_path', 'separate question image'),
        ('question_image_1_path', 'separate question image 1'),
        ('question_image_2_path', 'separate question image 2'),
        ('option_a_image_path', 'separate option A image'),
        ('option_b_image_path', 'separate option B image'),
        ('option_c_image_path', 'separate option C image'),
        ('option_d_image_path', 'separate option D image'),
    ]

    DATASET_URL = {name: '' for name in SPLITS}
    DATASET_MD5 = {}

    _TAR_INDEX_CACHE = {}

    def __init__(self, dataset='Spatial-DISE', skip_noimg=True):
        self.image_mode = self.MODES.get(dataset, 'merge')
        super().__init__(dataset=dataset, skip_noimg=skip_noimg)

    @staticmethod
    def _dataset_root():
        local_root = os.environ.get('SPATIAL_DISE_ROOT')
        if local_root:
            local_root = osp.expanduser(osp.expandvars(local_root))
            if osp.isdir(local_root):
                return local_root

        return snapshot_download(
            repo_id=SpatialDISE.REPO_ID,
            repo_type='dataset',
            revision='main',
            allow_patterns=['dataset/*.csv', 'DISE-bench/DISE-benchmark.csv', 'image/*.tar'],
        )

    @staticmethod
    def _csv_path_to_tar_member(path):
        path = str(path).strip()
        if path.startswith('images/'):
            path = path[len('images/'):]
        return path.lstrip('/\\')

    @staticmethod
    def _option_letters(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return list(string.ascii_uppercase[:4])
        letters = []
        raw_options = value if isinstance(value, list) else str(value).replace('，', ',').split(',')
        for option in raw_options:
            option = option.strip().upper()
            if option and option[0] in string.ascii_uppercase and option[0] not in letters:
                letters.append(option[0])
        return letters or list(string.ascii_uppercase[:4])

    @classmethod
    def _image_refs(cls, row, tar_index, image_mode):
        refs = []
        missing = []
        seen = set()
        columns = cls.SEPARATE_IMAGE_COLUMNS if image_mode == 'separate' else cls.MERGE_IMAGE_COLUMNS
        for column, label in columns:
            value = row.get(column, '')
            if pd.isna(value):
                continue
            value = str(value).strip()
            if not value:
                continue
            member = cls._csv_path_to_tar_member(value)
            if member in seen:
                continue
            shard = tar_index.get(member)
            if shard is None:
                missing.append(f'{column}={value}')
                continue
            refs.append((label, member, shard))
            seen.add(member)
        return refs, missing

    @classmethod
    def _tar_index(cls, dataset_root):
        dataset_root = osp.abspath(dataset_root)
        if dataset_root in cls._TAR_INDEX_CACHE:
            return cls._TAR_INDEX_CACHE[dataset_root]

        image_dir = osp.join(dataset_root, 'image')
        tar_paths = sorted(Path(image_dir).glob('*.tar'))
        if not tar_paths:
            raise FileNotFoundError(f'No tar shards found under {image_dir}')

        tar_index = {}
        for tar_path in tar_paths:
            with tarfile.open(tar_path) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        tar_index[member.name] = str(tar_path)

        cls._TAR_INDEX_CACHE[dataset_root] = tar_index
        return tar_index

    def load_data(self, dataset):
        split = self.SPLITS[dataset]
        dataset_root = self._dataset_root()
        if split == 'benchmark':
            csv_path = osp.join(dataset_root, 'DISE-bench', 'DISE-benchmark.csv')
        else:
            csv_path = osp.join(dataset_root, 'dataset', f'{split}.csv')
        if not osp.isfile(csv_path):
            raise FileNotFoundError(f'Spatial-DISE split file not found: {csv_path}')

        raw = pd.read_csv(csv_path, skipinitialspace=True)
        raw.columns = [str(x).strip() for x in raw.columns]
        for col in raw.columns:
            if raw[col].dtype == object:
                raw[col] = raw[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        tar_index = self._tar_index(dataset_root)

        records = []
        missing = []
        for row_id, row in raw.iterrows():
            image_refs, row_missing = self._image_refs(row, tar_index, self.image_mode)
            if row_missing:
                missing.extend(row_missing)
                continue
            if not image_refs:
                missing.append(f'image={row.get("image", "")}')
                continue
            options = self._option_letters(row.get('options', ''))

            record = {
                'index': f'{split}_{row_id}',
                'question': str(row['question']).strip(),
                'answer': str(row['answer']).strip().upper(),
                'options': options,
                'image_path': [member for _, member, _ in image_refs],
                'image_shard': [shard for _, _, shard in image_refs],
                'image_role': [label for label, _, _ in image_refs],
                'image_mode': self.image_mode,
                'split': split,
                'category': row.get('category', ''),
                'difficulty': row.get('difficulty', ''),
                'source': row.get('source', ''),
                'dise_category': row.get('dise_category', ''),
            }
            for option in options:
                record[option] = option
            records.append(record)

        if missing:
            examples = ', '.join(map(str, missing[:5]))
            raise FileNotFoundError(
                f'{len(missing)} Spatial-DISE image references were not found in tar shards. '
                f'Examples: {examples}'
            )

        return pd.DataFrame.from_records(records)

    def dump_image(self, line):
        members = [self._csv_path_to_tar_member(x) for x in toliststr(line['image_path'])]
        shards = toliststr(line['image_shard'])
        if len(members) != len(shards):
            raise RuntimeError('Spatial-DISE image_path and image_shard lengths do not match.')

        root = osp.abspath(self.img_root)
        targets = []
        for member, shard in zip(members, shards):
            target = osp.abspath(osp.join(self.img_root, member))
            if not target.startswith(root + os.sep):
                raise RuntimeError(f'Unsafe Spatial-DISE image path: {member}')

            if not read_ok(target):
                os.makedirs(osp.dirname(target), exist_ok=True)
                with tarfile.open(shard) as tf:
                    image_file = tf.extractfile(member)
                    if image_file is None:
                        raise FileNotFoundError(f'{member} not found in {shard}')
                    with open(target, 'wb') as out:
                        out.write(image_file.read())
            targets.append(target)

        return targets

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = str(line['question']).strip()
        options = self._option_letters(line.get('options', ''))
        option_text = ', '.join(options)
        if str(line.get('image_mode', 'merge')) == 'separate':
            prompt = (
                f'{question}\n'
                'Images are provided as separate question/view/option images from the original sample. '
                f'Use all images together. The answer choices are labeled {option_text}. '
                'Please select the correct answer and respond with only one option letter.'
            )
        else:
            prompt = (
                f'{question}\n'
                f'The image contains answer choices labeled {option_text}. '
                'Please select the correct answer and respond with only one option letter.'
            )

        msgs = []
        for path in toliststr(tgt_path):
            msgs.append(dict(type='image', value=path))
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        score_fn = build_mcq_score_fn(**judge_kwargs)
        raw = eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=score_fn,
            group_col=['category', 'difficulty', 'dise_category'],
            order={
                'category': self.CATEGORY_ORDER,
                'difficulty': self.DIFFICULTY_ORDER,
                'dise_category': self.DISE_CATEGORY_ORDER,
            },
            dataset_name=getattr(self, 'dataset_name', 'Spatial-DISE'),
        )

        pretty = OrderedDict()
        pretty['overall'] = raw['overall']

        for category in self.CATEGORY_ORDER:
            key = f'category.{category}_accuracy'
            if key in raw:
                pretty[f'{category}_accuracy'] = raw[key]

        for difficulty in self.DIFFICULTY_ORDER:
            key = f'difficulty.{difficulty}_accuracy'
            if key in raw:
                pretty[f'{difficulty}_accuracy'] = raw[key]

        for dise_category in self.DISE_CATEGORY_ORDER:
            key = f'dise_category.{dise_category}_accuracy'
            if key in raw:
                pretty[f'{dise_category}_accuracy'] = raw[key]

        keys_str = ', '.join(pretty.keys())
        vals_str = ', '.join(f'{v:.3f}' for v in pretty.values())
        pretty['tabulated_keys'] = keys_str
        pretty['tabulated_results'] = vals_str
        return pretty
