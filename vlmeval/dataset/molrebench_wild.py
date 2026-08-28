"""MolRecBench-Wild benchmark backed by one self-contained TSV file."""

from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vlmeval.smp import LMUDataRoot, decode_base64_to_image_file, load
from .image_base import ImageBaseDataset

SMILES_DATASET = 'MolRecBench_Wild_SMILES'
SGRAPH_DATASET = 'MolRecBench_Wild_SGraph'
GRAPH_DATASET = 'MolRecBench_Wild_Graph'

DATASET_TRACKS = {
    SMILES_DATASET: ('SMILES', 'smiles'),
    SGRAPH_DATASET: ('SGraph', 's_graph'),
    GRAPH_DATASET: ('Graph', 'graph'),
}


class MolRecBenchWildDataset(ImageBaseDataset):
    """Load MolRecBench-Wild TSV, build prompts, and run its evaluator."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'

    # The prepared TSV is currently local. After uploading it, replace these
    # empty values with the same downloadable TSV URL for all three tracks.
    DATASET_URL = {name: '' for name in DATASET_TRACKS}
    DATASET_MD5 = {name: None for name in DATASET_TRACKS}

    def __init__(
        self,
        dataset: str = SMILES_DATASET,
        tsv_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if dataset not in DATASET_TRACKS:
            raise ValueError(f'unsupported MolRecBench-Wild dataset: {dataset}')
        self.track_name, self.track = DATASET_TRACKS[dataset]
        self.tsv_path = self._resolve_tsv_path(tsv_path)
        self._asset_paths: dict[str, str] = {}
        self._ground_truth: list[dict[str, Any]] = []
        super().__init__(dataset=dataset, skip_noimg=False)

    @classmethod
    def supported_datasets(cls) -> list[str]:
        return list(DATASET_TRACKS)

    @staticmethod
    def _resolve_tsv_path(tsv_path: str | os.PathLike[str] | None) -> Path:
        if tsv_path is not None:
            path = Path(tsv_path).expanduser().resolve()
        else:
            lmu_path = Path(LMUDataRoot()) / 'MolRecBench-Wild.tsv'
            repo_path = Path(__file__).resolve().parents[2] / 'data' / 'MolRecBench-Wild.tsv'
            path = lmu_path if lmu_path.is_file() else repo_path
        if not path.is_file():
            raise FileNotFoundError(
                f'MolRecBench-Wild TSV not found at {path}. '
                'Pass tsv_path or place MolRecBench-Wild.tsv under $LMUData.'
            )
        return path

    @staticmethod
    def _string(value: Any) -> str:
        if value is None or pd.isna(value):
            return ''
        return str(value)

    @staticmethod
    def _reference_key(value: Any) -> str:
        """Normalize TSV numeric references such as ``0.0`` back to ``0``."""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return MolRecBenchWildDataset._string(value)

    @staticmethod
    def _integer_index(value: Any) -> int:
        """Return a VLMEvalKit integer index without accepting lossy values."""
        if value is None or pd.isna(value) or isinstance(value, bool):
            raise ValueError(f'invalid integer index: {value!r}')
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'invalid integer index: {value!r}') from error
        if not number.is_integer():
            raise ValueError(f'invalid integer index: {value!r}')
        return int(number)

    @staticmethod
    def _resolve_image(reference: Any, image_map: Mapping[str, str]) -> str:
        value = MolRecBenchWildDataset._string(reference)
        seen: set[str] = set()
        while value and len(value) <= 64:
            if value in seen:
                raise ValueError(f'cyclic image reference: {value}')
            seen.add(value)
            if value not in image_map:
                raise ValueError(f'unknown image reference: {value}')
            value = image_map[value]
        if not value:
            raise ValueError('empty image value')
        try:
            base64.b64decode(value, validate=True)
        except Exception as error:
            raise ValueError('invalid base64 image value') from error
        return value

    def _materialize_assets(
        self,
        assets: pd.DataFrame,
        image_map: Mapping[str, str],
    ) -> None:
        asset_root = Path(self.img_root) / 'assets'
        asset_root.mkdir(parents=True, exist_ok=True)
        for _, row in assets.iterrows():
            index = self._string(row['index'])
            filename = Path(self._string(row['image_path'])).name
            if not filename:
                raise ValueError(f'asset {index} has no image_path')
            image = self._resolve_image(index, image_map)
            path = asset_root / filename
            if not path.is_file():
                decode_base64_to_image_file(image, str(path))
            self._asset_paths[index] = str(path)

    def load_data(self, dataset: str) -> pd.DataFrame:
        frame = load(str(self.tsv_path))
        if not isinstance(frame, pd.DataFrame):
            raise TypeError('MolRecBench-Wild TSV must load as a pandas DataFrame')
        required = {
            'index', 'record_type', 'sample_id', 'track', 'image', 'image_path',
            'question', 'answer', 'reference_image_1', 'reference_image_2',
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f'MolRecBench-Wild TSV is missing: {", ".join(missing)}')

        frame = frame.copy()
        frame['index'] = frame['index'].map(self._string)
        if frame['index'].duplicated().any():
            raise ValueError('MolRecBench-Wild TSV contains duplicate indices')
        image_map = dict(zip(frame['index'], frame['image'].map(self._string)))

        assets = frame[frame['record_type'] == 'asset']
        self._materialize_assets(assets, image_map)

        all_samples = frame[frame['record_type'] == 'sample']
        records: dict[str, dict[str, Any]] = {}
        for _, row in all_samples.iterrows():
            sample_id = self._string(row['sample_id'])
            record = json.loads(self._string(row['answer']))
            if not isinstance(record, dict) or record.get('id') != sample_id:
                raise ValueError(f'invalid annotation for {sample_id}')
            if sample_id in records and records[sample_id] != record:
                raise ValueError(f'inconsistent annotations for {sample_id}')
            records[sample_id] = record
        self._ground_truth = list(records.values())

        samples = all_samples[all_samples['track'] == self.track_name].copy()
        if samples.empty:
            raise ValueError(f'TSV contains no {self.track_name} rows')
        samples['image'] = [self._resolve_image(value, image_map) for value in samples['image']]
        samples['index'] = samples['index'].map(self._integer_index)
        samples['image_path'] = samples['sample_id'].map(lambda value: Path(str(value)).name)
        return samples.reset_index(drop=True)

    def build_prompt(self, line: int | pd.Series) -> list[dict[str, str]]:
        if isinstance(line, int):
            line = self.data.iloc[line]
        target = self.dump_image(line)[0]
        messages: list[dict[str, str]] = []
        if self.track == 'graph':
            for column in ('reference_image_1', 'reference_image_2'):
                reference = self._reference_key(line[column])
                if reference not in self._asset_paths:
                    raise ValueError(f'unknown Graph asset reference: {reference}')
                messages.append({'type': 'image', 'value': self._asset_paths[reference]})
        messages.extend([
            {'type': 'image', 'value': target},
            {'type': 'text', 'value': self._string(line['question'])},
        ])
        return messages

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        del judge_kwargs
        predictions = load(eval_file)
        if not isinstance(predictions, pd.DataFrame):
            raise TypeError('prediction file must load as a pandas DataFrame')
        if not {'index', 'prediction'} <= set(predictions.columns):
            raise ValueError('prediction file requires index and prediction columns')

        selected_indices = [self._integer_index(index) for index in self.data['index']]
        prediction_map = {
            self._integer_index(index): prediction
            for index, prediction in zip(predictions['index'], predictions['prediction'])
        }
        missing = [index for index in selected_indices if index not in prediction_map]
        if missing:
            raise ValueError(f'prediction file misses {len(missing)} samples')
        sample_ids = list(self.data['sample_id'].map(self._string))
        selected = pd.DataFrame({
            # The official converter keys records by the molecule image name,
            # while VLMEvalKit uses the TSV's integer index during inference.
            'index': sample_ids,
            'prediction': [prediction_map[index] for index in selected_indices],
        })

        from .utils.molrecbench_wild import convert_dataframe, score_records

        converted, _ = convert_dataframe(selected, track=self.track)
        selected_set = set(sample_ids)
        ground_truth = [row for row in self._ground_truth if row['id'] in selected_set]
        result = score_records(
            ground_truth,
            converted,
            self.track,
            full_gt_records=self._ground_truth,
            timeout_seconds=5,
            ignore_cistrans=True,
        )
        rows = []
        for split in ('Full', 'A', 'B', 'C'):
            metric = result.summary['subset_metrics'][split]
            rows.append({
                'track': self.track_name,
                'split': split,
                'total': metric['total_gt_records'],
                'scored': metric['scored_records'],
                'correct': metric['correct_records'],
                'accuracy': metric['accuracy'],
            })
        return pd.DataFrame(rows)
