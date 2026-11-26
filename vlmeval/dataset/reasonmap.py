# ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps (CVPR 2026)
#   Paper: https://arxiv.org/abs/2505.18675
#   Code:  https://github.com/fscdc/ReasonMap
#   Data:  https://huggingface.co/datasets/FSCCS/ReasonMap
#
# Route planning on high-resolution metro maps. Every sample asks for a route between two stations
# twice: a short question (line names + transfer stops) and a long question (also the via stops or
# their number). Scoring is rule-based and identical to the official evaluation code.
#
# If any problem occurs, please open an issue at https://github.com/fscdc/ReasonMap.

import base64
import json
import os
import os.path as osp

import pandas as pd

from vlmeval.smp import (LMUDataRoot, d2df, download_file, dump, get_intermediate_file_path,
                         get_logger, load)
from .image_base import ImageBaseDataset
from .utils.reasonmap import (clean_station_table, extract_route_answer, reasonmap_weight,
                              score_long, score_short)

logger = get_logger(__name__)

HF_REPO = 'FSCCS/ReasonMap'
# The test set (11 cities, 312 samples) is published as the ``validation`` split.
HF_TEST_FILE = 'data/reasonmap-1008-validation.parquet'
# Station tables used by the scorer. They are not in the HF dataset for every city, so they are
# taken from the official repository (pinned to a commit). To work offline, put the files under
# ``$LMUData/ReasonMap/stations/{country}/{city}.json``.
STATIONS_URL = ('https://raw.githubusercontent.com/fscdc/ReasonMap/'
                '8da0141fc359bb7c7536a507a3374959c5337691/stations/{country}/{city}.json')


def reasonmap_data_dir():
    """Where the converted ReasonMap / ReasonMap-Plus annotations are cached."""
    path = osp.join(LMUDataRoot(), 'ReasonMap')
    os.makedirs(path, exist_ok=True)
    return path


def reasonmap_maps_dir():
    """Map images, shared by ReasonMap and ReasonMap-Plus (one PNG per city)."""
    return osp.join(LMUDataRoot(), 'images', 'ReasonMap')


def map_image_name(country, city):
    return f'{country}/{city}.png'


def hf_dataset_file(repo_id, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, filename, repo_type='dataset')


def atomic_dump(data, path):
    root, ext = osp.splitext(path)
    tmp = f'{root}.tmp{os.getpid()}{ext}'
    dump(data, tmp)
    os.replace(tmp, path)


# An .xlsx cell holds at most 32,767 characters; longer predictions are cut when the file is written.
XLSX_CELL_LIMIT = 32767


def warn_truncated_predictions(eval_file, predictions):
    """Warn when predictions read from an .xlsx file were probably cut to the cell limit.

    Thinking models often write more than 32,767 characters, so the cut drops the final answer and
    the sample is scored as wrong. Saving predictions as TSV (``PRED_FORMAT=tsv``) avoids this.
    """
    if not str(eval_file).lower().endswith('.xlsx'):
        return 0
    n = int((predictions.astype(str).str.len() >= XLSX_CELL_LIMIT).sum())
    if n:
        logger.warning(
            f'{n} of {len(predictions)} predictions in {eval_file} are {XLSX_CELL_LIMIT} characters long, '
            'the .xlsx cell limit, so they were probably truncated and lost their final answer. '
            'Rerun inference with PRED_FORMAT=tsv for long-output (e.g. thinking) models.')
    return n


def ensure_map_images(cities, maps_dir):
    """Write the map of every (country, city) to ``maps_dir`` (decoded from the ReasonMap test set).

    The maps are stored as base64 PNGs in the parquet file; each city is decoded once.
    """
    missing = {(country, city) for country, city in cities
               if not osp.exists(osp.join(maps_dir, map_image_name(country, city)))}
    if not missing:
        return
    import pyarrow.parquet as pq

    logger.info(f'Extracting {len(missing)} ReasonMap map image(s) to {maps_dir}')
    parquet = pq.ParquetFile(hf_dataset_file(HF_REPO, HF_TEST_FILE))
    for batch in parquet.iter_batches(batch_size=8, columns=['country', 'city', 'figure']):
        for country, city, figure in zip(*(batch.column(i).to_pylist() for i in range(3))):
            if (country, city) not in missing:
                continue
            target = osp.join(maps_dir, map_image_name(country, city))
            os.makedirs(osp.dirname(target), exist_ok=True)
            tmp = f'{target}.tmp{os.getpid()}'
            with open(tmp, 'wb') as f:
                f.write(base64.b64decode(figure))
            os.replace(tmp, target)
            missing.discard((country, city))
        if not missing:
            break
    if missing:
        raise FileNotFoundError(f'No map image for {sorted(missing)} in {HF_REPO}/{HF_TEST_FILE}')


def load_station_tables(cities):
    """``{"country/city": {route name: [stations]}}`` for the given cities, downloaded once."""
    root = osp.join(reasonmap_data_dir(), 'stations')
    tables = {}
    for country, city in sorted(cities):
        path = osp.join(root, country, f'{city}.json')
        if not osp.exists(path):
            os.makedirs(osp.dirname(path), exist_ok=True)
            tmp = f'{path}.tmp{os.getpid()}'
            download_file(STATIONS_URL.format(country=country, city=city), tmp)
            os.replace(tmp, path)
        with open(path, encoding='utf-8') as f:
            tables[f'{country}/{city}'] = clean_station_table(json.load(f))
    return tables


class ReasonMap(ImageBaseDataset):
    """ReasonMap test set: 312 samples x (short, long) questions = 624 rows."""

    TYPE = 'VQA'
    MODALITY = 'IMAGE'
    # The questions spell out the required answer format; model-specific VQA prompts
    # (e.g. "answer with a single word") would break it.
    force_use_dataset_prompt = True
    DATASET_URL = {'ReasonMap': ''}  # built from the official HF dataset in ``load_data``
    DATASET_MD5 = {'ReasonMap': None}
    PASS_K_HIT_COL = 'acc'

    _COLUMNS = ('index', 'sample_idx', 'question_type', 'question', 'image_path', 'country', 'city',
                'station_1', 'station_2', 'difficulty_question', 'difficulty_city', 'city_line_count',
                'city_transfer_count', 'question_transfer_count', 'routes')

    @classmethod
    def supported_datasets(cls):
        return ['ReasonMap']

    # ------------------------------------------------------------------ data

    def load_data(self, dataset):
        self.img_root = reasonmap_maps_dir()
        data_file = osp.join(reasonmap_data_dir(), 'ReasonMap.tsv')
        data = load(data_file) if osp.exists(data_file) else None
        if data is None or not set(self._COLUMNS) <= set(data.columns):
            data = self._build_from_hf()
            atomic_dump(data, data_file)
        cities = set(zip(data['country'], data['city']))
        self.stations = load_station_tables(cities)
        ensure_map_images(cities, self.img_root)
        return data

    @classmethod
    def _build_from_hf(cls):
        import pyarrow.parquet as pq

        logger.info(f'Converting {HF_REPO} ({HF_TEST_FILE}) for VLMEvalKit')
        parquet = pq.ParquetFile(hf_dataset_file(HF_REPO, HF_TEST_FILE))
        columns = [c for c in parquet.schema_arrow.names if c not in ('figure', 'json')]
        records = parquet.read(columns=columns).to_pylist()

        rows = []
        for sample_idx, r in enumerate(records):
            common = dict(
                sample_idx=sample_idx,
                image_path=map_image_name(r['country'], r['city']),
                country=r['country'],
                city=r['city'],
                station_1=r['station_1'],
                station_2=r['station_2'],
                difficulty_question=r['difficulty_question'],
                difficulty_city=r['difficulty_city'],
                city_line_count=r['city_line_count'],
                city_transfer_count=r['city_transfer_count'],
                question_transfer_count=r['question_transfer_count'],
                routes=json.dumps(r['routes'], ensure_ascii=False),
            )
            for question_type in ('short', 'long'):
                rows.append(dict(index=len(rows), question_type=question_type,
                                 question=r[f'question_{question_type}'], **common))
        return pd.DataFrame(rows)[list(cls._COLUMNS)]

    def dump_image(self, line):
        return [osp.join(self.img_root, line['image_path'])]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        msgs = [dict(type='image', value=p) for p in self.dump_image(line)]
        msgs.append(dict(type='text', value=line['question']))
        return msgs

    # ------------------------------------------------------------------ evaluation

    def evaluate(self, eval_file, **judge_kwargs):
        preds = load(eval_file)
        warn_truncated_predictions(eval_file, preds['prediction'])
        meta = self.data.set_index(self.data['index'].astype(str))

        records = []
        for _, item in preds.iterrows():
            row = meta.loc[str(item['index'])]
            sample = dict(
                country=row['country'],
                city=row['city'],
                station_1=row['station_1'],
                station_2=row['station_2'],
                routes=json.loads(row['routes']),
                question_long=row['question'],
            )
            metro_data = self.stations[f"{row['country']}/{row['city']}"]
            answer = extract_route_answer(item['prediction'])
            scorer = score_short if row['question_type'] == 'short' else score_long
            scores = scorer(answer, sample, metro_data)
            records.append(dict(
                index=item['index'],
                sample_idx=row['sample_idx'],
                question_type=row['question_type'],
                country=row['country'],
                city=row['city'],
                difficulty_question=row['difficulty_question'],
                difficulty_city=row['difficulty_city'],
                weight=reasonmap_weight(row['difficulty_question'], row['difficulty_city']),
                prediction=item['prediction'],
                extracted_answer=answer,
                **scores,
            ))
        detail = pd.DataFrame(records)
        dump(detail, get_intermediate_file_path(eval_file, '_results'))

        metrics = self.summarize(detail)
        dump(d2df(metrics), get_intermediate_file_path(eval_file, '_acc'))
        return metrics

    @staticmethod
    def summarize(detail):
        """Paper metrics: difficulty-weighted accuracy (%) and map score, for short and long questions.

        ``Map Score`` is the plain average and ``Weighted Map Score`` uses the same weights as the
        accuracy; the map score is out of 20 (short) / 40 (long).
        """
        metrics = {}
        for question_type, label in (('short', 'Short'), ('long', 'Long')):
            sub = detail[detail['question_type'] == question_type]
            if not len(sub):
                continue
            weight = sub['weight'].astype(float)
            metrics[f'Weighted Acc ({label})'] = 100 * float((weight * sub['acc']).sum() / weight.sum())
            metrics[f'Acc ({label})'] = 100 * float(sub['acc'].mean())
            metrics[f'Weighted Map Score ({label})'] = float((weight * sub['map_score']).sum() / weight.sum())
            metrics[f'Map Score ({label})'] = float(sub['map_score'].mean())
            if question_type == 'long':
                metrics['Via Stops Score (Long)'] = float(sub['via_stops_score'].mean())
                metrics['Num Via Stops Score (Long)'] = float(sub['num_via_stop_score'].mean())
            metrics[f'Samples ({label})'] = int(len(sub))
        return metrics

    @classmethod
    def report_primary_metric(cls, metrics):
        if not isinstance(metrics, dict):
            return {}
        keys = ('Weighted Acc (Short)', 'Weighted Acc (Long)')
        return {k: float(metrics[k]) for k in keys if k in metrics}
