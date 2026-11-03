"""ImgMT text-in-image translation (paper Table 4, English -> X)."""
import json
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

from vlmeval.smp import dump, get_intermediate_file_path, load, read_ok
from .image_base import ImageBaseDataset
from .utils.imgmt import PROMPT_TEMPLATE, parse_prediction


class ImgMT(ImageBaseDataset):
    TYPE = 'VQA'
    force_use_dataset_prompt = True
    REPO_ID = 'sarkau/ImgMT-Dataset-For-Evaluating-Text-Image-Translation'
    REVISION = 'f8c7446c445630c3e0f3086a71e4bc632bcb5c2b'

    # Language codes present in each split at REVISION.
    SOURCE_LANGUAGES = {
        'ImgMT': (
            'af an ar as ast az be bg bn br ca ckb cs cy da de dty el en eo es et eu fa fi fr gl gu he hi hr '
            'hu hy id it ja ka kn ko ks ku la lb lt lv mkd ml mr ms my nb ne nl nn no oc pa pl pt ro ru sa '
            'sat sk sl sr sr_latn su sv ta te th tl tr uk ur uz vec vi yue zh zh_hans zh_hant '
        ).split(),
        'ImgMT_DEV': (
            'af an ar as az ban be bg bn br bs ca cs cy da de el en eo es et eu fa fi fr ga gl gu he hi hr '
            'hu hy id is it ja ka km kn ko ku la lb lt lv mk mkd ml mr ms my nb ne nl no or pa pl pt ro ru '
            'sa sat sd sh shn si sk sl sr sv ta te tg th tl tr uk ur uz vi xmf yue zh zh_hans zh_hant zh_tw '
        ).split(),
        'ImgMT_TRAIN': (
            'ab af alb an ang ar arz as ast az ba be bg bjn bn br bs bxr ca ce ckb cs cy da de dsb dty el en '
            'eo es et eu ext fa fi fr fy ga gd gl gu he hi hr hsb hu hy id io is it ja ka kk km kn ko ks ku '
            'ky la lb lmo lt lv mag mk mkd ml mn mr ms my nap nb ne nl nn no oc or pa pa_guru pl ps pt ro ru '
            'rue sa sat sco sd se sh si sk sl smn so sr su sv sw ta te tg th tl tok tr tt uk ur uz vi war '
            'wuu xmf yue zh zh_hans zh_hant zh_tw '
        ).split(),
    }

    def __init__(self, dataset='ImgMT_en', comet_checkpoint=None):
        if dataset not in self.supported_datasets():
            raise ValueError(f'Unknown ImgMT dataset: {dataset}')
        preset = next(name for name in self.SOURCE_LANGUAGES
                      if dataset.removeprefix(name + '_') in self.SOURCE_LANGUAGES[name])
        self.split = {'ImgMT': 'test', 'ImgMT_DEV': 'dev', 'ImgMT_TRAIN': 'train'}[preset]
        self.source_lang = dataset[len(preset) + 1:]
        self.comet_checkpoint = comet_checkpoint
        super().__init__(dataset)

    @classmethod
    def supported_datasets(cls):
        return [f'{preset}_{language}'
                for preset, languages in cls.SOURCE_LANGUAGES.items() for language in languages]

    def load_data(self, dataset):
        self.data_root = Path(snapshot_download(
            repo_id=self.REPO_ID, repo_type='dataset', revision=self.REVISION,
            allow_patterns=[f'data/{self.split}/*/*.json',
                            f'data/{self.split}/*/png/{self.source_lang}.png']))
        rows = []
        for reference in sorted((self.data_root / 'data' / self.split).glob('*/*.json')):
            raw = json.loads(reference.read_text(encoding='utf-8'))
            src, tgt = raw['source_language'], raw['target_language']
            if self.source_lang not in (src, tgt):
                continue
            if tgt == self.source_lang:
                src, tgt = tgt, src
            image = Path(self.img_root) / self.split / reference.parent.name / f'{src}.png'
            if not read_ok(str(image)):
                source = reference.parent / 'png' / f'{src}.png'
                if not read_ok(str(source)):
                    raise FileNotFoundError(f'Missing or invalid ImgMT image: {source}')
                image.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, image)
            rows.append(dict(index=f'{reference.parent.name}_{src}-{tgt}',
                             image_path=str(image.resolve()), reference=str(reference.resolve()),
                             source_language=src, target_language=tgt,
                             question=PROMPT_TEMPLATE.replace('<source_language>', src).replace(
                                 '<target_language>', tgt)))
        if not rows:
            raise ValueError(f'No {self.source_lang} translation pairs under {self.data_root}/data/{self.split}')
        return pd.DataFrame(rows)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        return [dict(type='text', value=line['question']),
                dict(type='image', value=self.dump_image(line)[0])]

    def evaluate(self, eval_file, **judge_kwargs):
        # Preserve official matching, merging, rounding and per-pair averaging.
        try:
            from comet.models import download_model, load_from_checkpoint
        except ImportError:
            print('ImgMT evaluation requires unbabel-comet. Install it with: pip install unbabel-comet')
            raise
        from .utils.imgmt.evaluation import Evaluator, PipelineDataLoader

        checkpoint = self.comet_checkpoint or download_model('Unbabel/wmt22-comet-da')
        comet = load_from_checkpoint(checkpoint)
        details = []
        metadata = self.data.set_index('index')
        for _, line in load(eval_file).iterrows():
            item = metadata.loc[str(line['index'])]
            loader = PipelineDataLoader(item['source_language'], item['target_language'])
            gold = loader.load_gold(item['reference'])
            try:
                prediction = parse_prediction(line['prediction'], item['image_path'])
            except (ValueError, TypeError) as error:
                # Official run_system skips invalid responses; record exclusions explicitly.
                details.append(dict(index=line['index'], valid=False, error=str(error)))
                continue
            predicted = dict(text_bounding_boxes=prediction['source_text_bounding_boxes'],
                             texts=prediction['source_texts'], translated_texts=prediction['target_texts'])
            evaluator = Evaluator(gold, predicted)
            cer, area, count = evaluator.evaluate_text_detection(merging=True, rendered=False)
            chrf = evaluator.evaluate_translation('chrf', merging=True)
            score = evaluator.evaluate_translation('comet', merging=True, model=comet)
            details.append(dict(index=line['index'], valid=True, CER=cer, BBA=area,
                                BBC=count, chrF=chrf, COMET=score))
        metrics = ['chrF', 'COMET', 'CER', 'BBA', 'BBC']
        details = pd.DataFrame(details).reindex(columns=['index', 'valid', 'error'] + metrics)
        # *_all are custom full-case means, not the official valid-case metrics.
        # Keep each valid case's score unchanged. Treat JSON/schema failures
        # (including API failure messages that cannot be parsed) as empty output:
        # chrF=0: no translated content matches; BBA=BBC=0: no detected boxes;
        # CER=1: deleting all nonempty reference text costs one edit per character.
        # COMET=0 is an explicit failure penalty, not a model prediction for an
        # empty translation or a guaranteed minimum of COMET's score range.
        # The official empty-output CER shortcut returns 0; override it only for
        # invalid cases in *_all. Valid cases, including valid empty JSON, retain
        # the official calculation. No unmatched-block rules are changed.
        empty_scores = dict(chrF=0.0, COMET=0.0, CER=1.0, BBA=0.0, BBC=0.0)
        for key in metrics:
            details[key + '_all'] = details[key].where(details['valid'].eq(True), empty_scores[key])
        dump(details, get_intermediate_file_path(eval_file, '_details', 'tsv'))
        valid = details[details['valid'].eq(True)]
        result = {key: float(valid[key].mean()) if len(valid) else None for key in metrics}
        result.update({key + '_all': float(details[key + '_all'].mean()) if len(details) else None
                       for key in metrics})
        result.update(total=len(details), valid=len(valid), invalid=len(details) - len(valid))
        dump(result, get_intermediate_file_path(eval_file, '_score', 'json'))
        return result
