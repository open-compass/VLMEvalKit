from vlmeval.smp import toliststr
from .image_base import ImageBaseDataset


class SArena(ImageBaseDataset):

    TYPE = "VQA"

    DATASET_URL = {
        "SArena": "https://huggingface.co/datasets/JoeLeelyf/SArena-VLMEvalKit/resolve/main/SArena.tsv",
        "SArena_MINI": "https://huggingface.co/datasets/JoeLeelyf/SArena-VLMEvalKit/resolve/main/SArena_MINI.tsv"
    }

    DATASET_MD5 = {
        "SArena": "2a747c13c063a6c9839c66611b61526c",
        "SArena_MINI": "c87fa82819a5fce652df40f6332266ff"
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        l2_cat = line['l2-category']
        question = line['question']
        msgs = []

        if 'I2SVG' in l2_cat:
            if self.meta_only:
                tgt_path = toliststr(line['image'])
            else:
                tgt_path = self.dump_image(line)

            if isinstance(tgt_path, list):
                msgs.extend([dict(type='image', value=p) for p in tgt_path])
            else:
                msgs.append(dict(type='image', value=tgt_path))

        else:
            if 'Edit' in l2_cat:
                question = question + '\nOnly output the svg code, no other text.'

        msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.sarena import evaluate_sarena
        return evaluate_sarena(eval_file, dataset=self.dataset_name)

    @classmethod
    def report_primary_metric(cls, metrics: dict | None) -> dict:
        if not isinstance(metrics, dict) or not metrics:
            return {}

        primary = {}

        # SArena-Icon
        try:
            t2s_clip = metrics['SArena-Icon|T2SVG|CLIP-Score-I2I']
            i2s_dino = metrics['SArena-Icon|I2SVG|DINO-Score']
            i2s_ssim = metrics['SArena-Icon|I2SVG|LPIPS']
            i2s_lpips = metrics['SArena-Icon|I2SVG|SSIM']

            icon_score = sum((
                0.3 * float(t2s_clip),
                0.3 * float(i2s_dino) * 100,
                0.2 * float(i2s_ssim) * 100,
                0.2 * (1 - float(i2s_lpips)) * 100,
            ))
            primary['Icon'] = icon_score
        except Exception:
            pass

        # SArena-Illustration
        try:
            t2s_clip = metrics['SArena-Illustration|T2SVG|CLIP-Score-I2I']
            i2s_dino = metrics['SArena-Illustration|I2SVG|DINO-Score']
            i2s_ssim = metrics['SArena-Illustration|I2SVG|LPIPS']
            i2s_lpips = metrics['SArena-Illustration|I2SVG|SSIM']

            illu_score = sum((
                0.3 * float(t2s_clip),
                0.3 * float(i2s_dino) * 100,
                0.2 * float(i2s_ssim) * 100,
                0.2 * (1 - float(i2s_lpips)) * 100,
            ))
            primary['Illustration'] = illu_score
        except Exception:
            pass

        return primary
