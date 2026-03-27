import os
import os.path as osp

from ..smp import decode_base64_to_image_file, dump, load, read_ok
from ..utils import track_progress_rich
from .image_base import ImageBaseDataset
from .utils import DEBUG_MESSAGE, build_judge


class MMOralBase(ImageBaseDataset):
    """Shared image-dumping logic for MMOral-OPG benchmarks."""

    def dump_image(self, line):
        os.makedirs(self.img_root, exist_ok=True)

        tgt_path_z = []
        if isinstance(line['image'], list):
            for i in range(len(line['image'])):
                tgt_path = osp.join(self.img_root, f"{line['index']}--{i + 1}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'][i], tgt_path)
                tgt_path_z.append(tgt_path)
        else:
            tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path_z.append(tgt_path)
        return tgt_path_z


class MMOral_OPG_OPEN(MMOralBase):
    """Open-ended MMOral-OPG benchmark (VQA)."""

    TYPE = 'VQA'

    DATASET_URL = {
        'MMOral_OPG_OPEN': 'https://huggingface.co/datasets/OralGPT/MMOral-OPG-Bench/resolve/main/MMOral-OPG-Bench-Open-Ended.tsv'  # noqa: E501
    }

    DATASET_MD5 = {
        'MMOral_OPG_OPEN': 'd328b1b527ef7467b328d8b35d5f8155'
    }

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line)
        question = line['question']
        prompt = (
            f'Question: {question}\n'
            'Please provide a detailed and accurate answer to the question.'
        )

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    @classmethod
    def evaluate(cls, eval_file, **judge_kwargs):
        """Evaluation with LLM-as-a-judge for open-ended answers."""
        from .utils.mmoral_opg import MMOral_opg_acc, MMOral_opg_auxeval

        suffix = eval_file.split('.')[-1]
        # Some call sites may not explicitly set `judge_kwargs['model']`,
        # so we fall back to a default name for the judge model.
        judge_model_name = judge_kwargs.pop('model', 'mmoral-opg-judge')
        storage = eval_file.replace(f'.{suffix}', f'_{judge_model_name}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{judge_model_name}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            model = build_judge(model=judge_model_name, max_tokens=16384, **judge_kwargs)
            assert model.working(), (
                'MMOral-Open-ended evaluation requires a working OPENAI API\n'
                + DEBUG_MESSAGE
            )

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = load(tmp_file) if osp.exists(tmp_file) else {}
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                new_results = track_progress_rich(
                    MMOral_opg_auxeval,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
                ans = load(tmp_file)
                for k, v in zip(indices, new_results):
                    assert k in ans
                    assert ans[k]['log'] == v['log'] and ans[k]['score'] == v['score']

            data['score'] = [ans[idx]['score'] for idx in data['index']]
            data['log'] = [ans[idx]['log'] for idx in data['index']]
            dump(data, storage)

        score, score_fine = MMOral_opg_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')
        score_fine_pth = storage.replace('.xlsx', '_score_fine.csv')
        dump(score, score_pth)
        dump(score_fine, score_fine_pth)
        return score
