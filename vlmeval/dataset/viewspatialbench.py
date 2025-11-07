import ast
import os.path as osp

from ..smp import *
from ..smp.file import LMUDataRoot
from ..smp.misc import toliststr, get_cache_path, modelscope_flag_set
from .image_mcq import ImageMCQDataset

from huggingface_hub import snapshot_download


class ViewSpatialBench(ImageMCQDataset):
    TYPE = 'MCQ'

    LMUData_root = LMUDataRoot()
    DATASET_URL = {}

    DATASET_URL["ViewSpatialBench"] = "https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/ViewSpatialBench.tsv"  # noqa: E501
    DATASET_MD5 = {key: None for key in DATASET_URL}

    def _task_category(self):
        return [
            'Camera perspective - Relative Direction',
            'Camera perspective - Object View Orientation',
            'Person perspective - Object View Orientation',
            'Person perspective - Relative Direction',
            'Person perspective - Scene Simulation Relative Direction'
        ]

    def prepare_tsv(self, url, file_md5=None, repo_id='lidingm/ViewSpatial-Bench'):
        data = super().prepare_tsv(url, file_md5)

        cache_path = get_cache_path(repo_id)
        SENTINEL_NAME = ".viewspatialbench_extracted"
        if (cache_path and os.path.isdir(cache_path)
                and os.path.isfile(os.path.join(cache_path, SENTINEL_NAME))):
            dataset_path = cache_path
        else:
            def _write_sentinel(sentinel_path, text="ok"):
                tmp = sentinel_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write(text)
                os.replace(tmp, sentinel_path)

            def unzip_hf_zip(pth):
                import zipfile

                base_dir = pth
                zip_files = [
                    os.path.join(base_dir, f) for f in os.listdir(base_dir)
                    if f.endswith('.zip')
                ]
                zip_files.sort()

                for zip_file in tqdm(zip_files, desc='Unpacking Origin Data...'):
                    with zipfile.ZipFile(zip_file, 'r') as zf:
                        for info in zf.infolist():
                            if info.is_dir():
                                continue

                            rel = os.path.normpath(info.filename).lstrip('/\\')
                            dst = os.path.join(pth, rel)

                            absp = os.path.abspath(pth)
                            absd = os.path.abspath(dst)
                            if not absd.startswith(absp + os.sep):
                                raise RuntimeError(f'Unsafe path in zip: {info.filename}')

                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            with zf.open(info, 'r') as src, open(dst, 'wb') as out:
                                out.write(src.read())

                sentinel_path = os.path.join(pth, SENTINEL_NAME)
                _write_sentinel(sentinel_path, text="done")
                print('ViewSpatial data extracted to current directory with original layout.')

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            unzip_hf_zip(dataset_path)

        # === Transfer rel path to abs path ===
        if 'image_path' in data.columns:
            def fix_one(x: str):
                if not isinstance(x, str):
                    return x
                s = x.strip()
                s = os.path.expanduser(os.path.expandvars(s))

                if not dataset_path:
                    return os.path.normpath(s)
                return os.path.normpath(os.path.join(dataset_path, s.lstrip(r'\/')))

            def to_abs(p):
                if isinstance(p, list):
                    return [fix_one(xx) for xx in p]
                if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                    try:
                        lst = ast.literal_eval(p)
                        if isinstance(lst, list):
                            return [fix_one(xx) for xx in lst]
                    except Exception:
                        pass
                return fix_one(p)

            data['image_path'] = data['image_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question']
        choices = line['candidates']

        # prompt format from viewspatial bench paper
        question_text = f"Question: {question}\n"
        choices_text = f"Choices: {choices}\n"
        post_prompt = "Reply only to the corresponding option.\nAnswer:"

        prompt = question_text + choices_text + post_prompt

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import compute_mcq_score, eval_mcq_core

        return eval_mcq_core(
            load_fn=load,
            eval_file=eval_file,
            score_fn=compute_mcq_score,
            group_col='question_type',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'ViewSpatialBench')
        )
