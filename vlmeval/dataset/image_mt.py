from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge
from ..smp import *
from ..smp.file import get_intermediate_file_path
from ..utils import track_progress_rich


class ImageMTDataset(ImageBaseDataset):

    TYPE = 'MT'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        questions = toliststr(line['question'])
        if 'answer' in line:
            answers = toliststr(line['answer'])
        else:
            answers = [''] * len(questions)
        assert len(questions) == len(answers)

        dlgs, pics_number = [], 0
        for i in range(len(questions)):
            q, a = questions[i], answers[i]
            if '<ImageHere>' in q:
                content = []
                tag_number = q.count('<ImageHere>')
                images = tgt_path[pics_number: pics_number + tag_number]
                pics_number += tag_number
                q_split = q.split('<ImageHere>')
                for i in range(tag_number):
                    qsp, im = q_split[i], images[i]
                    if qsp != '':
                        content.append(dict(type='text', value=qsp))
                    content.append(dict(type='image', value=im))
                if q_split[-1] != '':
                    content.append(dict(type='text', value=q_split[-1]))
            else:
                content = [dict(type='text', value=q)]
            dlgs.append(dict(role='user', content=content))
            assert '<ImageHere>' not in a, 'We currently do not support images in the answer. '
            content = [dict(type='text', value=a)]
            dlgs.append(dict(role='assistant', content=content))
        return dlgs


class MMDUDataset(ImageMTDataset):

    DATASET_URL = {'MMDU': 'https://opencompass.openxlab.space/utils/VLMEval/MMDU.tsv'}
    DATASET_MD5 = {'MMDU': '848b635a88a078f49aebcc6e39792061'}
    DIMS = [
        'Creativity', 'Richness', 'Visual Perception', 'Logical Coherence',
        'Answer Accuracy', 'Image Relationship Understanding', 'Overall Score'
    ]

    def calculat_metric(self, ans):
        all = defaultdict(lambda: 0)
        tot = defaultdict(lambda: 0)
        valid = defaultdict(lambda: 0)
        for k in ans:
            res = ans[k]['res']
            assert isinstance(res, pd.DataFrame)
            lt = len(res)
            for i in range(lt):
                line = res.iloc[i]
                for k in self.DIMS:
                    tot[k] += 1
                    if k in line and line[k] is not None:
                        try:
                            score = int(line[k])
                            score = np.clip(score, 0, 10)
                            all[k] += score
                            valid[k] += 1
                        except Exception as e:
                            print(f'Failed to parse the score: {str(e)}')
        sp1 = {'set': 'all'}
        sp1.update({k: all[k] / tot[k] * 10 for k in self.DIMS})
        sp2 = {'set': 'valid'}
        sp2.update({k: all[k] / valid[k] * 10 for k in self.DIMS})

        return pd.DataFrame([sp1, sp2])

    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs['model']

        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'csv')
        nproc = judge_kwargs.pop('nproc', 4)

        data = load(eval_file)
        model = judge_kwargs.pop('model', 'gpt-4o')
        judge_model = build_judge(model=model, **judge_kwargs)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        tups = [(judge_model, line) for line in lines]
        indices = [line['index'] for line in lines]

        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)

        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]

        from .utils.mmdu import mmdu_score

        if len(indices):
            new_results = track_progress_rich(
                mmdu_score,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
                save=tmp_file,)
            ans = load(tmp_file)
            for k, v in zip(indices, new_results):
                assert k in ans

        metric = self.calculat_metric(ans)
        dump(metric, score_file)
        return metric

class MMMTBenchDataset(ImageMTDataset):
    """
    MM-MT-Bench 数据集
    参考: https://github.com/mistralai/mistral-evals/blob/main/eval/tasks/mm_mt_bench.py
    """
    DATASET_URL = {'MM_MT_Bench': 'https://opencompass.openxlab.space/utils/VLMEval/MM-MT-Bench.tsv'}
    DATASET_MD5 = {'MM_MT_Bench': '5c9f2a4b7c97f44a7a3c603ce7dee0ce'}

    def evaluate(self, eval_file, **judge_kwargs):
        """
        使用 GPT-4o 作为 judge 评估 MM-MT-Bench 结果
        
        评估流程:
        1. 对每个样本，构建包含用户问题、参考答案和模型答案的 prompt
        2. 使用 judge 模型（默认 gpt-4o）对模型回答进行 1-10 的评分
        3. 聚合各个 category 和 turn 的分数
        4. 计算 micro_average 和 macro_average
        """
        nproc = judge_kwargs.pop('nproc', 4)
        model = judge_kwargs.pop('model', 'gpt-4o')
        judge_model = build_judge(model=model, **judge_kwargs)

        tmp_file = get_intermediate_file_path(eval_file, f'_{model}', 'pkl')
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'json')
        data = load(eval_file)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        # tups = [(judge_model, line) for line in lines]
        # indices = [line['index'] for line in lines]
        tups, indices = [], []
        index = 0
        for line in lines:
            predictions = eval(line['prediction'])
            for i, prediction in enumerate(predictions):
                line['pred'] = prediction
                tups.append((judge_model, line, i))
                indices.append(index)
                index += 1
        
        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)

        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]

        from .utils.mm_mt_bench import mm_mt_bench_score, calculate_mm_mt_bench_metrics

        if len(indices):
            new_results = track_progress_rich(
                mm_mt_bench_score,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
                save=tmp_file,)
            ans = load(tmp_file)
            for k, v in zip(indices, new_results):
                assert k in ans

        # 计算聚合指标
        metrics = calculate_mm_mt_bench_metrics(ans)
        
        # 构建清晰的 JSON 格式结果
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score', 'json')
        
        # 分离 category 和 turn 的分数
        category_scores = {}
        turn_scores = {}
        for key, value in metrics.items():
            if key.endswith('_average') and key not in ['micro_average_score', 'macro_average_score']:
                name = key.replace('_average', '')
                if name.startswith('turn_'):
                    turn_scores[name] = round(value, 4)
                else:
                    category_scores[name] = round(value, 4)
        
        result = {
            'overall': {
                'micro_average': round(metrics['micro_average_score'], 4),
                'macro_average': round(metrics['macro_average_score'], 4),
                'valid_count': metrics['valid_count'],
            },
            'category_scores': category_scores,
            'turn_scores': turn_scores,
        }
        
        dump(result, score_file)
        return result