from vlmeval import *
from .image_base import ImageBaseDataset

EVAL_TEMPLATE = """
You are a strict evaluator assessing answer correctness. You must score the model's prediction on a scale from 0 to 9.
0 represents an entirely incorrect answer and 9 indicates a highly correct answer.

# Input
Question
{question}
Ground Truth Answer
{answer}
Model Prediction
{prediction}

# Evaluation Rules
- The model prediction may contain the reasoning process, you should spot the final answer
from it.
- For multiple-choice questions: Assign a higher score if the predicted answer matches the
ground truth, either by option letters or content. Include partial credit for answers that are
close in content.
- For exact match and open-ended questions:
    - Assign a high score if the prediction matches the answer semantically, considering variations in format.
    - Deduct points for partially correct answers or those with incorrect additional information.
- Ignore minor differences in formatting, capitalization, or spacing since the model may explain in a different way.
- Treat numerical answers as correct if they match within reasonable precision
- For questions requiring units, both value and unit must be correct

# Scoring Guide
Provide a single integer from 0 to 9 to reflect your judgment of the answer's correctness.
# Strict Output format example
4
"""


def report_score(df):
    # 按学科分别统计
    res = defaultdict(list)

    if 'split' in df:
        splits = list(set(df['split']))
        res['split'] = splits
    else:
        df['split'] = ['none'] * len(df)
        res['split'] = ['none']

    for group in [None, 'Astrnomy', 'Chemistry', 'Earth', 'Life', 'Materials']:
        if group is None:
            res['Overall'] = [np.mean(df[df['split'] == sp]['score']) for sp in res['split']]
        elif group not in df:
            continue
        else:
            diciplines = list(set(df[group]))
            diciplines.sort()
            for dc in diciplines:
                sub_df = df[df[group] == dc]
                res[dc] = [np.mean(sub_df[sub_df['split'] == sp]['score']) for sp in res['split']]
    return pd.DataFrame(res)


def make_prompt(line):
    question = line['question']
    answer = line['answer']
    tmpl = EVAL_TEMPLATE
    prompt = tmpl.format(
        question=question,
        answer=answer,
        prediction=line['prediction']
    )
    return prompt


def SFE_auxeval(model, data):
    if isinstance(data, pd.DataFrame) and len(data) > 1:
        lt = len(data)
        for i in range(lt):
            total_score = 0
            item = data.iloc[i]
            prompt = make_prompt(item)
            retry = 3
            for j in range(retry):
                output = model.generate(prompt, temperature=0.5 * j)
                if output.isdigit() and 0 <= int(output) <= 9:
                    total_score += int(output)
                    break
        avg_score = total_score / lt
        return dict(score=avg_score, log='Success to Judge')
    else:
        item = data.iloc[0] if isinstance(data, pd.DataFrame) else data
        prompt = make_prompt(item)
        retry = 3
        for i in range(retry):
            output = model.generate(prompt, temperature=0.5 * i)
            if output.isdigit() and 0 <= int(output) <= 9:
                return dict(score=int(output), log='Success to Judge')
        return dict(score=0, log='Fail to Judge')


class SFE(ImageBaseDataset):

    DATASET_URL = {
        'SFE': '',
    }

    def request_image(self, image_name, repo_id="PrismaX/SFE"):
        """
        从HuggingFace仓库下载图片
        """
        import requests
        from io import BytesIO
        from PIL import Image

        image_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/images/{image_name}"
        response = requests.get(image_url)
        response.raise_for_status()
        print("load image successfully!")
        image_pil = Image.open(BytesIO(response.content))
        image_base = encode_image_to_base64(image_pil)
        return image_base

    def load_data(self, dataset="SFE", repo_id="PrismaX/SFE"):
        """
        将HuggingFace parquet后缀dataset生成规范的DataFrame
        需要字段:
            index, question, answer, image
        现有字段:
            query, response, image(Path)
        """
        import re
        import pandas as pd
        import datasets
        import json
        from ..tools import encode_image_to_base64

        MCQ_PROMPT = (
            "You are an expert in {discipline} and need to solve the following question. "
            + "The question is a multiple-choice question. "
            + "Answer with the option letter from the given choices."
        )
        EXACT_MATCH_PROMPT = (
            "You are an expert in {discipline} and need to solve the following question. "
            + "The question is an exact match question. Answer the question using a single word or phrase."
        )
        OPEN_QUESTION_PROMPT = (
            "You are an expert in {discipline} and need to solve the following question. "
            + "The question is an open-ended question. Answer the question using a phrase."
        )

        # 读取huggingface数据
        hf_ds = datasets.load_dataset(repo_id, data_dir='20250611-en', split='test')
        records = []
        for idx, sample in enumerate(hf_ds):
            question_type = sample['question_type']
            field = sample['field']
            if question_type == 'exact_match':
                prompt = EXACT_MATCH_PROMPT.format(discipline=field)
                question = prompt + " " + sample['question']
                image = []
                answer = sample['answer']
                for image_name in sample['images']:
                    image_base = self.request_image(image_name, repo_id)
                    image.append(image_base)
            if question_type == 'mcq':
                prompt = MCQ_PROMPT.format(discipline=field)
                question = prompt + " " + sample['question']
                answer = ""
                image = []
                for option in sample['options']:
                    question += " " + option
                answer = sample['answer']
                for image_name in sample['images']:
                    image_base = self.request_image(image_name, repo_id)
                    image.append(image_base)
            if question_type == 'open_ended':
                prompt = OPEN_QUESTION_PROMPT.format(discipline=field)
                question = prompt + " " + sample["question"]
                answer = sample['answer']
                for image_name in sample['images']:
                    image_base = self.request_image(image_name, repo_id)
                    image.append(image_base)
            rec = {
                "index": idx,
                "question": question,
                "answer": answer,
                "image": image
            }
            records.append(rec)
        df = pd.DataFrame(records)
        df.reset_index(drop=True, inplace=True)
        return df

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)
        prompt = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        _ = self.dataset_name
        assert 'answer' in data and 'prediction' in data
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]
        storage = eval_file.replace('.xlsx', '_judge.xlsx')
        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        nproc = judge_kwargs.pop('nproc', 4)
        if not osp.exists(storage):
            ans_map = {} if not osp.exists(tmp_file) else load(tmp_file)

            model = judge_kwargs.get('model', 'gpt-4o-mini')
            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(model=model, **judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                model = None
                warnings.warn('OPENAI_API_KEY is not working properly, will use exact matching for evaluation')

            if model is not None:
                if 'g_index' not in data:
                    lines = [data.iloc[i] for i in range(len(data))]
                    indices = [x['index'] for x in lines if x['index'] not in ans_map]
                    lines = [x for x in lines if x['index'] not in ans_map]
                    tups = [(model, line) for line in lines]
                else:
                    main_data = data[[x == y for x, y in zip(data['index'], data['g_index'])]]
                    lines = [data[data['g_index'] == x] for x in main_data['index']]
                    indices = [x.iloc[0]['g_index'] for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    lines = [x for x in lines if x.iloc[0]['g_index'] not in ans_map]
                    tups = [(model, x) for x in lines]
                    data = main_data

                if len(lines):
                    res = track_progress_rich(
                        SFE_auxeval, tups, nproc=nproc, chunksize=nproc, keys=indices, save=tmp_file)
                    for k, v in zip(indices, res):
                        ans_map[k] = v

            judge_results = [ans_map[x] for x in data['index']]
            data['score'] = [x['score'] for x in judge_results]
            dump(data, storage)
        data = load(storage)
        score = report_score(data)

        score_file = eval_file.replace('.xlsx', '_score.csv')
        dump(score, score_file)
        return score
