import json
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.smp import *
from .utils.judge_util import build_judge
import pydantic
from pydantic import BaseModel
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

COMPARE_ANSWER_PROMPT = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。
首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
## 以下是【正确】的答复示例：
```
问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
- 完整地包含了标准答案中的重要信息。
- 不包含任何与标准答案矛盾的信息。
- 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
- 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

## 以下是【错误】的答复示例：
```
问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
- 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

## 以下是【未尝试】的答复示例：
```
问题：图中人物（贝拉克·奥巴马）的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
模型预测5：我无法识别图中的人物。
模型预测6：N/A。
```
这些答复均为【未尝试】，因为：
- 没有包含标准答案中的重要信息。
- 回复中没有与标准答案矛盾的陈述。
只返回字母”A”、”B”或”C”，无须添加其他文本。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
- 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
- 预测答案“3520”和“3600”均为【错误】。
- 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
- 例如，考虑问题“菱镁矿的主要化学成分是什么？”,标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作【正确】。
- 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为【正确】。
- 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均【正确】。
- 预测答案和标准答案对应的是同一事物，但是称呼不同，如“天坛”和“祈年殿”，那么算作【正确】

## 下面是一个新的问题示例。对每一个预测答案，请只回复"正确"、"错误"、"未尝试"之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {answer}
预测答案: {prediction}
```

请严格按照以下格式回复，以JSON格式返回一个字典,而且字典第一层key不要替换为具体答案。不要返回其他任何内容。
[回复格式]:
```json
{{
    "conclusion": "整体结论，只返回'正确','错误','未尝试'中的一个，无须添加其他文本",
    "reasoning": "判断依据，详细解释判断结果，不能超过200字"
}}
```
""".strip()


class JUDGE_FORMAT(BaseModel):
    conclusion: str
    reasoning: str


def evaluate_single_sample(judge, line):
    question = line['question']
    answer = line['answer']
    prediction = line['prediction']
    judge_prompt = COMPARE_ANSWER_PROMPT.format(
        question=question,
        answer=answer,
        prediction=prediction
    )

    def validate_resp(resp):
        if not isinstance(resp, dict):
            return False
        if 'conclusion' not in resp or 'reasoning' not in resp:
            return False
        cands = ['正确', '错误', '未尝试']
        candin = [x in resp['conclusion'] for x in cands]
        if np.sum(candin) != 1:
            return False
        ret = cp.deepcopy(resp)
        for item in cands:
            if item in ret['conclusion']:
                ret['conclusion'] = item
                break
        return ret

    judge_res = judge.generate(judge_prompt, response_format=JUDGE_FORMAT)
    retry = 3
    while retry > 0 and not validate_resp(judge_res):
        retry -= 1
        judge_res = judge.generate(judge_prompt, response_format=JUDGE_FORMAT)
    if not validate_resp(judge_res):
        ret = {}
        ret['conclusion'] = '未尝试'
        ret['reasoning'] = f'Invalid format: {judge_res}'
        judge_res = ret
    else:
        judge_res = validate_resp(judge_res)
    return judge_res


class SimpleVQA(ImageBaseDataset):
    TYPE = "VQA"
    DATASET_URL = {
        "SimpleVQA": "https://opencompass.openxlab.space/utils/VLMEval/SimpleVQA.tsv",
    }
    DATASET_MD5 = {
        "SimpleVQA": "e3ce3c11df59a2a15d37489a8b245a87",
    }
    DEFAULT_JUDGE = "gpt-4o"
    JUDGE_FORMAT = "{model_name}_{dataset_name}_judge.tsv"

    def build_prompt(self, line: Union[int, pd.Series], qa_type: str = 'Direct') -> List[Dict[str, str]]:
        """
        Build a prompt for the model from a data line.

        Args:
            line: Either an index into the dataset or a pandas Series
            qa_type: Choose from ['Direct', 'CoT', 'PoT']

        Returns:
            List of message dictionaries containing the image and question
        """
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = misc.toliststr(line["image"])
        else:
            tgt_path = self.dump_image(line)

        # load data line elements
        question = line['question']
        assert len(tgt_path) == 1

        # form messages
        msgs = []
        msgs = [dict(type='image', value=tgt_path[0])]
        msgs.append(dict(type='text', value=question))

        return msgs

    def evaluate(self, eval_file: str, **judge_kwargs: Any) -> pd.DataFrame:
        score_file = get_intermediate_file_path(eval_file, "_acc", "csv")
        tmp_file = get_intermediate_file_path(eval_file, "_tmp", "pkl")
        judge_file = get_intermediate_file_path(eval_file, "_judge", "tsv")

        data = load(eval_file)
        data['prediction'] = data['prediction'].astype(str)
        lines = [row for _, row in data.iterrows()]
        inds = [row['index'] for row in lines]
        res = {}
        if osp.exists(tmp_file):
            try:
                res.update(load(tmp_file))
            except:
                os.remove(tmp_file)
        tups = [(idx, line) for idx, line in zip(inds, lines) if idx not in res]

        if len(tups):
            nproc = judge_kwargs.pop('nproc', 16)
            judge = build_judge(**judge_kwargs)

            indices = [x[0] for x in tups]
            payload = [dict(judge=judge, line=x[1]) for x in tups]
            _ = track_progress_rich(
                evaluate_single_sample, payload, nproc=nproc, save=tmp_file, keys=indices, desc='SimpleVQA Eval')
            res = load(tmp_file)

        keys = ['conclusion', 'reasoning']
        for k in keys:
            data[k] = [res[idx][k] for idx in data['index']]
        dump(data, judge_file)

        n_correct = sum([x == '正确' for x in data['conclusion']])
        n_wrong = sum([x == '错误' for x in data['conclusion']])
        n_missing = sum([x == '未尝试' for x in data['conclusion']])
        acc_attempted = n_correct / (n_correct + n_wrong)
        acc = n_correct / len(data)
        f1 = 2 * acc_attempted * acc / (acc_attempted + acc) if acc_attempted + acc > 0 else 0

        score = {
            'overall': n_correct / len(data),
            'overall_wo_missing': n_correct / (len(data) - n_missing),
            'missing_rate': n_missing / len(data),
            'f1': f1
        }
        df = d2df(score)
        dump(df, score_file)
        return df
