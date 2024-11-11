import json
import os

import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE


def generate_prompt(d, response):
    instruction = d['question']
    weight = d['component_weight'] * 1
    weight = [int(i) for i in weight[1:-1].split(', ')]

    for i in range(len(weight)):
        weight[i] = str(weight[i])
    if d['num_of_component'] == 1:
        components = (
            "The first component is: '"
            f"{d['componnets'][0]}"
            "'"
        )
        score = (
            "The first component is worth: "
            f"{weight[0]} scores."
        )
    elif d['num_of_component'] == 2:
        components = (
            "The first component is: '"
            f"{d['componnets'][0]}"
            "', and the second component is "
            f"{d['componnets'][1]}"
            "'"
        )
        score = (
            "The first and second component is each worth "
            f"{weight[0]} and {weight[1]} scores."
        )
    elif d['num_of_component'] == 3:
        components = (
            "The first component is: '"
            f"{d['componnets'][0]}"
            "', and the second component is "
            f"{d['componnets'][1]}"
            "', and the third component is "
            f"{d['componnets'][2]}"
            "'"
        )
        score = (
            "The first, second, and third component is each worth "
            f"{weight[0]}, {weight[1]}, and {weight[2]} scores."
        )
    elif d['num_of_component'] == 4:
        components = (
            "The first component is: '"
            f"{d['componnets'][0]}"
            "', and the second component is "
            f"{d['componnets'][1]}"
            "', and the third component is "
            f"{d['componnets'][2]}"
            "', and the fourth component is "
            f"{d['componnets'][3]}"
            "'"
        )
        score = (
            "The first, second, third, and fourth component is each worth "
            f"{weight[0]}, {weight[1]}, {weight[2]}, and {weight[3]} scores."
        )
    elif d['num_of_component'] == 5:
        components = (
            "The first component is: '"
            f"{d['componnets'][0]}"
            ", and the second component is "
            f"{d['componnets'][1]}"
            ", and the third component is "
            f"{d['componnets'][2]}"
            ", and the fourth component is "
            f"{d['componnets'][3]}"
            ", and the fifth component is "
            f"{d['componnets'][4]}"
            "'"
        )
        score = (
            "The first, second, third, fourth, and fifth component is each worth "
            f"{weight[0]}, {weight[1]}, {weight[2]}, {weight[3]}, and {weight[4]} scores."
        )
    return (
        "Here is an instruction for a multimodal LLM: '"
        f"{instruction}"
        "'. You need to grade if the response from the model follows each component of the instruction. "
        f"{components}"
        "The response is: '"
        f"{response}"
        "'. You need to score the response and be strict. The total score ranges from 0 to 10, "
        "depending on if the response follows the instruction. "
        f"{score}"
        "List scores of each component, and the total score in one sentence in this format: "
        "score of component 1: x/2, score of component 2: y/8, total score: z/10. Then explain your reasons."
    )


def process_rawscore(component_type, raw_score):
    first_sentence = raw_score.split('.')[0].split(',')
    score_dict = {}
    for i in range(len(first_sentence) - 1):
        score_ = first_sentence[i].split(':')[1][1:].split('/')
        score = int(score_[0]) / int(score_[1])
        score_dict[component_type[i]] = score
    total_score_ = first_sentence[i + 1].split(':')[1][1:].split('/')
    total_score = int(total_score_[0]) / int(total_score_[1])
    score_dict['total_score'] = total_score
    return score_dict


def get_score_dict(data, score_raw):
    cat_score_dict = {}
    for i in range(len(data)):
        try:
            cmp = data['component_type'][i][2:-2]
            cmp_list = cmp.split('\', \'')
            score_dict = process_rawscore(cmp_list, score_raw[i])
            for key, val in score_dict.items():
                if key not in cat_score_dict.keys():
                    cat_score_dict[key] = [val]
                else:
                    cat_score_dict[key].append(val)
        except:
            pass
    cat_score_dict_average = {}
    for key, val in cat_score_dict.items():
        cat_score_dict_average[key] = sum(val) / len(val)
    return cat_score_dict_average


class MIABench(ImageBaseDataset):
    TYPE = 'Caption'

    DATASET_URL = {
        'MIA-Bench': 'https://opencompass.openxlab.space/utils/VLMEval/MIA-Bench.tsv',
    }
    DATASET_MD5 = {
        'MIA-Bench': '0b9de595f4dd40af18a69b94d89aba82',
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.Mia_Bench import generate_prompt
        from openai import OpenAI
        import requests
        from io import BytesIO
        openai_base = os.environ.get("OPENAI_API_BASE")
        if openai_base is not None:
            openai_base = openai_base[:openai_base.index('v1') + 2]
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=openai_base)
        else:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')  # noqa: F841
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')  # noqa: F841
        nproc = judge_kwargs.pop('nproc', 4)  # noqa: F841

        if not osp.exists(storage):
            data = load(eval_file)
            score_raw = ['' for _ in range(len(data))]

            for i in tqdm(range(len(data))):
                line = data.loc[i]
                response = line['prediction']
                image = line['image_url']

                question = generate_prompt(line, response)
                generated = False

                attempt = 5
                while attempt > 0 and not generated:
                    try:
                        rev_response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": question},
                                        {"type": "image_url",
                                         "image_url": {"url": image}
                                         },
                                    ],
                                }
                            ],
                            max_tokens=2000
                        )
                        score_raw[i] = rev_response.choices[0].message.content.strip()
                        generated = True
                    except:
                        attempt -= 1
            data['score_raw'] = score_raw
            dump(data, storage)

        goresult = load(storage)
        results = get_score_dict(goresult, goresult['score_raw'])
        result_pth = storage.replace('.xlsx', '_score.csv')
        results_pd = pd.DataFrame.from_dict(list(results.items()))
        dump(results_pd, result_pth)

        return results
