from .image_base import ImageBaseDataset
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE


class VTCBench(ImageBaseDataset):
    TYPE = 'VQA'
    _DATASET_PATH = "https://huggingface.co/datasets/MLLM-CL/VTCBench"
    # Dataset URL mapping - points to different splits of HuggingFace dataset
    DATASET_URL = {
        "VTCBench": "",
    }
    DEFAULT_JUDGE = 'gpt-4o-mini'

    @classmethod
    def supported_datasets(cls):
        return ['VTCBench']

    def load_data(self, dataset: str):
        """Load dataset from HuggingFace"""

        from datasets import load_dataset

        COLUMNS_ORGINIAL = ["problem", "answers", "images"]
        all_dataframes = []
        current_index = 0

        for task in ["Retrieval", "Reasoning", "Memory"]:

            def _gen_fields(example: dict, idx: int) -> dict:
                # example schema:
                # problem: str
                # answers: list[str]
                # images: list[dict[str, bytes]] # bytes obj <=> jpeg image
                def encode_image_bytes_to_base64(image_bytes) -> str:
                    """Encode image bytes to base64 string."""
                    return base64.b64encode(image_bytes).decode()

                b64_imgs: list[str] = [
                    encode_image_bytes_to_base64(img["bytes"]) for img in example["images"]
                ]
                return {
                    "index": int(idx) + current_index,
                    "question": example["problem"],
                    "answer": json.dumps(example["answers"], ensure_ascii=False),
                    "image": json.dumps(b64_imgs),
                    "category": task,
                }

            hf_dataset = load_dataset(
                self._DATASET_PATH, split=task, columns=COLUMNS_ORGINIAL,
            )
            # apply transformation to VLMEval format
            hf_dataset = hf_dataset.map(
                _gen_fields,
                remove_columns=COLUMNS_ORGINIAL,
                with_indices=True,
                num_proc=16,
            )
            data = hf_dataset.to_pandas()
            all_dataframes.append(data)
            current_index += len(data)

        # Concatenate all dataframes
        merged_data = pd.concat(all_dataframes, ignore_index=False)

        # now data has schema:
        # index <class 'int'> 0
        # question <class 'str'>
        # What are all the special magic numbers for 019cc30e-2da8-4162-b145-df514e17
        # and demonic-heaven mentioned in the provided text?
        # answer <class 'str'> ["9199619", "1202641"]
        # image <class 'list'>
        # [/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQN..., ...]
        # category <class 'str'> Retrieval

        return merged_data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        base64_list = line['image']

        question = line['question']
        category = line['category']

        if category == 'Reasoning':
            prompt = """\
Answer a question based on the above book snippet. \
Your answer should be short and based on either explicitly stated facts or strong, logical inferences. \
Return only the final answer with no additional explanation or reasoning. Question: """ + question
        elif category == 'Retrieval':
            prompt = """\
Answer a question based on the above book snippet. \
Some special magic numbers are hidden within the following text. Make sure to memorizeit. \
I will quiz you about the numbers afterwards. Question: """ + question
        elif category == 'Memory':
            prompt = """\
Based on the above context, write an answer in the form of a short phrase for the following question. \
Answer with exact words from the context whenever possible. Question: """ + question
        else:
            raise ValueError(f"Unknown category: {category}")

        msgs = []
        if isinstance(base64_list, list):
            msgs.extend([dict(type='image', value='data:image/jpeg;base64,' + p) for p in base64_list])
        else:
            msgs = [dict(type='image', value='data:image/jpeg;base64,' + base64_list)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        try:
            judge_model = build_judge(max_tokens=1024, **judge_kwargs)
            assert judge_model.working(), ('VTCBench evaluation requires a working OPENAI API\n')
            return self.get_scores_gpt(eval_file, **judge_kwargs)
        except:
            print('No GPT model specified, using heuristic evaluation.')
            return self.get_scores(eval_file, **judge_kwargs)

    @classmethod
    def get_scores_gpt(self, eval_file, **judge_kwargs):
        from .utils.vtcbench import gpt_eval_vtcbemch
        model = build_judge(max_tokens=128, **judge_kwargs)
        score_file = get_intermediate_file_path(eval_file, f'_{model}_score')
        tmp_file_score = get_intermediate_file_path(eval_file, f'_{model}_score', 'pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]

            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            scores = {}
            if osp.exists(tmp_file_score):
                scores = load(tmp_file_score)
            tups = [x for x, i in zip(tups, indices) if i not in scores]
            indices = [i for i in indices if i not in scores]

            if len(indices):
                new_result = track_progress_rich(
                    gpt_eval_vtcbemch,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file_score,
                )
                scores = load(tmp_file_score)
                for k, v in zip(indices, new_result):
                    assert k in scores
                    assert scores[k]['score'] == v['score'] and scores[k]['category'] == v['category'] and scores[k]['calc_metric'] == v['calc_metric']  # noqa: E501
            data['score'] = [scores[idx]['score'] for idx in data['index']]
            data['category'] = [scores[idx]['category'] for idx in data['index']]
            data['calc_metric'] = [scores[idx]['calc_metric'] for idx in data['index']]

            dump(data, score_file)

        # 加载评分后的数据并按calc_metric聚合
        data = load(score_file)

        # 按照calc_metric聚合结果
        category_scores = {}
        category_counts = {}

        # 遍历data中的每个结果
        for _, row in data.iterrows():
            category = row['calc_metric']
            score = row['score']

            # 累加分数和计数
            if category not in category_scores:
                category_scores[category] = 0
                category_counts[category] = 0

            category_scores[category] += score
            category_counts[category] += 1

        # 计算每个category的平均分数
        ret = dict()
        for category in category_scores:
            ret[category] = category_scores[category] / category_counts[category]

        # 添加Overall平均分
        if category_scores:  # 确保有数据才计算
            total_score = sum(category_scores.values())
            total_count = sum(category_counts.values())
            ret['Overall'] = total_score / total_count
        else:
            ret['Overall'] = 0.0

        result_file = get_intermediate_file_path(eval_file, '_result', 'json')
        dump(ret, result_file)
        return ret

    @classmethod
    def get_scores(self, eval_file, **judge_kwargs):
        from .utils.vtcbench import process_vtc_line

        result_file = get_intermediate_file_path(eval_file, '_tmp')

        if not osp.exists(result_file):
            data = load(eval_file)

            assert 'answer' in data and 'prediction' in data
            data['prediction'] = [str(x) for x in data['prediction']]
            data['answer'] = [str(x) for x in data['answer']]
            data['category'] = [str(x) for x in data['category']]

            lt = len(data)
            pool = mp.Pool(1)
            lines = [data.iloc[i] for i in range(lt)]

            res = pool.map(process_vtc_line, lines)

            # 按照category聚合结果
            category_scores = {}
            category_counts = {}

            # 遍历res中的每个结果
            for item in res:
                category = item['category']
                score = item['score']

                # 累加分数和计数
                if category not in category_scores:
                    category_scores[category] = 0
                    category_counts[category] = 0

                category_scores[category] += score
                category_counts[category] += 1

            ret = dict()

            # 计算每个category的平均分数
            for category in category_scores:
                ret[category] = category_scores[category] / category_counts[category]

            # 添加Overall平均分
            if category_scores:  # 确保有数据才计算
                total_score = sum(category_scores.values())
                total_count = sum(category_counts.values())
                ret['Overall'] = total_score / total_count
            else:
                ret['Overall'] = 0.0

            dump(ret, result_file)

        retz = load(result_file)
        return retz
