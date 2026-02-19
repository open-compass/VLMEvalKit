import pandas as pd

from .image_base import ImageBaseDataset

from .utils.multiple_choice import report_acc
from ..smp import *
from ..utils import track_progress_rich


def extract_answer_from_item(response, all_choices, index2ans, default_answer=None, do_strip=False):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/multimodal-art-projection/OmniBench/blob/main/inference/answer_parsing.py
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    if do_strip:
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        if default_answer is None:
            pred_index = random.choice(all_choices)
        else:
            pred_index = default_answer
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def eval_omnibench(model, item, dataset_name=None):
    response = item["prediction"]

    all_choices = []
    index2ans = {}
    for i in range(4):
        current_option = chr(ord("A") + i)
        index2ans[current_option] = item[current_option]
        all_choices.append(current_option)

    opt = extract_answer_from_item(response, all_choices, index2ans, default_answer='N/A')
    if opt == item['GT']:
        return dict(hit=1, log='Match Log: Correct. ')
    else:
        return dict(hit=0, log='Match Log: Incorrect. ')


def omnibench_eval(model, data, meta, nproc, result_file, dataset_name=None):
    result = {}
    if osp.exists(result_file):
        result = load(result_file)
    answer_map = {i: c for i, c in zip(meta['index'], meta['answer'])}

    data = data[data['index'].isin(answer_map)]
    data['GT'] = [answer_map[idx] for idx in data['index']]
    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item['index'] not in result:
            items.append(item)

    tups = [dict(model=model, item=x, dataset_name=dataset_name) for x in items]
    keys = [x['index'] for x in items]
    if len(tups):
        res = track_progress_rich(eval_omnibench, tups, nproc=nproc, chunksize=nproc, save=result_file, keys=keys)
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v
    data['hit'] = [result[i]['hit'] for i in data['index']]
    data['log'] = [result[i]['log'] for i in data['index']]
    if 'GT' in data:
        data.pop('GT')
    return data


class OmniBench(ImageBaseDataset):

    MODALITY = 'OMNI'
    TYPE = 'ImageAudioMCQ'

    DATASET_URL = {
        "OmniBench": "https://huggingface.co/datasets/jamess/OmniBench/resolve/main/OmniBench.tsv",
    }

    def __init__(self, dataset='OmniBench', skip_noimg=True, skip_noaudio=True):
        ROOT = LMUDataRoot()
        # You can override this variable to save image files to a different directory
        self.dataset_name = dataset
        self.img_root = osp.join(ROOT, "omni", "OmniBench", "images")
        self.audio_root = osp.join(ROOT, "omni", "OmniBench", "audios")

        data = self.load_data(dataset)
        self.skip_noimg = skip_noimg
        if skip_noimg and 'image' in data:
            data = data[~pd.isna(data['image'])]
        if skip_noaudio and 'audio' in data:
            data = data[~pd.isna(data['audio'])]

        data['index'] = [str(x) for x in data['index']]

        self.meta_only = True

        # The image field can store the base64 encoded image or another question index (for saving space)
        if 'image' in data:
            data['image'] = [str(x) for x in data['image']]
            image_map = {x: y for x, y in zip(data['index'], data['image'])}
            for k in image_map:
                if len(image_map[k]) <= 64:
                    idx = image_map[k]
                    assert idx in image_map and len(image_map[idx]) > 64
                    image_map[k] = image_map[idx]

            images = [toliststr(image_map[k]) for k in data['index']]
            data['image'] = [x[0] if len(x) == 1 else x for x in images]
            self.meta_only = False

        # The audio field can store the base64 encoded audio or another question index (for saving space)
        if 'audio' in data:
            data['audio'] = [str(x) for x in data['audio']]
            audio_map = {x: y for x, y in zip(data['index'], data['audio'])}
            for k in audio_map:
                if len(audio_map[k]) <= 64:
                    idx = audio_map[k]
                    assert idx in audio_map and len(audio_map[idx]) > 64
                    audio_map[k] = audio_map[idx]

            audios = [toliststr(audio_map[k]) for k in data['index']]
            data['audio'] = [x[0] if len(x) == 1 else x for x in audios]
            self.meta_only = False

        if 'image_path' in data:
            paths = [toliststr(x) for x in data['image_path']]
            data['image_path'] = [x[0] if len(x) == 1 else x for x in paths]

        if 'audio_path' in data:
            paths = [toliststr(x) for x in data['audio_path']]
            data['audio_path'] = [x[0] if len(x) == 1 else x for x in paths]
        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            image_path = toliststr(os.path.join(self.img_root, line['image_path']))
            audio_path = toliststr(os.path.join(self.audio_root, line['audio_path']))
        else:
            image_path = self.dump_image(line)
            audio_path = self.dump_audio(line)

        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        options_prompt = ''
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'

        prompt = f"""
            Please answer the following question based on the given image and audio:
            {question}.
            Please choose only one answer from the following options:
            {options_prompt}
        """

        msgs = []
        if isinstance(image_path, list):
            msgs.extend([dict(type='image', value=p) for p in image_path])
        else:
            msgs = [dict(type='image', value=image_path)]

        if isinstance(audio_path, list):
            msgs.extend([dict(type='audio', value=p) for p in audio_path])
        else:
            msgs = [dict(type='audio', value=audio_path)]

        msgs.append(dict(type='text', value=prompt))

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        nproc = judge_kwargs.pop('nproc', 4)
        model = None

        result_file = get_intermediate_file_path(eval_file, '_exact_matching_result', 'pkl')

        data = load(eval_file)
        data = data.sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        # If not choice label, then use lower case
        for k in data.keys():
            data[k.lower() if k not in list(string.ascii_uppercase) else k] = data.pop(k)

        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
        data_map = {x: y for x, y in zip(data['index'], data['question'])}
        for k in data_map:
            assert k in meta_q_map, (
                f'eval_file should be the same as or a subset of dataset {self.dataset_name}'
            )

        data = omnibench_eval(model, data, meta, nproc, result_file, self.dataset_name)

        # load split
        eval_record = get_intermediate_file_path(eval_file, '_omnibench_result')
        dump(data, eval_record)
        data = load(eval_record)

        acc = report_acc(data)

        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(acc, score_file)

        return acc

    def dump_audio(self, line):
        os.makedirs(self.audio_root, exist_ok=True)

        if 'audio' in line:
            if isinstance(line['audio'], list):
                tgt_path = []
                if 'audio_path' in line:
                    audio_path = line['audio_path']
                else:
                    index = line['index']
                    audio_path = [f'{index}_{i}.wav' for i in range(len(line['audio']))]
                for aud, aud_name in zip(line['audio'], audio_path):
                    path = osp.join(self.audio_root, aud_name)
                    if not read_ok(path):
                        decode_base64_to_audio_file(aud, path)
                    tgt_path.append(path)

            elif isinstance(line['audio'], str) and 'audio_path' in line:
                assert isinstance(line['audio_path'], str)
                tgt_path = osp.join(self.audio_root, line['audio_path'])
                if not read_ok(tgt_path):
                    decode_base64_to_audio_file(line['audio'], tgt_path)
                tgt_path = [tgt_path]
            else:
                tgt_path = osp.join(self.audio_root, f"{line['index']}.wav")
                if not read_ok(tgt_path):
                    decode_base64_to_audio_file(line['audio'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'audio_path' in line
            tgt_path = toliststr(line['audio_path'])
            read_ok_flag = [read_ok(x) for x in tgt_path]
            # Might be the Relative Path
            if not all(read_ok_flag):
                tgt_path_abs = [osp.join(self.audio_root, x) for x in tgt_path]
                read_ok_flag = [read_ok(x) for x in tgt_path_abs]
                assert read_ok_flag, f"Field `audio` is missing and we could not find {tgt_path} both as absolute or relative paths. "  # noqa
                tgt_path = tgt_path_abs

        return tgt_path
