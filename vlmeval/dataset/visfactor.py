import re
from vlmeval import *
from .image_base import ImageBaseDataset


def print_msgs(msgs):
    for m in msgs:
        if m['type'] == 'text':
            print(m['value'])
        elif m['type'] == 'image':
            image_data = base64.b64decode(m['value'])
            image = Image.open(io.BytesIO(image_data))
            display(image)


class VisFactor(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'VisFactor': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor.tsv',
    }

    DATASET_MD5 = {
        'VisFactor': '51d9b907d47a438868691a29d8817e83',
    }

    def replace_additional_placeholders(self, text, additional):
        def replacer(match):
            index = int(match.group(1))
            if 0 <= index < len(additional):
                return additional[index]
            else:
                return match.group(0)
        return re.sub(r"<ADDITIONAL_(\d+)>", replacer, text)

    def split_image_tags(self, text):
        parts = re.split(r'(<IMAGE_\d+>)', text)
        return [part for part in parts if part != '']

    def build_prompt(self, line):
        msgs = line['question'].replace('<br>', '\n')
        image_paths = self.dump_image(line)

        if str(line['additional']) != 'nan':
            additional = str(line['additional']).replace('<br>', '\n').split(';')
            msgs = self.replace_additional_placeholders(msgs, additional)

        msgs = self.split_image_tags(msgs)

        for i in range(len(msgs)):
            if msgs[i][0] != '<':
                msgs[i] = dict(type="text", value=msgs[i])
            else:
                msgs[i] = dict(type="image", value=image_paths[int(msgs[i][-2])])

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data
        subtests = sorted(set([str(i['category_id']) for i in data]))
        accuracy = {key: {} for key in subtests}

        for row in data:
            if row['answer'] in ['T', 'F'] and row['category_id'] in ['CF1', 'CF2', 'MV1', 'MV2', 'MV3', 'P3', 'RL2', 'S1', 'S2', 'SS2', 'VZ1', 'VZ2', 'VZ3']:  # noqa
                if row['prediction'].lower() in ['t', 'y', 'true', 'yes']:
                    pred = 'T'
                elif row['prediction'].lower() in ['f', 'n', 'false', 'no']:
                    pred = 'F'
                row['correct'] = (pred == row['answer'])
            elif row['answer'][0] == '(' and row['category_id'] in ['CF3']:
                correct1 = (row['prediction'] == row['answer'])
                pred = row['prediction'].split(',')
                pred = [i.strip().replace('(', '').replace(')', '') for i in pred]
                correct2 = ((pred[0] == row['answer'][1]) and (pred[1] == row['answer'][-2]))
                row['correct'] = (correct1 or correct2)
            elif row['answer'].isdigit() and row['category_id'] in ['I3', 'MA1', 'SS3']:
                row['correct'] = (row['prediction'] == row['answer'])
            elif len(row['answer']) == 1 and row['category_id'] in ['VZ3']:
                row['correct'] = (row['prediction'] == row['answer'])
            elif row['category_id'] in ['CS1', 'CS2', 'CS3']:
                ans = row['answer'].split(',')
                row['correct'] = (row['prediction'] in ans)

        for subtest in subtests:
            for row in data:
                if row['eval_index'] not in accuracy[row['category_id']]:
                    accuracy[row['category_id']][row['eval_index']] = [row['correct']]
                else:
                    accuracy[row['category_id']][row['eval_index']].append(row['correct'])

        for subtest in accuracy:
            for eval_index in accuracy[subtest]:
                accuracy[subtest][eval_index] = 1 if all(accuracy[subtest][eval_index]) else 0

        for subtest in accuracy:
            results = [accuracy[subtest][eval_index] for eval_index in accuracy[subtest]]
            accuracy[subtest] = sum(results) / len(results)

        accuracy['ALL'] = sum([accuracy[subtest] for subtest in accuracy]) / len([accuracy[subtest] for subtest in accuracy])  # noqa

        return accuracy
