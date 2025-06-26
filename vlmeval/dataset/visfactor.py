import re
from vlmeval import *
from .image_base import ImageBaseDataset


class VisFactor(ImageBaseDataset):
    TYPE = 'VQA'

    DATASET_URL = {
        'VisFactor': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor.tsv',
        'VisFactor_CoT': 'https://opencompass.openxlab.space/utils/VLMEval/VisFactor.tsv',
    }

    DATASET_MD5 = {
        'VisFactor': 'a069bcd8eb529e1d8c66e4cd7e279d13',
        'VisFactor_CoT': 'a069bcd8eb529e1d8c66e4cd7e279d13',
    }

    def replace_additional_tags(self, text, additional):
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

    def extract_last_json_answer(self, s):
        pattern = r'\{\s*"answer"\s*:\s*(.+?)\s*\}'
        matches = list(re.finditer(pattern, s))
        if not matches:
            return ''

        raw_value = matches[-1].group(1).strip()

        if (raw_value.startswith('"') and raw_value.endswith('"')) or \
           (raw_value.startswith("'") and raw_value.endswith("'")):
            raw_value = raw_value[1:-1]
        return raw_value

    def build_prompt(self, line):
        msgs = line['question'].replace('<br>', '\n')
        image_paths = self.dump_image(line)

        if str(line['additional']) != 'nan':
            additional = str(line['additional']).replace('<br>', '\n').split(';')
            msgs = self.replace_additional_tags(msgs, additional)

        if 'cot' in self.dataset_name.lower():
            msgs = msgs.replace(
                'Output: Respond',
                'Output: Solve the problem step-by-step. First, reason through the problem clearly. At the end, respond'
            )
        msgs = self.split_image_tags(msgs)

        for i in range(len(msgs)):
            if msgs[i][0] != '<':
                msgs[i] = dict(type="text", value=msgs[i])
            else:
                msgs[i] = dict(type="image", value=image_paths[int(msgs[i][-2])])

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        subtests = sorted(set([str(i) for i in data['category_id']]))
        accuracy = {key: {} for key in subtests}

        for index, row in data.iterrows():
            cid = str(row['category_id'])
            prediction = self.extract_last_json_answer(str(row['prediction']))
            answer = str(row['answer'])
            additional = str(row['additional'])

            if cid in ['CF1', 'CF2', 'MV1', 'MV2', 'MV3', 'P3', 'RL2', 'S1', 'S2', 'SS2', 'VZ1', 'VZ2'] \
               or (cid == 'VZ3' and not additional.isdigit()):
                if prediction.lower() in ['t', 'y', '1', 'true', 'yes']:
                    data.at[index, 'pred'] = 'T'
                elif prediction.lower() in ['f', 'n', '0', 'false', 'no']:
                    data.at[index, 'pred'] = 'F'
                data.at[index, 'correct'] = (data.at[index, 'pred'] == answer)
            elif cid in ['CS1', 'CS2', 'CS3']:
                data.at[index, 'pred'] = prediction.lower()
                data.at[index, 'correct'] = (data.at[index, 'pred'] in [i.lower() for i in answer.split(',')])
            elif cid in ['CF3', 'I3', 'MA1', 'SS3', 'VZ3']:
                data.at[index, 'pred'] = prediction
                data.at[index, 'correct'] = (data.at[index, 'pred'] == answer)

        for subtest in subtests:
            for index, row in data.iterrows():
                cid = str(row['category_id'])
                eid = str(row['eval_index'])
                if eid not in accuracy[cid]:
                    accuracy[cid][eid] = [row['correct']]
                else:
                    accuracy[cid][eid].append(row['correct'])

        for subtest in accuracy:
            for eval_index in accuracy[subtest]:
                accuracy[subtest][eval_index] = 1 if all(accuracy[subtest][eval_index]) else 0

        for subtest in accuracy:
            results = [accuracy[subtest][eval_index] for eval_index in accuracy[subtest]]
            accuracy[subtest] = sum(results) / len(results)

        accuracy['ALL'] = sum([accuracy[s] for s in accuracy]) / len([accuracy[s] for s in accuracy])
        with open(eval_file.replace('.xlsx', '_acc.csv'), 'w') as f:
            for key in accuracy:
                f.write(f'{key},{accuracy[key]}\n')

        return accuracy
