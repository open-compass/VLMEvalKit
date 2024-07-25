from .image_base import ImageBaseDataset
from ..smp import *


class ImageMTDataset(ImageBaseDataset):

    TYPE = 'MT'
    DATASET_URL = {'MMDU': 'https://opencompass.openxlab.space/utils/VLMEval/MMDU.tsv'}
    DATASET_MD5 = {'MMDU': '848b635a88a078f49aebcc6e39792061'}

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

    def evaluate(self, eval_file, **judge_kwargs):
        pass
