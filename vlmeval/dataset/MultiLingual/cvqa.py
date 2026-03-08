from vlmeval.smp import *
from ..image_mcq import ImageMCQDataset


class CVQA(ImageMCQDataset):

    DATASET_URL = {
        'CVQA': 'https://opencompass.openxlab.space/utils/VLMEval/CVQA.tsv',
        # The all English version of CVQA
        'CVQA_EN': 'https://opencompass.openxlab.space/utils/VLMEval/CVQA.tsv'
    }
    DATASET_MD5 = {
        'CVQA': 'c04705c508998abe255e2b34f7bde699',
        'CVQA_EN': 'c04705c508998abe255e2b34f7bde699',
    }

    def build_prompt(self, line):

        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        question = line['question'] if self.dataset_name == 'CVQA' else line['question_en']

        def cand2key(cand):
            return cand if self.dataset_name == 'CVQA' else f'{cand}_en'

        options = {
            cand: line[cand2key(cand)]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand2key(cand)])
        }
        # Will not include additional prompt text here (cuz the problem is multi lingual)
        prompt = f'{question}\n'
        if len(options):
            for key, item in options.items():
                prompt += f'{key}. {item}\n'

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs
