import json

from .image_base import ImageBaseDataset
from .utils.Mia_Bench import get_score_dict
from ..smp import *
from .utils import build_judge, DEBUG_MESSAGE


class COCO_Caption_Scorer():
    def __init__(self, ref, gt):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider

        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    print('%s: %0.3f' % (m, sc * 100))
                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                print('%s: %0.3f' % (method, score * 100))
                total_scores[method] = score * 100

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores


class ImageCaptionDataset(ImageBaseDataset):

    TYPE = 'Caption'

    DATASET_URL = {
        'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
    }

    DATASET_MD5 = {
        'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
    }

    def load_data(self, dataset):
        data = super().load_data(dataset)
        if 'question' not in data:
            data['question'] = [(
                'Please describe this image in general. Directly provide the description, '
                'do not include prefix like "This image depicts". '
            )] * len(data)
        return data

    # It returns a dictionary of scores
    @classmethod
    def evaluate(self, eval_file, **kwargs):
        data = load(eval_file)
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        ref, gt = {}, {}
        for i, line in enumerate(lines):
            ref[str(i)] = [str(line['prediction'])]
            gt[str(i)] = eval(line['answer'])

        scorer = COCO_Caption_Scorer(ref, gt)
        coco_caption_score_dict = scorer.compute_scores()
        score_pth = eval_file.replace('.xlsx', '_score.json')
        dump(coco_caption_score_dict, score_pth)
        return coco_caption_score_dict


class Mia_Bench(ImageBaseDataset):
    TYPE = 'Caption'

    DATASET_URL = {
        'Mia-Bench': 'https://opencompass.openxlab.space/utils/VLMEval/Mia-Bench.tsv',
    }
    DATASET_MD5 = {
        'Mia-Bench': '3d120e2704666ff32eacf166609c114a',
    }

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.Mia_Bench import generate_prompt
        from openai import OpenAI
        import requests
        from io import BytesIO

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_API_BASE"))

        if 'model' in judge_kwargs:
            model = judge_kwargs['model']
        else:
            model = os.path.basename(os.environ.get('LOCAL_LLM'))
        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(storage):
            data = load(eval_file)
            data['score_raw'] = [_ for _ in range(len(data))]
            # model = build_judge(max_tokens=128, **judge_kwargs)
            # assert model.working(), ('MATH-Vision evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE)


            for i in tqdm(range(len(data))):
                line = data.loc[i]
                response = line['prediction']
                image = line['image_url']
                # image_res = requests.get(image)
                # temp_file = BytesIO()
                # temp_file.write(image_res.content)
                # imjs = json.load(temp_file)

                breakpoint()
                question = generate_prompt(line, response)
                generated = False

                attempt = 5
                while attempt > 0 and generated == False:
                    try:
                        rev_response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": question},
                                        {"type": "image_url",
                                         "image_url": image
                                         },
                                    ],
                                }
                            ],
                            max_tokens=2000
                        )
                        print(rev_response.choices[0].message.content.strip())
                        #data['score_raw'][i] = response.choices[0].message.content.strip()
                        generated = True
                    except:
                        attempt -= 1
            results = get_score_dict(data, 'score_raw')
            results.to_csv(storage, index=False, encoding='gbk')


