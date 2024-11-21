from .image_base import ImageBaseDataset
from ..smp import *


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

    DATASET_URL = {'COCO_VAL': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL.tsv',
        'COCO_VAL_impulse_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_impulse_noise_1.tsv',
        'COCO_VAL_zoom_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_zoom_blur_1.tsv',
        'COCO_VAL_snow_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_snow_1.tsv',
        'COCO_VAL_sample': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample.tsv',
        'COCO_VAL_sample_gaussian_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_gaussian_noise_1.tsv',
        'COCO_VAL_sample_gaussian_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_gaussian_noise_5.tsv',
        'COCO_VAL_sample_shot_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_shot_noise_1.tsv',
        'COCO_VAL_sample_shot_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_shot_noise_5.tsv',
        'COCO_VAL_sample_impulse_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_impulse_noise_1.tsv',
        'COCO_VAL_sample_impulse_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_impulse_noise_5.tsv',
        'COCO_VAL_sample_speckle_noise_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_speckle_noise_1.tsv',
        'COCO_VAL_sample_speckle_noise_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_speckle_noise_5.tsv',
        'COCO_VAL_sample_defocus_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_defocus_blur_1.tsv',
        'COCO_VAL_sample_defocus_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_defocus_blur_5.tsv',
        'COCO_VAL_sample_glass_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_glass_blur_1.tsv',
        'COCO_VAL_sample_glass_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_glass_blur_5.tsv',
        'COCO_VAL_sample_zoom_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_zoom_blur_1.tsv',
        'COCO_VAL_sample_zoom_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_zoom_blur_5.tsv',
        'COCO_VAL_sample_motion_blur_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_motion_blur_1.tsv',
        'COCO_VAL_sample_motion_blur_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_motion_blur_5.tsv',
        'COCO_VAL_sample_fog_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_fog_1.tsv',
        'COCO_VAL_sample_fog_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_fog_5.tsv',
        'COCO_VAL_sample_frost_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_frost_1.tsv',
        'COCO_VAL_sample_frost_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_frost_5.tsv',
        'COCO_VAL_sample_snow_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_snow_1.tsv',
        'COCO_VAL_sample_snow_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_snow_5.tsv',
        'COCO_VAL_sample_contrast_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_contrast_1.tsv',
        'COCO_VAL_sample_contrast_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_contrast_5.tsv',
        'COCO_VAL_sample_brightness_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_brightness_1.tsv',
        'COCO_VAL_sample_brightness_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_brightness_5.tsv',
        'COCO_VAL_sample_pixelate_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_pixelate_1.tsv',
        'COCO_VAL_sample_pixelate_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_pixelate_5.tsv',
        'COCO_VAL_sample_elastic_transform_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_elastic_transform_1.tsv',
        'COCO_VAL_sample_elastic_transform_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_elastic_transform_5.tsv',
        'COCO_VAL_sample_jpeg_compression_1': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_jpeg_compression_1.tsv',
        'COCO_VAL_sample_jpeg_compression_5': 'https://opencompass.openxlab.space/utils/VLMEval/COCO_VAL_sample_jpeg_compression_5.tsv'}

    DATASET_MD5 = {'COCO_VAL': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_impulse_noise_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_zoom_blur_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_snow_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_gaussian_noise_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_gaussian_noise_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_shot_noise_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_shot_noise_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_impulse_noise_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_impulse_noise_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_speckle_noise_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_speckle_noise_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_defocus_blur_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_defocus_blur_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_glass_blur_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_glass_blur_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_zoom_blur_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_zoom_blur_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_motion_blur_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_motion_blur_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_fog_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_fog_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_frost_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_frost_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_snow_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_snow_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_contrast_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_contrast_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_brightness_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_brightness_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_pixelate_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_pixelate_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_elastic_transform_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_elastic_transform_5': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_jpeg_compression_1': '72a5079dead060269ac222c5aa5128af',
        'COCO_VAL_sample_jpeg_compression_5': '72a5079dead060269ac222c5aa5128af'}

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
