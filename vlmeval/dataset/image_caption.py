from .image_base import ImageBaseDataset
from ..smp import *


class ImageCaptionDataset(ImageBaseDataset):

    TYPE = 'CAPTION'

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

    def post_build(self, dataset):
        if dataset == 'COCO_VAL':
            self.img_root = osp.join(LMUDataRoot(), 'images', 'COCO')
