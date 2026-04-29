import hashlib
import io
from base64 import b64decode

from PIL import Image

from .image_shortqa import ImageShortQADataset


class HLEDataset(ImageShortQADataset):
    HLE_BLANK_IMAGE_MD5 = 'b8718c65c71b998e9132229ac4b7c8a4'

    DATASET_URL = {
        'hle': 'https://opencompass.openxlab.space/utils/VLMEval/hle.tsv',
    }

    DATASET_MD5 = {
        'hle': 'a83cbdbea89f27c2aa5b8f34a8894b72',
    }

    def _is_hle_blank_image(self, image):
        if not isinstance(image, str):
            return False

        if hashlib.md5(image.encode('utf-8')).hexdigest() == self.HLE_BLANK_IMAGE_MD5:
            return True

        try:
            img = Image.open(io.BytesIO(b64decode(image))).convert('RGB')
        except Exception:
            return False

        colors = img.getcolors(maxcolors=8)
        if colors is None or len(colors) != 1:
            return False

        _, color = colors[0]
        return img.size == (101, 93) and color == (255, 255, 255)

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if 'image' in line and self._is_hle_blank_image(line['image']):
            return [dict(type='text', value=line['question'])]
        else:
            return super().build_prompt(line)
