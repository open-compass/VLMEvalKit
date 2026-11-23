"""ImgMT prompt/schema, ufal/imgmt-devel, revision 4421a8424973009a72f83e80ce740509025f4ac9.

Copyright (c) 2026 ÚFAL; MIT license in LICENSE.
"""
import re
from typing import List

from PIL import Image
from pydantic import BaseModel, Field, model_validator

PROMPT_TEMPLATE = """
You are an OCR and translation system.

Task:
1. Detect all the text visible in the image.
2. For each detected text element:
   - Extract the original text exactly as it appears.
   - Translate it to natural <target_language>.
   - Estimate a normalized bounding box around the text.

Bounding box requirements:
- Use normalized coordinates in the range [0.0, 1.0].
- Coordinates are relative to the full image dimensions.
- Format:
  - x = left edge / image width
  - y = top edge / image height
  - w = box width / image width
  - h = box height / image height
- w and h must be greater than 0
- x + w and y + h must be less than or equal to 1
- Bounding boxes should tightly enclose the visible text.
- Preserve reading order from top-to-bottom and left-to-right.
- Each source text must have exactly one corresponding bounding box and one translation.

Output requirements:
- Return ONLY valid JSON.
- Do not include markdown fences.
- Do not include explanations or additional fields.
- Do not omit any detected text.
- If a translation is uncertain, provide the most likely translation.

Example output structure:

{
  "source_text_bounding_boxes": [
    {
      "x": 0.10,
      "y": 0.20,
      "w": 0.30,
      "h": 0.08
    }
  ],
  "source_texts": [
    "<source_language> text"
  ],
  "target_texts": [
    "<target_language> translation"
  ]
}

Analyze the provided image and return the JSON result.
"""


class BoundingBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.0, le=1.0)
    h: float = Field(gt=0.0, le=1.0)

    @model_validator(mode='after')
    def validate_edges(self):
        if self.x + self.w > 1 or self.y + self.h > 1:
            raise ValueError('Bounding box extends outside the image')
        return self


class OCRTranslationResult(BaseModel):
    source_text_bounding_boxes: List[BoundingBox]
    source_texts: List[str]
    target_texts: List[str]

    @model_validator(mode='after')
    def validate_lengths(self):
        n = len(self.source_texts)
        if len(self.source_text_bounding_boxes) != n:
            raise ValueError('source_text_bounding_boxes and source_texts must have equal length')
        if len(self.target_texts) != n:
            raise ValueError('target_texts and source_texts must have equal length')
        return self


def denormalize_bounding_boxes(img_path: str, result: OCRTranslationResult) -> dict:
    with Image.open(img_path) as img:
        (width, height) = img.size
    data = result.model_dump()
    scaled_boxes = []
    for box in result.source_text_bounding_boxes:
        scaled_boxes.append(dict(x=round(box.x * width), y=round(box.y * height),
                                 w=round(box.w * width), h=round(box.h * height)))
    data['source_text_bounding_boxes'] = scaled_boxes
    return data


def parse_prediction(text, image_path):
    text = str(text).strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    result = OCRTranslationResult.model_validate_json(text)
    return denormalize_bounding_boxes(image_path, result)
