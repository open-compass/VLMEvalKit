from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import warnings
import requests

class InstructBLIPHF:

    INSTALL_REQ = False

    def __init__(self, model_path='Salesforce/instructblip-vicuna-7b', **kwargs):
        assert model_path in ['Salesforce/instructblip-vicuna-13b', 'Salesforce/instructblip-vicuna-7b']
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map='cpu')
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b", device_map='cpu')
        model.eval()

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        device = self.device
        model.to(device)
        self.model = model
        self.processor = processor

        default_kwargs = dict(
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text