from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import warnings
import requests

class InstructBLIPHF:

    INSTALL_REQ = False

    def __init__(self, model_path='Salesforce/instructblip-vicuna-7b', **kwargs):
        assert model_path in ['Salesforce/instructblip-vicuna-13b', 'Salesforce/instructblip-vicuna-7b']
        model = InstructBlipForConditionalGeneration.from_pretrained(model_path, device_map='cpu')
        processor = InstructBlipProcessor.from_pretrained(model_path, device_map='cpu')
        model.eval()

        self.dev    ice = torch.device("cuda") if torch.cuda.is_available() else "cpu"
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

    def has_hint(self, prompt):
        return 'Hint: ' in prompt and 'Question: ' in prompt
    
    def remove_hint(self, prompt):
        prefix = prompt.split('Hint: ')[0]
        suffix = prompt.split('Question: ')[1]
        return prefix + 'Question: ' + suffix

    def generate(self, image_path, prompt, dataset=None):
        image = Image.open(image_path).convert("RGB")        
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        
        if len(inputs['qformer_input_ids']) >= 500:
            if self.has_hint(prompt):
                prompt = self.remove_hint(prompt)
                inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
                if len(inputs['qformer_input_ids']) >= 500:
                    warnings.warn("Prompt Length Exceeded 500. ")
                    raise NotImplementedError
            else:
                warnings.warn("Prompt Length Exceeded 500. ")
                raise NotImplementedError

        outputs = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return generated_text