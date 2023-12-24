import torch
from PIL import Image
import os.path as osp
import warnings
from ..smp import splitlen

class IDEFICS:

    INSTALL_REQ = False

    def __init__(self, 
                 name, 
                 with_context=False, 
                 model_path_map = {
                     'idefics_9b_instruct': "HuggingFaceM4/idefics-9b-instruct",
                     'idefics_80b_instruct': "HuggingFaceM4/idefics-80b-instruct"
                 },
                 **kwargs):
        assert name in ['idefics_9b_instruct', 'idefics_80b_instruct'] or osp.exists(name)
        self.model_path_map = model_path_map
        if name in self.model_path_map:
            pth = self.model_path_map[name]
        else:
            pth = name
        assert osp.exists(pth) or splitlen(pth) == 2
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        self.model = IdeficsForVisionText2Text.from_pretrained(pth, torch_dtype=torch.bfloat16, device_map='auto')
        self.processor = AutoProcessor.from_pretrained(pth)
        self.with_context = with_context
        kwargs_default = {'max_length': 128}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.file_root = osp.dirname(__file__)
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def multi_generate(self, image_paths, prompt, dataset=None):
        image_prompts = []
        for i, pth in enumerate(image_paths):
            image_prompts.append(f'Image {i + 1}: ')
            image_prompts.append(Image.open(pth))

        prompts = ['User: ' + prompt] + image_prompts + ['<end_of_utterance>', '\nAssistant: ']
        inputs = self.processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, **self.kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = generated_text[0].split("\nAssistant: ")[-1]
        return text

    def generate(self, image_path, prompt, dataset=None):
        if self.with_context:
            prompts = [
                [
                    "User: What is in this image?",
                    Image.open(osp.join(self.file_root, 'misc/Idefics.jpg')),
                    "<end_of_utterance>",
                    "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix. Idefix is running on the ground.<end_of_utterance>",
                    "\nUser: " + prompt,
                    Image.open(image_path), 
                    "<end_of_utterance>", 
                    "\nAssistant: "
                ]
            ]
        else:
            prompts = [
                [
                    "User: " + prompt,
                    Image.open(image_path), 
                    "<end_of_utterance>", 
                    "\nAssistant: "
                ]
            ]
        inputs = self.processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
        exit_condition = self.processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, **self.kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = generated_text[0].split("\nAssistant: ")[-1]
        return text
    