import torch
from PIL import Image
import os.path as osp

class IDEFICS:

    INSTALL_REQ = False

    def __init__(self, name, with_context=False):
        assert name in ['idefics_9b_instruct', 'idefics_80b_instruct'] or osp.exists(name)
        self.name_map = {
            'idefics_9b_instruct': [
                '/mnt/petrelfs/share_data/duanhaodong/idefics-9b-instruct/',
                '/cpfs01/shared/llmeval/dhd/idefics-9b-instruct/',
                "HuggingFaceM4/idefics-9b-instruct"
            ],
            'idefics_80b_instruct': "HuggingFaceM4/idefics-80b-instruct"
        }
        if name in self.name_map:
            if isinstance(self.name_map[name], str):
                pth = self.name_map[name]
            elif isinstance(self.name_map[name], list):
                pth = None
                for s in self.name_map[name]:
                    if osp.exists(s):
                        pth = s
                    elif len(s.split('/')) == 2:
                        pth = s
                assert pth is not None
        else:
            pth = name
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        self.model = IdeficsForVisionText2Text.from_pretrained(pth, torch_dtype=torch.bfloat16, device_map='auto')
        self.processor = AutoProcessor.from_pretrained(pth)
        self.with_context = with_context

    def generate(self, image_path, prompt, dataset=None):
        if self.with_context:
            prompts = [
                [
                    "User: What is in this image?",
                    Image.open('/mnt/petrelfs/share_data/duanhaodong/Idefics.jpg'),
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
        generated_ids = self.model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=256)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = generated_text[0].split("\nAssistant: ")[-1]
        return text
    