import torch
from PIL import Image
import os.path as osp
import warnings
from .base import BaseModel
from ..smp import splitlen, listinstr
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image


class IDEFICS(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self,
                 model_pth='HuggingFaceM4/idefics-9b-instruct',
                 **kwargs):
        assert osp.exists(model_pth) or splitlen(model_pth) == 2
        from transformers import IdeficsForVisionText2Text, AutoProcessor
        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_pth, torch_dtype=torch.bfloat16, device_map='auto')
        self.processor = AutoProcessor.from_pretrained(model_pth)
        kwargs_default = {'max_length': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        self.file_root = osp.dirname(__file__)
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        prompts = ['Users:'] + [x['value'] for x in message] + ['<end_of_utterance>', '\nAssistant: ']
        inputs = self.processor(prompts, add_end_of_utterance_token=False, return_tensors='pt').to('cuda')
        exit_condition = self.processor.tokenizer('<end_of_utterance>', add_special_tokens=False).input_ids
        bad_words_ids = self.processor.tokenizer(
            ['<image>', '<fake_token_around_image>'],
            add_special_tokens=False).input_ids

        generated_ids = self.model.generate(
            **inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, **self.kwargs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        text = generated_text[0].split('\nAssistant: ')[-1]
        return text


class IDEFICS2(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='HuggingFaceM4/idefics2-8b',
                 **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map='cuda'
        )
        kwargs_default = {'max_new_tokens': 512}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def _process(self, formatted_messages, formatted_images):
        inputs = self.processor(text=formatted_messages, images=formatted_images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def build_prompt_default(self, message, add_brief=False):
        prompt, images = "User:", []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                prompt += msg['value'].strip()
        if add_brief:
            prompt += "\nGive a very brief answer."
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_puremcq(self, message):
        prompt, images = "User:", []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                instruction = instruction.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter."
                )
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        return prompt, images

    def build_prompt_mmmu(self, message):
        prompt, images, img_counter = "User: Question: ", [], 1
        for msg in message:
            if msg['type'] == 'image':
                prompt += f'<image {img_counter}>:<image>\n'
                img_counter += 1
        img_counter = 1

        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += f'<image {img_counter}> '
                img_counter += 1
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                instruction = instruction.replace(
                    "Question:",
                    ""
                )
                instruction = instruction.replace(
                    "Please select the correct answer from the options above.",
                    "Answer with the letter."
                )
                prompt += instruction.strip()
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def build_prompt_mathvista(self, message):
        prompt, images = "User:", []
        for msg in message:
            if msg['type'] == 'image':
                img = load_image(msg['value'])
                images.append(img)
                prompt += '<image>'
            elif msg['type'] == 'text':
                instruction = msg['value'].strip()
                instruction = instruction.replace(
                    "Hint: ",
                    ""
                )
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant:"
        return prompt, images

    def generate_inner(self, message, dataset=None):
        if dataset == "MMBench_TEST_EN":
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset == "MMBench_TEST_CN":
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset == "MME":
            formatted_messages, formatted_images = self.build_prompt_default(message, add_brief=True)
        elif dataset == "MMStar":
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset == "MMMU_DEV_VAL":
            formatted_messages, formatted_images = self.build_prompt_mmmu(message)
        elif dataset == "MathVista_MINI":
            formatted_messages, formatted_images = self.build_prompt_mathvista(message)
        elif dataset == "HallusionBench":
            formatted_messages, formatted_images = self.build_prompt_default(message)
        elif dataset == "AI2D_TEST":
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset == "OCRBench":
            formatted_messages, formatted_images = self.build_prompt_default(message)
        elif dataset == "SEEDBench_IMG":
            formatted_messages, formatted_images = self.build_prompt_puremcq(message)
        elif dataset == "MMVet":
            formatted_messages, formatted_images = self.build_prompt_default(message, add_brief=True)
        elif dataset == "LLaVABench":
            formatted_messages, formatted_images = self.build_prompt_default(message)
        else:
            raise ValueError("Unknown dataset.")

        inputs = self._process(formatted_messages, formatted_images)

        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_text = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)[0]
        response = generated_text.strip()
        print(dataset, " | ", formatted_messages.replace("\n", "\\n"), " | ", response)
        return response
