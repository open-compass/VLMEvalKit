import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY
import copy
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import logging

class LLaVA_HF(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", **kwargs):

        self.model_path = model_path

        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map="cuda"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
        except Exception as err:
            logging.critical(f"Failed to load Hugging Face LLaVA model from {model_path}.")
            raise err

        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        
        # Hugging Face's generation config doesn't accept temperature=0 with do_sample=False
        if not kwargs_default["do_sample"] and kwargs_default["temperature"] == 0:
            kwargs_default.pop("temperature", None)
            kwargs_default.pop("top_p", None)

        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )

        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def chat_inner(self, message, dataset=None):
        
        
        conversation = []
        images = []

        # Convert framework messages to HF Chat Template format
        for utter in message:
            content_list = []
            for item in utter["content"]:
                if item["type"] == "text":
                    content_list.append({"type": "text", "text": item["value"]})
                elif item["type"] == "image":
                    content_list.append({"type": "image"})
                    images.append(Image.open(item["value"]).convert("RGB"))
            
            conversation.append({
                "role": utter["role"],
                "content": content_list
            })

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.kwargs
            )

        # Slice the output to remove the input prompt tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        
        output = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return output

    def generate_inner(self, message, dataset=None):
        import torch

        content_list = []
        images = []
        
        # Convert single-turn framework message to HF Chat Template format
        for item in message:
            if item["type"] == "text":
                content_list.append({"type": "text", "text": item["value"]})
            elif item["type"] == "image":
                content_list.append({"type": "image"})
                images.append(Image.open(item["value"]).convert("RGB"))

        conversation = [
            {
                "role": "user",
                "content": content_list
            }
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt"
        ).to(self.model.device, torch.float16)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.kwargs
            )

        # Slice the output to remove the input prompt tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        
        output = self.processor.decode(generated_ids, skip_special_tokens=True).strip()
        return output
