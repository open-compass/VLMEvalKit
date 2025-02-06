import torch
import re
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import copy


class Moondream1(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="vikhyatk/moondream1", **kwargs):
        try:
            from transformers import (
                AutoModelForCausalLM,
                CodeGenTokenizerFast as Tokenizer,
            )
        except Exception as e:
            logging.critical(
                "Please install Transformers version 4.36.2 by running: 'pip install transformers==4.36.2', "
                "please intall torchvision>=0.16."
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
        self.tokenizer = Tokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))

        prompt_wtmpl = f"<image>\n\nQuestion: {prompt}\n\nAnswer:"
        answer = self.model.generate(
            enc_image,
            prompt_wtmpl,
            eos_text="<END>",
            tokenizer=self.tokenizer,
            **self.kwargs,
        )[0]
        cleaned_answer = re.sub("<$", "", re.sub("END$", "", answer)).strip()
        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(["MMMU"], dataset):
            return False
        if DATASET_TYPE(dataset) == "MCQ" or dataset in [
            "MMVet",
        ]:
            return True

        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]
        if dataset == "MMVet":
            prompt = question + "\nAnswer the question directly. "
        elif DATASET_TYPE(dataset) == "MCQ":
            options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
            options_prompt = ""
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            prompt = f"Hint: {hint}\n" if hint is not None else ""
            prompt += f"{question}\n"
            prompt += (
                f"{options_prompt}\nAnswer with the option’s letter from the given choices directly. "
                if len(options)
                else "Answer the question directly. "
            )
        else:
            raise NotImplementedError

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message


class Moondream2(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path="vikhyatk/moondream2", revision="2025-01-09", **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            logging.critical(
                """Please install Transformers version 4.44 by running: "pip install transformers==4.44.0",
            please intall torchvision>=0.16."""
            )
            raise e

        assert osp.exists(model_path) or splitlen(model_path) == 2

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map={"": "cuda"},
            revision=revision,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        default_kwargs = dict(max_new_tokens=512)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs

        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        prompt, img = self.message_to_promptimg(message)
        enc_image = self.model.encode_image(Image.open(img))
        print(f"prompt for {dataset} -> ", prompt)

        answer = self.model.query(enc_image, prompt)["answer"]
        cleaned_answer = answer.strip()

        return cleaned_answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None

        if listinstr(["MMMU"], dataset):
            return False
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        elif dataset in [
            "ChartQA_TEST",
            "TextVQA_VAL",
            "DocVQA_VAL",
            "POPE",
            "RealWorldQA",
            "TallyQA",
            "CountbenchQA",
            "MMVet",
        ]:
            return True
        else:
            return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)
        question = line["question"]

        if dataset == "ChartQA_TEST":
            prompt = (
                "Analyze the chart carefully, consider both visual features and data values,"
                " and provide a precise answer without any additional explanation or formatting. "
                + question
            )
        elif dataset == "TextVQA_VAL":
            prompt = (
                "Read the text in the image and provide a brief lowercase answer. "
                "Respond 'unanswerable' only if there is no plausible answer. "
                + question
            )
        elif dataset == "DocVQA_VAL":
            prompt = question + " The answer should be a short text span taken verbatim from the document."
        elif dataset == "POPE":
            prompt = f"{question}\nAnswer yes or no."
        elif dataset == "RealWorldQA":
            prompt = question
        elif dataset == "TallyQA" or dataset == "CountbenchQA":
            prompt = (
                "Look at the image carefully and count the objects. "
                "Answer with just a number, without any additional text. "
                + question
            )

        elif dataset == "MMVet":
            prompt = question + "\nAnswer the question directly. "
        elif DATASET_TYPE(dataset) == "MCQ":
            options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
            options_prompt = ""
            for key, item in options.items():
                options_prompt += f"{key}. {item}\n"

            hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
            prompt = f"Hint: {hint}\n" if hint is not None else ""
            prompt += f"{question}\n"
            prompt += (
                f"{options_prompt}\nAnswer with the option’s letter from the given choices directly. "
                if len(options)
                else "Answer the question directly. "
            )
        else:
            raise NotImplementedError

        message = [dict(type="text", value=prompt)]
        message.extend([dict(type="image", value=s) for s in tgt_path])
        return message
