from vlmeval.smp import *
from functools import partial


MCQ_EXTRACT_SINGLE = """\
You are an AI assistant to help me to match a model prediction with several options of a multiple-choice question with one single answer. \
You are provided with a question, several options, and a model prediction. Your task is to find which option is most similar to the answer. \
If the meaning of all options are significantly different from the answer, output Z. \
Your should output **one single uppercase character** that is the choice label of the most similar option.

Example 1:
Question: What is the main object in image?
Options: A. teddy bear, B. rabbit, C. cat, D. dog.
Prediction: a cute teddy bear
Your output: A

Example 2:
Question: What is the main object in image?
Options: A. teddy bear, B. rabbit, C. cat, D. dog.
Prediction: Spider
Your output: Z

Your Task:
Question: {question}?\nOptions: {options}.\nPrediction: {prediction}\nYour output:
"""  # noqa: E501


MCQ_EXTRACT_MULTIPLE = """\
You are an AI assistant to help me to match a model prediction with several options of a multiple-choice question with one or multiple correct answers. \
You are provided with a question, several options, and a model prediction. Your task is to find out all options that are supported by the model prediction. \
Your should output **one or multiple uppercase character** that is the choice label of the options supported by the model prediction. \
The characters should be concatenated as a single string with no separators (space, newline, etc.) among them.

Example 1:
Question: What is the main object in image?
Options: A. teddy bear, B. rabbit, C. cat, D. dog.
Prediction: a cute teddy bear
Your output: A

Example 2:
Question: Select the options that best represent the types of land use depicted in the image.
Options: (A) Storage Tank, (B) Airport, (C) Park, (D) Farm.
Prediction: The correct options are (A) and (D).
Your output: AD

Example 3:
Question: Select all land use types that best represent those depicted in the image. Ensure to include all types present in the picture.
Options: A. Highway Bridge, B. Farmland, C. Dock, D. Residential unit.
Prediction: Since (C) Dock is not in the picture, the land use types present in the image are:\n\n(A) Highway Bridge\n(B) Farmland\n(D) Residential unit.
Your output: ABD

Your Task:
Question: {question}?\nOptions: {options}.\nPrediction: {prediction}\nYour output:
"""  # noqa: E501


def MCQ_VERIFIER(x, mode='single'):
    assert mode in ['single', 'multiple'], mode
    x = x.upper()
    chs = [ch for ch in x if ch in string.ascii_uppercase]
    if len(chs) == 0 or len(chs) != len(set(chs)):
        return False
    if mode == 'single':
        return len(chs) == 1
    elif mode == 'multiple':
        return len(chs) > 0


def MCQ_POST_PROCESSOR(x, mode='single'):
    x = x.upper()
    chs = [ch for ch in x if ch in string.ascii_uppercase]
    assert len(chs) > 0 and len(chs) == len(set(chs)), (x, chs)
    chs.sort()
    if mode == 'single':
        assert len(chs) == 1, (x, chs)
    return ''.join(chs)


class LLM_VERIFIER:

    def __init__(self, model, prompt, verifier):
        assert model.working(), f"The LLM Verifier is not working: {model}"
        self.model = model
        self.prompt = prompt
        self.verifier = verifier
        assert hasattr(self.verifier, '__call__'), "verifier should be a callable function"

    def verify(self, struct):
        if isinstance(struct, pd.Series):
            struct = dict(struct)
        assert isinstance(struct, dict), f"struct should be a dict, but got {type(struct)}"
        prompt = self.prompt.format(**struct)
        retry, temperature = 3, 0
        while retry:
            model_resp = self.model.generate(prompt, temperature=temperature)
            if self.verifier(model_resp) is not None:
                return self.verifier(model_resp)
            else:
                retry -= 1
                temperature += 0.5
        return None


class LLM_Extractor:

    def __init__(self, model, prompt, verifier=None):
        assert model.working(), f"The LLM Extractor is not working: {model}"
        self.model = model
        self.prompt = prompt
        self.verifier = verifier

    def extract(self, response):
        response = str(response)
        if isinstance(self.verifier, type):
            if istype(response, self.verifier):
                return self.verifier(response)
            else:
                prompt = self.prompt + f'\nResponse: {response}'
                retry, temperature = 3, 0
                while retry:
                    model_resp = self.model.generate(prompt, temperature=temperature)
                    if istype(model_resp, self.verifier):
                        return self.verifier(model_resp)
                    else:
                        retry -= 1
                        temperature += 0.5
                return None
        elif hasattr(self.verifier, '__call__'):
            if self.verifier(response):
                return response
            else:
                prompt = self.prompt + f'\nResponse: {response}'
                retry, temperature = 3, 0
                while retry:
                    model_resp = self.model.generate(prompt, temperature=temperature)
                    if self.verifier(model_resp):
                        return model_resp
                    else:
                        retry -= 1
                        temperature += 0.5
                return None
        else:
            raise NotImplementedError(f"Verifier {self.verifier} is not implemented.")


class LLM_Extractor_with_Context:

    def __init__(self, model, prompt_tmpl, verifier=None, post_processor=None):
        assert model.working(), f"The LLM Extractor is not working: {model}"
        self.model = model
        self.prompt_tmpl = prompt_tmpl
        self.verifier = verifier
        self.post_processor = post_processor

    def extract(self, sample):
        if isinstance(sample, pd.Series):
            sample = dict(sample)
        assert isinstance(sample, dict), sample
        # Manually handle choices
        if '{options}' in self.prompt_tmpl and 'options' not in sample:
            if 'A' in sample:
                choices_str = ''
                for ch in string.ascii_uppercase:
                    if ch in sample and not pd.isna(sample[ch]):
                        choices_str += f'{ch}: {sample[ch]}; '
                sample['options'] = choices_str
            elif sum(['option' in x for x in sample.keys()]) == 1:
                key_name = None
                for k in sample.keys():
                    if 'option' in k:
                        key_name = k
                        break
                sample['options'] = sample.pop(key_name)

        prompt = self.prompt_tmpl.format(**sample)
        prediction = sample['prediction']
        assert hasattr(self.verifier, '__call__'), self.verifier
        if self.verifier(prediction):
            return prediction if self.post_processor is None else self.post_processor(prediction)
        else:
            retry, temperature = 3, 0
            while retry:
                model_resp = self.model.generate(prompt, temperature=temperature)
                if self.verifier(model_resp):
                    return model_resp if self.post_processor is None else self.post_processor(model_resp)
                else:
                    retry -= 1
                    temperature += 0.5
            return None


class LLM_Extractor_MCQ_Single_Answer(LLM_Extractor_with_Context):

    def __init__(self, model):
        assert model.working(), f"The LLM Extractor is not working: {model}"
        self.model = model
        self.prompt_tmpl = MCQ_EXTRACT_SINGLE
        self.verifier = partial(MCQ_VERIFIER, mode='single')
        self.post_processor = partial(MCQ_POST_PROCESSOR, mode='single')


class LLM_Extractor_MCQ_Multiple_Answer(LLM_Extractor_with_Context):

    def __init__(self, model):
        assert model.working(), f"The LLM Extractor is not working: {model}"
        self.model = model
        self.prompt_tmpl = MCQ_EXTRACT_MULTIPLE
        self.verifier = partial(MCQ_VERIFIER, mode='multiple')
        self.post_processor = partial(MCQ_POST_PROCESSOR, mode='multiple')
