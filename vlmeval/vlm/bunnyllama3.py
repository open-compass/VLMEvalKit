import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings

from .base import BaseModel
from ..smp import *
from ..utils import DATASET_TYPE


class BunnyLLama3(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='BAAI/Bunny-Llama-3-8B-V', **kwargs):
        assert model_path is not None
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        text = f"A chat between a curious user and an artificial intelligence assistant. \
            The assistant gives helpful, detailed, and polite answers to the user's questions. \
            USER: <image>\n{prompt} ASSISTANT:"
        text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)

        output_ids = self.model.generate(input_ids, images=image_tensor, max_new_tokens=100, use_cache=True)[0]
        response = self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)
        return response
