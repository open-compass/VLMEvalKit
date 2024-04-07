import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
from PIL import Image
from .base import BaseModel


class MMAlaya(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='DataCanvas/MMAlaya', **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cpu', trust_remote_code=True).eval()
        # need initialize tokenizer
        model.initialize_tokenizer(self.tokenizer)
        self.model = model.cuda()

        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')
        torch.cuda.empty_cache()

    def generate_inner(self, message, dataset=None):
        # read image
        prompt, image_path = self.message_to_promptimg(message)
        image = Image.open(image_path).convert('RGB')
        # tokenize prompt, and proprecess image
        input_ids, image_tensor, stopping_criteria = self.model.prepare_for_inference(
            prompt,
            self.tokenizer,
            image,
            return_tensors='pt')
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids.cuda(),
                images=image_tensor.cuda(),
                do_sample=False,
                max_new_tokens=512,
                num_beams=1,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )
            # truncate input_ids in generate_ids and then decode to text
            input_token_len = input_ids.shape[1]
            response = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:].cpu(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0].strip()
        return response


if __name__ == '__main__':
    model = MMAlaya()
    response = model.generate(['./assets/apple.jpg', '请详细描述一下这张图片。'])
    print(response)

"""
export PYTHONPATH=$PYTHONPATH:/tmp/VLMEvalKit
CUDA_VISIBLE_DEVICES=0 python vlmeval/vlm/mmalaya.py
"""
