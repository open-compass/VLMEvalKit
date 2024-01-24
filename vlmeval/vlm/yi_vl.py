import torch
import sys
from abc import abstractproperty
import os.path as osp
import warnings
from PIL import Image

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    
class Yi_VL:

    INSTALL_REQ = True
    
    def __init__(self, 
                 model_path='01-ai/Yi-VL-6B', 
                 root=None,
                 **kwargs):
        
        if root is None:
            warnings.warn('Please set root to the directory of Yi, which is cloned from here: https://github.com/01-ai/Yi ')
        
        self.root = osp.join(root,'VL')
        sys.path.append(self.root)

        from llava.mm_utils import get_model_name_from_path, load_pretrained_model
        from llava.model.constants import key_info
        
        disable_torch_init()
        key_info["model_path"] = model_path
        get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            device_map='cpu')
        self.model = self.model.cuda()
        
        kwargs_default = dict(temperature= 0.2,
                              num_beams= 1,
                              conv_mode= "mm_default",
                              top_p= None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")
        
    def generate(self, image_path, prompt, dataset=None):
        
        from llava.conversation import conv_templates
        from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import KeywordsStoppingCriteria, expand2square, tokenizer_image_token
        
        qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[self.kwargs['conv_mode']].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
            )
        
        image = Image.open(image_path)
        if getattr(self.model.config, "image_aspect_ratio", None) == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in self.image_processor.image_mean)
                )
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
            ][0]
        
        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        self.model = self.model.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                temperature=self.kwargs['temperature'],
                top_p=self.kwargs['top_p'],
                num_beams=self.kwargs['num_beams'],
                stopping_criteria=[stopping_criteria],
                max_new_tokens=1024,
                use_cache=True,
                )
            
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs