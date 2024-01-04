import torch
from PIL import Image
from abc import abstractproperty
import os.path as osp
import os 
from ..smp import *

class Emu:

    def __init__(self, 
                 name, 
                 model_path_map={
                     "emu2":"BAAI/Emu2",
                     "emu2_chat":"BAAI/Emu2_Chat"
                     }, 
                 **kwargs):
        
        self.model_path_map = model_path_map
        assert name in self.model_path_map or osp.exists(name) or splitlen(name) == 2
        if name in self.model_path_map:
            model_path = self.model_path_map[name]
        else:
            model_path = name

        assert osp.exists(model_path) or splitlen(model_path) == 2
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path) # "BAAI/Emu2-Chat"
        self.tokenizer = tokenizer
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        device = self.device
        model = AutoModelForCausalLM.from_pretrained(
            model_path, # "BAAI/Emu2-Chat"
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).to(device).eval()
        self.model = model
        self.kwargs = {'max_length': 128}
        
        
    def generate(self, image_path, prompt, dataset=None):
        query = '[<IMG_PLH>]' + prompt
        image = Image.open(image_path).convert('RGB')
        
        inputs = self.model.build_input_ids(
            text=[query],
            tokenizer=self.tokenizer,
            image=[image]
        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=64,
                length_penalty=-1)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text
    
    def interleave_generate(self, ti_list, dataset=None):
        query, images = '',[]
        for item in ti_list:
            if isimg(item):
                images.append(Image.open(item).convert('RGB'))
                query += '[<IMG_PLH>]'
            else:
                query += item
        
        inputs = self.model.build_input_ids(
            text=[query],
            tokenizer=self.tokenizer,
            image=images

        )

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=64,
                length_penalty=-1)

        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return output_text