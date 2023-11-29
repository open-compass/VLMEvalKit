import os.path as osp
import warnings
from vlmeval.smp import *


class VisualGLM:

    INSTALL_REQ = False

    def __init__(self):
        try:
            import sat
        except:
            warnings.warn("Please install SwissArmyTransformer to use VisualGLM")
        self.model_paths = [
            "THUDM/visualglm-6b"
        ]
        model_path = None
        for m in self.model_paths:
            if osp.exists(m):
                model_path = m
                break
            elif len(m.split('/')) == 2:
                model_path = m
                break
        assert model_path is not None
                
        from transformers import AutoModel
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model =  AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = model

    def generate(self, image_path, prompt, dataset=None):
        
        output, _ = self.model.chat(
            image_path = image_path,
            tokenizer = self.tokenizer,
            query = prompt,
            history = []
        )
        return output