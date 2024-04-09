import warnings
from .base import BaseModel
from ..smp import *


class VisualGLM(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='THUDM/visualglm-6b', **kwargs):
        try:
            import sat
        except:
            warnings.warn('Please install SwissArmyTransformer to use VisualGLM')
        assert model_path is not None
        self.model_path = model_path

        from transformers import AutoModel
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        self.model = model
        self.kwargs = kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message)
        output, _ = self.model.chat(
            image_path=image_path,
            tokenizer=self.tokenizer,
            query=prompt,
            history=[],
            **self.kwargs
        )
        return output
