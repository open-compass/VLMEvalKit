import contextlib
import logging
import os

import torch
import torch.nn as nn
from transformers import BertTokenizer, LlamaTokenizer, StoppingCriteriaList

from .minigpt4_utils.eva_vit import create_eva_vit_g
from .minigpt4_utils.modelling_llama import LlamaForCausalLM
from .minigpt4_utils.Qformer import BertConfig, BertLMHeadModel
from .minigpt4_utils.utils import StoppingCriteriaSub, download_cached_file, is_url
from vlmeval.utils import DATASET_TYPE


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
class MiniGPT4_MMB(nn.Module):
    def __init__(
            self,
            model_name='7B',
            q_former_model='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth',  # noqa
            img_size=224,
            num_query_token=32,
            **kwargs):
        
        super().__init__()

        self.model_name = model_name
        assert model_name in ['7B', '13B'], 'Invalid model name'
        if model_name == '7B':
            self.llama_path = '/cpfs01/shared/llmeval/dhd/vicuna-7b-v0'
            self.minigpt_path = 'http://opencompass.openxlab.space/utils/Weights/pretrained_minigpt4_7b.pth'
        elif model_name == '13B':
            self.llama_path = '/cpfs01/shared/llmeval/dhd/vicuna-13b-v0'
            self.minigpt_path = 'http://opencompass.openxlab.space/utils/Weights/pretrained_minigpt4.pth'

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = self.init_tokenizer()

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder('eva_clip_g', img_size)
    
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = False
        for name, param in self.ln_vision.named_parameters():
            param.requires_grad = False
        self.ln_vision = self.ln_vision.eval()
        self.ln_vision.train = False

        print('Loading VIT Done')

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features)
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        for _, param in self.Qformer.named_parameters():
            param.requires_grad = False
        self.Qformer = self.Qformer.eval()
        self.Qformer.train = False
        self.query_tokens.requires_grad = False
        print('Loading Q-Former Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.llama_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        self.llama_model = LlamaForCausalLM.from_pretrained(
            self.llama_path,
            torch_dtype=torch.float16,
        )

        for _, param in self.llama_model.named_parameters():
            param.requires_grad = False
        print('Loading LLAMA Done')

        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size,
                                    self.llama_model.config.hidden_size)
        stop_words_ids = [
            torch.tensor([835]).to(self.device),
            torch.tensor([2277, 29937]).to(self.device),
        ]
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
        self.load_from_pretrained(self.minigpt_path)
        self.to(self.device)
        default_kwargs = dict(
            max_new_tokens=128,
            num_beams=1,
            do_sample=True,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1,
            length_penalty=-1,
            temperature=1.0,
            stopping_criteria=self.stopping_criteria,
            num_return_sequences=1)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        

    @classmethod
    def init_Qformer(cls,
                     num_query_token,
                     vision_width,
                     cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained('bert-base-uncased')
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size))
        query_tokens.data.normal_(mean=0.0,
                                  std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(url_or_filename,
                                               check_hash=False,
                                               progress=True)
            checkpoint = torch.load(cached_file, map_location='cpu')
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location='cpu')
        else:
            raise RuntimeError('checkpoint url or path is invalid')

        state_dict = checkpoint['model']

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info('load checkpoint from %s' % url_or_filename)

        return msg

    def init_vision_encoder(cls, model_name, img_size):
        assert model_name == 'eva_clip_g', 'vit model must be eva_clip_g for current version of MiniGPT-4'
        visual_encoder = create_eva_vit_g(img_size, drop_path_rate=0, use_checkpoint=False, precision='fp16')

        ln_vision = LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision

    @classmethod
    def init_tokenizer(cls):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        return tokenizer

    def maybe_autocast(self, dtype=torch.float16):
        if self.device == 'cuda':
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_img(self, image):
        device = image.device

        with self.maybe_autocast():
            image_embeds = self.ln_vision(
                self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1],
                                    dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True)

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1],
                                    dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
    
    def load_image(self, image_path):
        from PIL import Image
        from torchvision import transforms
        img = Image.open(image_path)
        resize = transforms.Resize(size=(224, 224), interpolation=3)
        convert_tensor = transforms.ToTensor()
        norm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        img = resize(img)
        img = convert_tensor(img)
        img = norm(img)
        return img.to(self.device)
    
    def generate(self, image_path, prompt, dataset=None):
        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            self.kwargs['num_beams'] = 5
            self.kwargs['max_new_tokens'] = 20
            self.kwargs['length_penalty'] = -1
            self.kwargs['do_sample'] = False

        prompt = f'###Human: <Img><ImageHere></Img> {prompt} ###Assistant:'
        image_tensor = self.load_image(image_path)
        image_embeds, _ = self.encode_img(image_tensor[None])
        prompt_segs = prompt.split('<ImageHere>')
        prompt_seg_tokens = [
            self.llama_tokenizer(seg,
                                return_tensors='pt',
                                add_special_tokens=i == 0).to(self.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        prompt_seg_embs = [
            self.llama_model.model.embed_tokens(seg)
            for seg in prompt_seg_tokens
        ]
        prompt_seg_embs = [prompt_seg_embs[0], image_embeds, prompt_seg_embs[1]]
        prompt_embs = torch.cat(prompt_seg_embs, dim=1)
        outputs = self.llama_model.generate(inputs_embeds=prompt_embs, **self.kwargs)
        output_token = outputs[0]
        if output_token[0] == 0:
            output_token = output_token[1:]
        if output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = self.post_process(output_text)
        return output_text

    def post_process(self, output_text):
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        return output_text
