"""Requires Transformer 4.28 and above, implementation may change according the
Llama implementation."""
import logging
import re

import torch
import torch.nn as nn
from vlmeval import DATASET_TYPE
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import LlamaForCausalLM, LlamaTokenizer


class InstructBLIP_MMB(Blip2Base):
    def __init__(
        self,
        model_name='7B', 
        llama_model=None,
        img_size=224,
        num_query_token=32,
        query_len=128, 
        prompt_template='v1',
        **kwargs
    ):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        assert model_name in ['7B', '13B'], 'Invalid model name'
        if model_name == '7B':
            self.instructblip_path = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna7b_trimmed.pth'
        elif model_name == '13B':
            self.instructblip_path = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_vicuna13b_trimmed.pth'
        
        self.llama_model = llama_model
        self.tokenizer = self.init_tokenizer(truncation_side='left')

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            'eva_clip_g', img_size, 0, False, 'fp16')
        
        for name, param in self.visual_encoder.named_parameters():
            param.requires_grad = False
        self.visual_encoder = self.visual_encoder.eval()
        self.visual_encoder.train = disabled_train
        logging.info('freeze vision encoder')
        self.prompt_template = prompt_template
        assert prompt_template in ['v1', 'v0'], 'Invalid prompt template'

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None

        self.llm_tokenizer = LlamaTokenizer.from_pretrained(
            self.llama_model, use_fast=False, truncation_side='left')

        self.llm_model = LlamaForCausalLM.from_pretrained(
            self.llama_model, torch_dtype=torch.float16)
        
        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})
        self.llm_tokenizer.padding_side = 'left'

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False

        self.llm_proj = nn.Linear(self.Qformer.config.hidden_size,
                                  self.llm_model.config.hidden_size)
        self.load_from_pretrained(self.instructblip_path)

        self.query_len = query_len
        
        self._lemmatizer = None
        default_kwargs = dict(
            do_sample=False,
            num_beams=1,
            max_length=128,
            min_length=1,
            repetition_penalty=1.5,
            length_penalty=1,
            num_return_sequences=1)
        default_kwargs.update(kwargs)
        self.kwargs = default_kwargs
        self.to(self.device)

    def load_from_pretrained(self, url_or_filename):
        import os
        from .minigpt4_utils.utils import is_url, download_cached_file
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

        logging.info('load checkpoint from %s' % url_or_filename)
        return msg

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

    @torch.no_grad()
    def generate(self, image_path, prompt, dataset=None):
        kwargs = self.kwargs.copy()
        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            kwargs['num_beams'] = 5
            kwargs['max_length'] = 32
        image = self.load_image(image_path)[None]

        prompt = [prompt]
        query_tokens = self.query_tokens.expand(1, -1, -1)
        
        text_Qformer = self.tokenizer(
            prompt,
            padding='longest',
            truncation=True,
            max_length=self.query_len,
            return_tensors='pt',
        ).to(image.device)

        query_atts = torch.ones(query_tokens.size()[:-1],
                                dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],
                                    dim=1)

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        query_output = self.Qformer.bert(
            text_Qformer.input_ids,
            attention_mask=Qformer_atts,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.llm_proj(
            query_output.last_hidden_state[:, :query_tokens.size(1), :])
        atts_llm = torch.ones(inputs_llm.size()[:-1],
                              dtype=torch.long).to(image.device)

        if self.prompt_template == 'v0':
            prompt = ['###Human: ' + p + '###Assistant:' for p in prompt]
        elif self.prompt_template == 'v1':
            prompt = ['USER: ' + p + 'ASSISTANT: '  for p in prompt]

        llm_tokens = self.llm_tokenizer(prompt,
                                        padding='longest',
                                        return_tensors='pt').to(image.device)

        with self.maybe_autocast():
            inputs_embeds = self.llm_model.get_input_embeddings()(
                llm_tokens.input_ids)
            inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
            attention_mask = torch.cat([atts_llm, llm_tokens.attention_mask],
                                       dim=1)

            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs
            )
        outputs[outputs == 0] = 2  # convert output id 0 to 2 (eos_token_id)
        output_text = self.llm_tokenizer.batch_decode(outputs,
                                                      skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        # output_text = self.post_process(output_text[0])
        return output_text

    def post_process(self, output_text):
        output_text = output_text.split('###')[0]
        output_text = output_text.split('Assistant:')[-1].strip()
        output_text = output_text.strip('</s><s>')
        output_text = output_text.strip('</Img>')
        output_text = output_text.strip()
        
        return output_text
