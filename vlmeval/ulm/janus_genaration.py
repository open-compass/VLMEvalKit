import os
from copy import deepcopy
import requests
from io import BytesIO

import torch
import os.path as osp
import warnings
from .base import BaseGenModel
from ..smp import splitlen, listinstr
from ..dataset import DATASET_TYPE
import PIL.Image

import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

import torchvision
import torchvision.transforms as T

import logging


class JanusGeneration(BaseGenModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    EXPERTISE = ['T2I']

    def __init__(self, model_path='deepseek-ai/Janus-1.3B', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor

        except Exception as err:
            logging.critical(
                'Please first install Janus from source codes in: https://github.com/deepseek-ai/Janus')
            raise err

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).cuda().eval()

        default_kwargs = dict(
            temperature=1,
            cfg_weight=5,
            parallel_size=1,
            image_token_num_per_image=576,
            img_size=384,
            patch_size=16
        )

        default_kwargs.update(kwargs)

        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        def prepare_itlist(msgs):
            content, images = '', []
            for s in msgs:
                if s['type'] == 'image':
                    images.append(s['value'])
                    content += '<image_placeholder>'
                elif s['type'] == 'text':
                    content += s['value']
            return content, images
        conversation = []
        if 'role' not in message[0]:
            content, images = prepare_itlist(message)
            conversation.append(dict(role='User', content=content, images=images))
        else:
            role_map = {'user': 'User', 'assistant': 'Assistant'}
            for msgs in message:
                role = role_map[msgs['role']]
                content, images = prepare_itlist(msgs['content'])
                conversation.append(dict(role=role, content=content, images=images))
        conversation.append(dict(role='Assistant', content=''))

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        return prompt

    def generate_inner(self, message, dataset=None):
        prompt = self.prepare_inputs(message)

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        parallel_size = self.kwargs['parallel_size']
        image_token_num_per_image = self.kwargs['image_token_num_per_image']
        cfg_weight = self.kwargs['cfg_weight']
        temperature = self.kwargs['temperature']
        img_size = self.kwargs['img_size']
        patch_size = self.kwargs['patch_size']

        assert parallel_size == 1, "ULMEvalKit for Janus only supports parallel_size=1 for generation."

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()

        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        outputs = None
        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return PIL.Image.fromarray(visual_img[0])

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt = self.prepare_inputs(message)

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        parallel_size = num_generations
        image_token_num_per_image = self.kwargs['image_token_num_per_image']
        cfg_weight = self.kwargs['cfg_weight']
        temperature = self.kwargs['temperature']
        img_size = self.kwargs['img_size']
        patch_size = self.kwargs['patch_size']

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()

        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        outputs = None
        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return [PIL.Image.fromarray(img) for img in visual_img]


class JanusPro(BaseGenModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='deepseek-ai/Janus-Pro-1B', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        try:
            from janus.models import MultiModalityCausalLM, VLChatProcessor

        except Exception as err:
            logging.critical(
                'Please first install janus from source codes in: https://github.com/deepseek-ai/Janus')
            raise err

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).cuda().eval()

        default_kwargs = dict(
            temperature=1,
            cfg_weight=5,
            parallel_size=1,
            image_token_num_per_image=576,
            img_size=384,
            patch_size=16
        )

        default_kwargs.update(kwargs)

        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        def prepare_itlist(msgs):
            content, images = '', []
            for s in msgs:
                if s['type'] == 'image':
                    images.append(s['value'])
                    content += '<image_placeholder>'
                elif s['type'] == 'text':
                    content += s['value']
            return content, images
        conversation = []
        if 'role' not in message[0]:
            content, images = prepare_itlist(message)
            conversation.append(dict(role='User', content=content, images=images))
        else:
            role_map = {'user': 'User', 'assistant': 'Assistant'}
            for msgs in message:
                role = role_map[msgs['role']]
                content, images = prepare_itlist(msgs['content'])
                conversation.append(dict(role=role, content=content, images=images))
        conversation.append(dict(role='Assistant', content=''))

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_start_tag
        return prompt

    def generate_inner(self, message, dataset=None):
        prompt = self.prepare_inputs(message)

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)
        parallel_size = self.kwargs['parallel_size']
        image_token_num_per_image = self.kwargs['image_token_num_per_image']
        cfg_weight = self.kwargs['cfg_weight']
        temperature = self.kwargs['temperature']
        img_size = self.kwargs['img_size']
        patch_size = self.kwargs['patch_size']

        assert parallel_size == 1, "ULMEvalKit for Janus only supports parallel_size=1 for generation."

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()

        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        outputs = None
        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return PIL.Image.fromarray(visual_img[0])

    def batch_generate_inner(self, message, dataset, num_generations):
        prompt = self.prepare_inputs(message)
        input_ids = self.tokenizer.encode(prompt)

        input_ids = torch.LongTensor(input_ids)
        parallel_size = num_generations
        image_token_num_per_image = self.kwargs['image_token_num_per_image']
        cfg_weight = self.kwargs['cfg_weight']
        temperature = self.kwargs['temperature']
        img_size = self.kwargs['img_size']
        patch_size = self.kwargs['patch_size']

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()

        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        outputs = None
        for i in range(image_token_num_per_image):
            outputs = self.model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if i != 0 else None
            )
            hidden_states = outputs.last_hidden_state

            logits = self.model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]

            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = self.model.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

        dec = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int),
            shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
        )
        dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

        dec = np.clip((dec + 1) / 2 * 255, 0, 255)

        visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        visual_img[:, :, :] = dec

        return [PIL.Image.fromarray(img) for img in visual_img]


class JanusFlow(BaseGenModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path='deepseek-ai/Janus-Flow-1.3B', **kwargs):
        assert osp.exists(model_path) or splitlen(model_path) == 2

        try:
            from janus.janusflow.models import MultiModalityCausalLM, VLChatProcessor
            from diffusers.models import AutoencoderKL

        except Exception as err:
            logging.critical(
                'Please first install janus from source codes in: https://github.com/deepseek-ai/Janus')
            raise err

        print('loading JanusFlow model from:', model_path)
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        self.model = model.to(torch.bfloat16).cuda().eval()

        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
        self.vae = vae.to(torch.bfloat16).cuda().eval()

        default_kwargs = dict(
            cfg_weight=2.0,
            num_inference_steps=30,
            batchsize=5
        )

        default_kwargs.update(kwargs)

        self.kwargs = default_kwargs
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def prepare_inputs(self, message):
        def prepare_itlist(msgs):
            content, images = '', []
            for s in msgs:
                if s['type'] == 'image':
                    images.append(s['value'])
                    content += '<image_placeholder>'
                elif s['type'] == 'text':
                    content += s['value']
            return content, images
        conversation = []
        if 'role' not in message[0]:
            content, images = prepare_itlist(message)
            conversation.append(dict(role='User', content=content, images=images))
        else:
            role_map = {'user': 'User', 'assistant': 'Assistant'}
            for msgs in message:
                role = role_map[msgs['role']]
                content, images = prepare_itlist(msgs['content'])
                conversation.append(dict(role=role, content=content, images=images))
        conversation.append(dict(role='Assistant', content=''))

        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.vl_chat_processor.image_gen_tag
        return prompt

    def generate_inner(self, message, dataset=None):
        prompt = self.prepare_inputs(message)

        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids)

        batchsize = self.kwargs['batchsize']
        cfg_weight = self.kwargs['cfg_weight']
        num_inference_steps = self.kwargs['num_inference_steps']

        tokens = torch.stack([input_ids] * 2 * batchsize).cuda()
        tokens[batchsize:, 1:] = self.vl_chat_processor.pad_id
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        # we remove the last <bog> token and replace it with t_emb later
        inputs_embeds = inputs_embeds[:, :-1, :]

        # generate with rectified flow ode
        # step 1: encode with vision_gen_enc
        z = torch.randn((batchsize, 4, 48, 48), dtype=torch.bfloat16).cuda()

        dt = 1.0 / num_inference_steps
        dt = torch.zeros_like(z).cuda().to(torch.bfloat16) + dt

        # step 2: run ode
        attention_mask = torch.ones((2 * batchsize, inputs_embeds.shape[1] + 577)).to(self.model.device)
        attention_mask[batchsize:, 1:inputs_embeds.shape[1]] = 0
        attention_mask = attention_mask.int()
        for step in range(num_inference_steps):
            # prepare inputs for the llm
            z_input = torch.cat([z, z], dim=0)  # for cfg
            t = step / num_inference_steps * 1000.
            t = torch.tensor([t] * z_input.shape[0]).to(dt)
            z_enc = self.model.vision_gen_enc_model(z_input, t)
            z_emb, t_emb, hs = z_enc[0], z_enc[1], z_enc[2]
            z_emb = z_emb.view(z_emb.shape[0], z_emb.shape[1], -1).permute(0, 2, 1)
            z_emb = self.model.vision_gen_enc_aligner(z_emb)
            llm_emb = torch.cat([inputs_embeds, t_emb.unsqueeze(1), z_emb], dim=1)

            # input to the llm
            # we apply attention mask for CFG: 1 for tokens that are not masked, 0 for tokens that are masked.
            if step == 0:
                outputs = self.model.language_model.model(
                    inputs_embeds=llm_emb,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=None
                )
                past_key_values = outputs.past_key_values
            else:
                outputs = self.model.language_model.model(
                    inputs_embeds=llm_emb,
                    use_cache=True,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.last_hidden_state

            # transform hidden_states back to v
            hidden_states = self.model.vision_gen_dec_aligner(
                self.model.vision_gen_dec_aligner_norm(hidden_states[:, -576:, :])
            )
            hidden_states = hidden_states.reshape(z_emb.shape[0], 24, 24, 768).permute(0, 3, 1, 2)
            v = self.model.vision_gen_dec_model(hidden_states, hs, t_emb)
            v_cond, v_uncond = torch.chunk(v, 2)
            v = cfg_weight * v_cond - (cfg_weight - 1.) * v_uncond
            z = z + dt * v

        # step 3: decode with vision_gen_dec and sdxl vae
        decoded_image = self.vae.decode(z / self.vae.config.scaling_factor).sample

        result = decoded_image.clip_(-1.0, 1.0) * 0.5 + 0.5

        if result.dim() == 4:
            result = result[0]

        result = result.float().clamp(0.0, 1.0)
        to_pil = T.ToPILImage()
        return to_pil(result)
