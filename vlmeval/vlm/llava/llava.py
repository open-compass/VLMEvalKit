import torch
from PIL import Image
from abc import abstractproperty
import sys
import os.path as osp
from ..base import BaseModel
from ...smp import *
from ...dataset import DATASET_TYPE
import copy


class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self,
                 model_path='liuhaotian/llava_v1.5_7b',
                 **kwargs):
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install llava before using LLaVA')
            sys.exit(-1)

        warnings.warn('Please install the latest version of llava from github before you evaluate the LLaVA model. ')
        assert osp.exists(model_path) or splitlen(model_path) == 2
        self.system_prompt = (
            'A chat between a curious human and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = '</s>'

        if model_path == 'Lin-Chen/ShareGPT4V-7B':
            model_name = 'llava-v1.5-7b'
        elif model_path == 'Lin-Chen/ShareGPT4V-13B':
            model_name = 'llava-v1.5-13b'
        else:
            model_name = get_model_name_from_path(model_path)

        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                device='cpu',
                device_map='cpu'
            )
        except:
            if 'ShareGPT4V' in model_path:
                import llava
                warnings.warn(
                    'Please manually remove the encoder type check in '
                    f'{llava.__path__[0]}/model/multimodal_encoder/builder.py '
                    'Line 8 to use the ShareGPT4V model. ')
            else:
                warnings.warn('Unknown error when loading LLaVA model.')
            exit(-1)

        self.model = self.model.cuda()
        self.conv_mode = 'llava_v1'

        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1, use_cache=True) # noqa E501
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def concat_tilist(self, message):
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])
        return text, images

    def chat_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        prompt = self.system_prompt
        images = []
        for utter in message:
            prompt += 'USER: ' if utter['role'] == 'user' else 'ASSISTANT: '
            content, images_sub = self.concat_tilist(utter['content'])
            prompt += content
            images.extend(images_sub)
            prompt += ' ' if utter['role'] == 'user' else self.stop_str
        assert message[-1]['role'] == 'user', message
        prompt += 'ASSISTANT: '

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)
        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output

    def generate_inner(self, message, dataset=None):
        from llava.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX

        # Support interleave text and image
        content, images = self.concat_tilist(message)

        images = [Image.open(s).convert('RGB') for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = 'pad'
        if images:
            image_tensor = process_images(images, self.image_processor, args).to('cuda', dtype=torch.float16)
        else:
            image_tensor = None

        prompt = self.system_prompt + 'USER: ' + content + ' ASSISTANT: '

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        keywords = [self.stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=image_tensor, stopping_criteria=[stopping_criteria], **self.kwargs)

        output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return output


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='llava-hf/llava-v1.6-vicuna-7b-hf', **kwargs):
        import transformers
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, \
            AutoProcessor, LlavaForConditionalGeneration
        self.model_path = model_path
        if '34b' in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path, use_fast=False)
        elif 'interleave' in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        flash_attn_flag = False
        try:
            import flash_attn
            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            if 'interleave' in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True)
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_flash_attention_2=True)
        else:
            if 'interleave' in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=512, top_p=None, num_beams=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    def apply_prompt_template(self, prompt):
        model_path = self.model_path.lower()
        if 'mistral' in model_path:
            template = '[INST] PLACEHOLDER [/INST]'
        elif 'vicuna' in model_path:
            template = (
                'A chat between a curious human and an artificial intelligence assistant. '
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                'USER: PLACEHOLDER ASSISTANT:'
            )
        elif '34b' in model_path:
            template = (
                '<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nPLACEHOLDER<|im_end|>'
                '<|im_start|>assistant\n'
            )
        else:
            raise NotImplementedError(f'Prompt template for {model_path} not implemented.')

        prompt = template.replace('PLACEHOLDER', f'<image>\n{prompt}')
        return prompt

    def output_process(self, answer):
        if '<s>' in answer:
            answer = answer.replace('<s>', '').strip()
        if '[/INST]' in answer:
            answer = answer.split('[/INST]')[1].strip()
        elif 'ASSISTANT:' in answer:
            answer = answer.split('ASSISTANT:')[1].strip()
        elif 'assistant\n' in answer:
            answer = answer.split('assistant\n')[1].strip()
        elif '<|end_header_id|>\n\n' in answer:
            answer = answer.split('<|end_header_id|>\n\n')[2].strip()

        if '</s>' in answer:
            answer = answer.split('</s>')[0].strip()
        elif '<|im_end|>' in answer:
            answer = answer.split('<|im_end|>')[0].strip()
        elif '<|eot_id|>' in answer:
            answer = answer.split('<|eot_id|>')[0].strip()
        return answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'MCQ':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += (
                '\n请直接回答选项字母。' if cn_string(prompt) else
                "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'
        message = [dict(type='image', value=s) for s in tgt_path]
        message.append(dict(type='text', value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        for msg in message:
            if msg['type'] == 'text':
                content.append({'type': msg['type'], 'text': msg['value']})
            else:
                content.append({'type': 'image'})
                images.append(Image.open(msg['value']).convert('RGB'))
        conversation = [
            {
                'role': 'user',
                'content': content,
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(prompt, images, return_tensors='pt').to('cuda', torch.float16)
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer)
        return answer


class LLaVA_Next2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='lmms-lab/llama3-llava-next-8b', **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
        except:
            warnings.warn('Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`')

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map=None)
        model.cuda().eval()
        model.tie_weights()

        if 'llama3' in model_path.lower():
            conv_mode = 'llava_llama_3'
        elif 'qwen' in model_path.lower():
            conv_mode = 'qwen_1_5'
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token

    def generate_inner(self, message, dataset=None):
        content, images = '', []
        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                images.append(Image.open(msg['value']).convert('RGB'))
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors='pt')['pixel_values'][0].half().cuda() for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = image_tokenizer(prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = '<image>'
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path='lmms-lab/llava-onevision-qwen2-7b-si', **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
            from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        except ImportError:
            warnings.warn('Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`')

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name, device_map=None)
        model.cuda().eval()
        model.tie_weights()

        if 'llava' in model_path.lower():
            conv_mode = 'qwen_1_5'
        self.nframe = 16
        if '72b' in model_path.lower():
            self.nframe = 32
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = process_images  # Store process_images as a class attribute

    def generate_inner_image(self, message, dataset=None):
        content, images = '', []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                img = Image.open(msg['value']).convert('RGB')
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(images, self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device='cuda') for _image in image_tensor]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt_question,
                                               self.tokenizer,
                                               self.IMAGE_TOKEN_INDEX,
                                               return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_video(self, message, dataset=None):
        content, videos = '', []

        for msg in message:
            if msg['type'] == 'text':
                content += msg['value']
            else:
                videos.append(msg['value'])
                content += (self.DEFAULT_IMAGE_TOKEN + '\n')

        if len(videos) > 1:
            raise ValueError('LLaVA-OneVision does not support multiple videos as input.')
        video_frames = self.load_video(videos[0], self.nframe)
        image_tensors = []
        frames = self.image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values'].half().cuda()
        image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(prompt_question,
                                               self.tokenizer,
                                               self.IMAGE_TOKEN_INDEX,
                                               return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities = ['video'] * len(video_frames)

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=512,
            modalities=modalities
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def load_video(self, video_path, max_frames_num):
        from decord import VideoReader, cpu
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames  # (frames, height, width, channels)

    def generate_inner(self, message, dataset=None):
        if dataset in ['MMBench-Video', 'Video-MME', 'MVBench']:
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)
