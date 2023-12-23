import os, torch
from PIL import Image
from ..smp import *
from .utils import CustomPrompt
from ..utils import DATASET_TYPE


class mPLUG_Owl2(CustomPrompt):

    INSTALL_REQ = True

    def __init__(self, model_path='MAGAer13/mplug-owl2-llama2-7b', **kwargs): 
        try:
            from mplug_owl2.model.builder import load_pretrained_model
            from mplug_owl2.mm_utils import get_model_name_from_path
        except:
            warnings.warn('Please install mPLUG_Owl2 before using mPLUG_Owl2. ')
            exit(-1)
            
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cpu")

        self.model = model.cuda()
        self.device = self.model.device
        self.image_processor = image_processor
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer
        self.context_len = context_len

        kwargs_default = dict(
            max_new_tokens=10, do_sample=False, num_beams=1, 
            min_new_tokens=1, length_penalty=1, num_return_sequences=1)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Following kwargs received: {self.kwargs}, will use as generation config. ")

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == 'multi-choice' or dataset == 'MMVet':
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert dataset is None or isinstance(dataset, str)
        assert self.use_custom_prompt(dataset)
        tgt_path = self.dump_image(line, dataset)

        if dataset == 'MMVet':
            prompt_tmpl = "USER: <|image|>{}\nAnswer the question directly. ASSISTANT:"
            prompt = prompt_tmpl.format(line['question'])
        elif DATASET_TYPE(dataset) == 'multi-choice':
            prompt_tmpl = "USER: <|image|>{}\n{}\n{}\nAnswer with the option’s letter from the given choices directly. ASSISTANT:"
            options = {
                cand: line[cand]
                for cand in string.ascii_uppercase
                if cand in line and not pd.isna(line[cand])
            }
            options_prompt = ''
            for key, item in options.items():
                options_prompt += f'{key}. {item}\n'
            
            hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else 'N/A'
            prompt = prompt_tmpl.format(hint, line['question'], options_prompt)
        else:
            raise NotImplementedError

        return {'image': tgt_path, 'text': prompt}
    
    def generate_vanilla(self, image_path, prompt):
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from mplug_owl2.conversation import conv_templates
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token, KeywordsStoppingCriteria
        conv = conv_templates["mplug_owl2"].copy()

        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        inp = DEFAULT_IMAGE_TOKEN + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                **self.kwargs)

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs.split('</s>')[0]

    def generate_multichoice(self, image_path, prompt):
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token
        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids, 
                images=image_tensor, 
                output_hidden_states=True, 
                use_cache=True, 
                **self.kwargs)
        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip()
        return answer.split('</s>')[0]
    
    def generate_mmvet(self, image_path, prompt):
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token
        image = Image.open(image_path).convert('RGB')
        max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        kwargs = cp.deepcopy(self.kwargs)
        kwargs['max_new_tokens'] = 64
        kwargs['length_penalty'] = 0

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids, 
                images=image_tensor, 
                output_hidden_states=True, 
                use_cache=True, 
                **kwargs)
        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip()
        return answer.split('</s>')[0]

    def generate(self, image_path, prompt, dataset=None):
        if dataset is not None and DATASET_TYPE(dataset) == 'multi-choice':
            return self.generate_multichoice(image_path, prompt)
        elif dataset == 'MMVet':
            return self.generate_mmvet(image_path, prompt)
        else:
            return self.generate_vanilla(image_path, prompt)
        
    def multi_generate(self, image_paths, prompt, dataset=None):
        from mplug_owl2.constants import IMAGE_TOKEN_INDEX
        from mplug_owl2.mm_utils import process_images, tokenizer_image_token
        image_prompt = ''
        for i in range(len(image_paths)):
            image_prompt += f'Image {i + 1}: <|image|>; '

        prompt_tmpl = "USER: " + image_prompt + "{}\nASSISTANT: "
        prompt = prompt_tmpl.format(prompt)
        
        images = []
        for pth in image_paths:
            image = Image.open(pth).convert('RGB')
            max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
            image = image.resize((max_edge, max_edge))
            images.append(image)

        image_tensor = process_images(images, self.image_processor)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids, 
                images=image_tensor, 
                output_hidden_states=True, 
                use_cache=True, 
                **self.kwargs)
        answer = self.tokenizer.decode(output_ids[0, input_ids.shape[1]: ]).strip()
        return answer.split('</s>')[0]
    