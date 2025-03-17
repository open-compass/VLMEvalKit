from __future__ import annotations


class Qwen2VLPromptMixin:
    """
    Mixin class for WorldRWKV to build custom prompt for different datasets.
    
    This class formats prompts using WorldRWKV's special tokens:
    - \x16User: ... \x17Assistant: for user/assistant exchanges
    - \x16System: ... \x17 for system prompts

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST', 'MMMU_PRO', 'CMMMU'}:
            return True
        if dataset_type == 'MCQ':
            return True
        if dataset_type == 'Y/N' and dataset not in {'MME'}:  # MME has it's own prompt
            return True
        if dataset_type == 'VQA' and dataset not in {'MMVet'}:  # MMVet VQA has it's own prompt
            return True
        if dataset_type == 'CAPTION':
            return True
        return False

    def build_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        from vlmeval.dataset import DATASET_TYPE

        if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST', 'MMMU_PRO', 'CMMMU'}:
            return self._build_mmmu_prompt(line, dataset)
            
        # Handle image datasets by type
        dataset_type = DATASET_TYPE(dataset, default=None)
        if dataset_type == 'MCQ':
            return self._build_mcq_prompt(line, dataset)
        if dataset_type == 'Y/N':
            return self._build_yorn_prompt(line, dataset)
        if dataset_type == 'VQA':
            return self._build_vqa_prompt(line, dataset)
        if dataset_type == 'CAPTION':
            return self._build_caption_prompt(line, dataset)
            
        raise ValueError(f'Unsupported dataset: {dataset}')

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Format prompt for MMMU dataset using WorldRWKV special tokens."""

        import string
        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above.'
        prompt = prompt.rstrip()
        
        # Format with WorldRWKV special tokens
        formatted_prompt = f'\x16User: {prompt}\x17Assistant:'
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=formatted_prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Format prompt for MCQ dataset using WorldRWKV special tokens."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string
        import pandas as pd

        def cn_string(s):
            import re
            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        
        # Format with WorldRWKV special tokens
        formatted_prompt = f'\x16User: {prompt}\x17Assistant:'
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=formatted_prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Format prompt for YORN dataset using WorldRWKV special tokens."""
        YORN_PROMPT = ' Please answer yes or no.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question'] + YORN_PROMPT
        
        # Format with WorldRWKV special tokens
        formatted_prompt = f'\x16User: {question}\x17Assistant:'
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=formatted_prompt))
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Format prompt for VQA dataset using WorldRWKV special tokens."""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question'] + VQA_PROMPT
        
        # Format with WorldRWKV special tokens
        formatted_prompt = f'\x16User: {question}\x17Assistant:'
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=formatted_prompt))
        return msgs
        
    def _build_caption_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """Format prompt for image captioning datasets using WorldRWKV special tokens."""
        CAPTION_PROMPT = 'Please describe this image in detail.'
        
        tgt_path = self.dump_image(line, dataset)
        
        # Format with WorldRWKV special tokens
        formatted_prompt = f'\x16User: {CAPTION_PROMPT}\x17Assistant:'
        
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=formatted_prompt))
        return msgs
        
