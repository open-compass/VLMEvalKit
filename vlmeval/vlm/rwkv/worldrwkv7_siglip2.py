from __future__ import annotations

import os
import sys
import warnings
from PIL import Image

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin

# Add the WorldRWKV root directory to Python path
worldrwkv_root = '/home/lynn/rwkv/WorldRWKV'
sys.path.insert(0, worldrwkv_root)

# Import WorldRWKV specific modules
from infer.worldmodel import Worldinfer
from infer.rwkv.utils import PIPELINE_ARGS


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


class WorldRWKV7_Siglip2(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = False

    def __init__(
        self,
        model_path: str,
        encoder_path: str = None,
        max_new_tokens=2048,
        top_p=0.0,
        top_k=0,
        temperature=1.0,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,
        verbose: bool = False,
        strategy: str = 'cuda bf16',
        args: dict | None = None
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)

        self.system_prompt = system_prompt
        self.verbose = verbose

        
        # Initialize model paths
        assert model_path is not None, "Model path must be provided"
        self.model_path = model_path
        
        # If encoder_path is not provided, try to infer it
        if encoder_path is None:
            # Default to a common path if not specified
            encoder_path = 'google/siglip2-base-patch16-384'
            warnings.warn(f"Encoder path not specified, using default: {encoder_path}")
        
        # Convert args dictionary to PIPELINE_ARGS if it's a dict
        if isinstance(args, dict):
            args = PIPELINE_ARGS(
                temperature=args.get('temperature', 1.0),
                top_p=args.get('top_p', 0.85),
                top_k=args.get('top_k', 0),
                alpha_frequency=args.get('alpha_frequency', 0.2),
                alpha_presence=args.get('alpha_presence', 0.2),
                alpha_decay=args.get('alpha_decay', 0.996),
                token_ban=args.get('token_ban', []),
                token_stop=args.get('token_stop', []),
                chunk_len=args.get('chunk_len', 256)
            )
            
        # Initialize the WorldRWKV model
        self.model = Worldinfer(
            model_path=model_path,
            encoder_type="siglip",
            encoder_path=encoder_path,
            strategy=strategy,
            args=args
        )
        
        if self.verbose:
            print(f"Initialized WorldRWKV7_Siglip2 with model_path={model_path}, encoder_path={encoder_path}")

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> tuple[str, Image.Image]:
        """
        Process inputs to extract text and image for WorldRWKV model.
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        Returns a tuple of (text, image)
        """
        text_parts = []
        image = None
        
        for s in inputs:
            if s['type'] == 'image':
                # Get the image path and load it
                image_path = s['value']
                if image_path.startswith('file://'):
                    image_path = image_path[7:]  # Remove file:// prefix
                
                try:
                    image = Image.open(image_path).convert('RGB')
                    if self.verbose:
                        print(f"Loaded image from {image_path}")
                except Exception as e:
                    warnings.warn(f"Failed to load image {image_path}: {e}")
            
            elif s['type'] == 'text':
                text_parts.append(s['value'])
            
            elif s['type'] == 'video':
                warnings.warn("Video input is not supported by WorldRWKV7_Siglip2")
            
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
        
        # Join text parts and ensure they have proper formatting
        formatted_text = ' '.join(text_parts)
        
        # If the text doesn't already have special tokens, add them
        if not formatted_text.startswith('\x16'):
            formatted_text = f'\x16User: {formatted_text}\x17Assistant:'
        
        return formatted_text, image

    def generate_inner(self, message, dataset=None):
        # Process the input message to get text and image
        text, image = self._prepare_content(message, dataset=dataset)
        
        if self.verbose:
            print(f"Input text: {text}")
            if image:
                print(f"Image provided with size: {image.size}")
        
        # If system prompt is provided, prepend it to the text
        if self.system_prompt is not None:
            # Format system prompt according to WorldRWKV's expected format
            # Only add system prompt if it's not already included
            if '\x16System:' not in text:
                text = f"\x16System: {self.system_prompt}\x17 {text}"
        
        # Generate response using the WorldRWKV model
        result, _ = self.model.generate(text, image)
        
        if self.verbose:
            print(f"Raw model output: {result}")
        
        # # Post-process the result if needed
        # if self.post_process:
        #     # Extract content from boxed notation if present
        #     if '\boxed{' in result:
        #         resp = result.split('\boxed{')[-1]
        #         lt = len(resp)
        #         counter, end = 1, None
        #         for i in range(lt):
        #             if resp[i] == '{':
        #                 counter += 1
        #             elif resp[i] == '}':
        #                 counter -= 1
        #             if counter == 0:
        #                 end = i
        #                 break
        #             elif i == lt - 1:
        #                 end = lt
        #                 break
        #         if end is not None:
        #             result = resp[:end]
            
        #     # Clean up any trailing special tokens that might have been generated
        #     if '\x16' in result:
        #         result = result.split('\x16')[0].strip()
        
        return result
