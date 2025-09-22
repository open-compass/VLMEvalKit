from ..base import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


class VideoLLaMA3(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False
    VIDEO_LLM = True

    def __init__(self, model_path="DAMO-NLP-SG/VideoLLaMA3-7B", **kwargs):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map={"": "cuda"},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.kwargs = kwargs
        self.fps = 1
        self.max_frames = 64

    def generate_inner(self, message, dataset=None):
        content_list = []
        for msg in message:
            if msg["type"] == "text":
                content_list.append({"type": "text", "text": msg["value"]})
            elif msg["type"] == "video":
                content_list.append(
                    {"type": "video", "video": {"video_path": msg["value"], "fps": self.fps, "max_frames": self.max_frames}}
                )
            elif msg["type"] == "image":
                content_list.append({"type": "image", "image": {"image_path": msg["value"]}})
            else:
                raise ValueError(f"Invalid message type: {msg['type']}, {msg}")
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content_list}
        ]
        
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self.model.generate(**inputs, **self.kwargs)
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return response