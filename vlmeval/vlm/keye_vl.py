

from transformers import AutoProcessor

from .base import BaseModel

class KeyeVL(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(self, model_path="Kwai-Keye/Keye-VL-1_5-8B", use_vllm=True, **kwargs):

        # check vllm and keye_vl_utils are installed
        if use_vllm:
            try:
                from vllm import LLM, SamplingParams
                from keye_vl_utils import process_vision_info
            except Exception as e:
                raise ImportError(
                    f"vllm and keye_vl_utils are not installed, please install them first, {e}"
                    "You can install them by running: "
                    "pip install keye-vl-utils==1.5.2 vllm>=0.10.2"
                )
        else:
            try:
                from transformers import AutoModel, AutoTokenizer
                from keye_vl_utils import process_vision_info
            except Exception as e:
                raise ImportError(
                    f"transformers and keye_vl_utils are not installed, please install them first, {e}"
                    "You can install them by running: "
                    "pip install keye-vl-utils==1.5.2 transformers>=4.56.1"
                )

        self.use_vllm = use_vllm
        self.fps = 1
        self.max_frames = 64 # 1024
        self.kwargs = kwargs
        # min_pixels = 32 * 28 * 28
        # max_pixels = 1280 * 28 * 28

        self.model_path = model_path
        if use_vllm:
            try:
                # Prefer eager mode to avoid torch.compile tracing of generators in custom model code
                self.llm = LLM(
                    model=model_path,
                    limit_mm_per_prompt={"image": 10, "video": 10},
                    trust_remote_code=True,
                    enforce_eager=True,
                )
            except TypeError:
                # Fallback for older vLLM versions without enforce_eager
                self.llm = LLM(
                    model=model_path,
                    limit_mm_per_prompt={"image": 10, "video": 10},
                    trust_remote_code=True,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.8,
                    max_num_batched_tokens=32768,
                    max_model_len=32768,
                )
            sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=4096,
            )
            self.sampling_params = sampling_params
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype="auto",
                trust_remote_code=True,
                # flash_attention_2 is recommended for better performance
                attn_implementation="flash_attention_2",
            ).eval()
            self.model.to("cuda")
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    def generate_inner_vllm(self, message, dataset=None):
        print(f'{self.model_path} is a video-llm model using vllm, can not set fps or nframe, using default sampling method in keye_vl_utils')
        content_list = []
        for msg in message:
            if msg["type"] == "text":
                content_list.append(
                    {"type": "text", "text": msg["value"]}
                )
            elif msg["type"] == "image":
                content_list.append(
                    {"type": "image", "image": msg["value"]}
                )
            elif msg["type"] == "video":
                content_list.append(
                    {"type": "video", "video": msg["value"]}
                )
            else:
                raise ValueError(f"Invalid message type: {msg['type']}, {msg}")
        conversation = [
            {"role": "user", "content": content_list}
        ]
        prompt = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True,
        )
        from keye_vl_utils import process_vision_info
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversation
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
            # FPS will be returned in video_kwargs
            "mm_processor_kwargs": video_kwargs,
        }

        outputs = self.llm.generate([llm_inputs], sampling_params=self.sampling_params)
        generated_text = outputs[0].outputs[0].text

        return generated_text

    def generate_inner_transformers(self, message, dataset=None):
        content_list = []
        for msg in message:
            if msg["type"] == "text":
                content_list.append(
                    {"type": "text", "text": msg["value"]}
                )
            elif msg["type"] == "image":
                content_list.append(
                    {"type": "image", "image": msg["value"]}
                )
            elif msg["type"] == "video":
                content_list.append(
                    {"type": "video", "video": msg["value"], "fps": self.fps, "max_frames": self.max_frames}
                )
            else:
                raise ValueError(f"Invalid message type: {msg['type']}, {msg}")
        conversation = [
            {"role": "user", "content": content_list}
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        from keye_vl_utils import process_vision_info
        image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(conversation)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **mm_processor_kwargs
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text


    def generate_inner(self, message, dataset=None):
        if self.use_vllm:
            return self.generate_inner_vllm(message, dataset)
        else:
            return self.generate_inner_transformers(message, dataset)


