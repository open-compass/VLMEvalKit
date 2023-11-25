import torch
from PIL import Image

class OtterI:
    
    def __init__(self, **kwargs):
        import transformers
        from otter_ai import OtterForConditionalGeneration
        self.model_path = '/mnt/petrelfs/share_data/duanhaodong/OTTER-Image-MPT7B'
        precision = {'torch_dtype': torch.bfloat16}
        model = OtterForConditionalGeneration.from_pretrained(self.model_path, device_map='cpu', **precision)
        model.text_tokenizer.padding_side = "left"
        self.tokenizer = model.text_tokenizer
        self.image_processor = transformers.CLIPImageProcessor()
        model.eval()
        self.model = model.cuda()

    @staticmethod
    def get_formatted_prompt(prompt: str) -> str:
        return f"<image>User: {prompt} GPT:<answer>"

    def generate(self, image_path, prompt):
        image = Image.open(image_path)
        vision_x = self.image_processor.preprocess([image], return_tensors='pt')['pixel_values'].unsqueeze(1).unsqueeze(0)
        lang_x = self.tokenizer([self.get_formatted_prompt(prompt)], return_tensors='pt')
        model_dtype = next(self.model.parameters()).dtype
        vision_x = vision_x.to(dtype=model_dtype)
        lang_x_input_ids = lang_x["input_ids"]
        lang_x_attention_mask = lang_x["attention_mask"]

        generated_text = self.model.generate(
            vision_x=vision_x.to(self.model.device),
            lang_x=lang_x_input_ids.to(self.model.device),
            attention_mask=lang_x_attention_mask.to(self.model.device),
            max_new_tokens=512,
            num_beams=3,
            no_repeat_ngram_size=3)
        parsed_output = (
            self.tokenizer.decode(generated_text[0]).split("<answer>")[-1]
            .lstrip().rstrip().split("<|endofchunk|>")[0]
            .lstrip().rstrip().lstrip('"').rstrip('"')
        )
        return parsed_output