import argparse
import torch

class Sphinx:
    
    def __init__(self, **kwargs):
        from LLaMA2_Accessory.accessory.model.meta import MetaModel
        from fairscale.nn.model_parallel import initialize as fs_init
        from LLaMA2_Accessory.accessory.util.tensor_parallel import load_tensor_parallel_model_list
        from LLaMA2_Accessory.accessory.util.tensor_type import default_tensor_type
        from LLaMA2_Accessory.accessory.util import misc
        from LLaMA2_Accessory.accessory.model.meta import MetaModel
        from LLaMA2_Accessory.accessory.data.conversation.lib import conv_templates, SeparatorStyle
        from LLaMA2_Accessory.accessory.data.transform import get_transform
        from PIL import Image

        # self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.get_transform = get_transform
        # import pdb;pdb.set_trace()
        sphinx_parser = argparse.ArgumentParser('simple sphinx demo', add_help=False)
        sphinx_parser.add_argument('--dist_on_itp', action='store_true')
        sphinx_parser.add_argument('--data', type=str, nargs='+', required=True)
        sphinx_parser.add_argument("--model", type=str, default=None)
        args = sphinx_parser.parse_args()
        # define the model
        # import pdb;pdb.set_trace()
        # misc.init_distributed_mode(args)
        fs_init.initialize_model_parallel(1)
        self.model_path = ['/mnt/petrelfs/share_data/duanhaodong/sphinx-sft']
        self.tokenizer_path = "/mnt/petrelfs/share_data/duanhaodong/llama2/tokenizer.model"
        self.llama_ens = []
        self.target_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }["bf16"]
        with default_tensor_type(dtype=self.target_dtype, device="cuda"):
            self.model = model = MetaModel(
                "llama_ens", self.llama_ens, self.tokenizer_path,
                with_visual=True, max_seq_len=2048,
            )
        # import pdb;pdb.set_trace()
        print(f"load pretrained from {self.model_path}")
        load_result = load_tensor_parallel_model_list(self.model, self.model_path)
        print("load result: ", load_result)
        self.model.eval()
        print("Model = %s" % str(self.model))
        self.model.bfloat16().cuda()
        # import pdb;pdb.set_trace()

    @staticmethod
    def get_formatted_prompt(prompt: str) -> str:
        from LLaMA2_Accessory.accessory.data.conversation.lib import conv_templates
        conv = conv_templates["v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv

    def generate(self, image_path, prompt):
       
        image = Image.open(image_path)
        image = self.get_transform("resized_center_crop")(image).unsqueeze(0).cuda()
        conv = self.get_formatted_prompt(prompt)
        max_gen_len, temperature, top_p = 512, 0.1, 0.1
        with torch.cuda.amp.autocast(dtype=self.target_dtype):
            # print(conv.get_prompt())
            for stream_response in self.model.stream_generate(
                    conv.get_prompt(), image,
                    max_gen_len, temperature, top_p
            ):
                conv_sep = (
                    conv.sep
                    if conv.sep_style == self.SeparatorStyle.SINGLE
                    else conv.sep2
                )
                end_pos = stream_response["text"].find(conv_sep)
                if end_pos != -1:
                    stream_response["text"] = (
                            stream_response['text'][:end_pos].rstrip() + "\n"
                    )
                    stream_response["end_of_content"] = True

                if stream_response["end_of_content"]:
                    print(stream_response["text"])
                    break
        return stream_response["text"]