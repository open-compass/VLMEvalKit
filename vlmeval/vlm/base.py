from ..smp import *
from ..utils.dataset_config import img_root_map
from abc import abstractmethod


class BaseModel:

    interleave_input = False

    def use_custom_prompt(self, dataset):
        """Whether to use custom prompt for the given dataset.

        Args:
            dataset (str): The name of the dataset.

        Returns:
            bool: Whether to use custom prompt. If True, will call `build_prompt` of the VLM to build the prompt.
                Default to False.
        """
        return False

    @abstractmethod
    def build_prompt(self, line, dataset):
        """Build custom prompts for a specific dataset. Called only if `use_custom_prompt` returns True.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str: The built message.
        """
        raise NotImplementedError

    def dump_image(self, line, dataset):
        """Dump the image(s) of the input line to the corresponding dataset folder.

        Args:
            line (line of pd.DataFrame): The raw input line.
            dataset (str): The name of the dataset.

        Returns:
            str | list[str]: The paths of the dumped images.
        """
        ROOT = LMUDataRoot()
        assert isinstance(dataset, str)
        img_root = osp.join(ROOT, 'images', img_root_map[dataset] if dataset in img_root_map else dataset)
        os.makedirs(img_root, exist_ok=True)
        if isinstance(line['image'], list):
            tgt_path = []
            assert 'image_path' in line
            for img, im_name in zip(line['image'], line['image_path']):
                path = osp.join(img_root, im_name)
                if not read_ok(path):
                    decode_base64_to_image_file(img, path)
                tgt_path.append(path)
        else:
            tgt_path = osp.join(img_root, f"{line['index']}.jpg")
            if not read_ok(tgt_path):
                decode_base64_to_image_file(line['image'], tgt_path)
            tgt_path = [tgt_path]
        return tgt_path

    @abstractmethod
    def generate(self, message, dataset=None):
        """Generate the output message.

        Args:
            message (list[dict]): The input message.
            dataset (str, optional): The name of the dataset. Defaults to None.

        Returns:
            str: The generated message.
        """
        raise NotImplementedError

    def message_to_promptimg(self, message):
        assert not self.interleave_input
        model_name = self.__class__.__name__
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 1:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        else:
            prompt = '\n'.join([x['value'] if x['type'] == 'text' else '<image>' for x in message])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image
