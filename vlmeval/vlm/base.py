import fitz
from ..smp import *
from ..dataset import img_root_map
from abc import abstractmethod


class BaseModel:

    INTERLEAVE = False
    allowed_types = ['text', 'image']

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

    def dump_image(self, line, dataset, max_pages=1000, concat_num=-1, column_num=-1):
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

        if dataset=='MMLongBench_DOC':
            line['image_path'] = line['image_path'][:max_pages]

            skip_pdf_parse = True
            for im_name in line['image_path']:
                path = osp.join(img_root, im_name)
                if not read_ok(path):
                    skip_pdf_parse = False
                    break

            if skip_pdf_parse:
                line['image'] = line['image_path'] # Just for being compatible with the zooped loop: zip(line['image'], line['image_path'])
            else:
                pdf_data = base64.b64decode(line["image"])
                with open("./tmp.pdf", "wb") as file:
                    file.write(pdf_data)
                encoded_images = list()
                with fitz.open("./tmp.pdf") as doc:
                    doc = doc[:max_pages]
                    for page in doc:
                        image = page.get_pixmap(dpi=144)
                        image.save("./tmp.png")
                        image = Image.open("./tmp.png")
                        encoded_image = encode_image_to_base64(image)
                        encoded_images.append(encoded_image)
                line['image'] = encoded_images
                print("process {}".format(line["doc_id"]))
        
        if 'image' in line:
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
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        if concat_num>0:
            assert(dataset=='MMLongBench_DOC')
            concatenated_images = concat_images(tgt_path, max_concat=concat_num, column_num=column_num)

            old_tgt_path = tgt_path
            assert(isinstance(old_tgt_path, list))
            tgt_path = ["_".join(old_tgt_path[0].split("_")[:-1]) + "_concat{}_{}.jpg".format(concat_num, i) for i in range(len(concatenated_images))]

            for path, concatenated_image in zip(tgt_path, concatenated_images):
                if not read_ok(path):
                    decode_base64_to_image_file(encode_image_to_base64(concatenated_image), path)
                    print("concat {} images to a new one with size {}. save at {}".format(len(old_tgt_path), concatenated_image.size, path))
        return tgt_path

    @abstractmethod
    def generate_inner(self, message, dataset=None):
        raise NotImplementedError

    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text'
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    def generate(self, message, dataset=None):
        """Generate the output message.

        Args:
            message (list[dict]): The input message.
            dataset (str, optional): The name of the dataset. Defaults to None.

        Returns:
            str: The generated message.
        """
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
        return self.generate_inner(message, dataset)

    def message_to_promptimg(self, message):
        assert not self.INTERLEAVE
        model_name = self.__class__.__name__
        warnings.warn(
            f'Model {model_name} does not support interleaved input. '
            'Will use the first image and aggregated texts as prompt. ')
        num_images = len([x for x in message if x['type'] == 'image'])
        if num_images == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = None
        else:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            image = [x['value'] for x in message if x['type'] == 'image'][0]
        return prompt, image
