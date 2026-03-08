import click
import os.path as osp
from PIL import Image


PTH = osp.realpath(__file__)
IMAGE_PTH = osp.join(osp.dirname(PTH), '../../assets/apple.jpg')

msg1 = [
    IMAGE_PTH,
    'What is in this image?'
]
msg2 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='What is in this image?')
]
msg3 = [
    IMAGE_PTH,
    IMAGE_PTH,
    'How many apples are there in these images?'
]
msg4 = [
    dict(type='image', value=IMAGE_PTH),
    dict(type='image', value=IMAGE_PTH),
    dict(type='text', value='How many apples are there in these images?')
]
EDIT_INST = "Double the object number in this image, turn the color to blue, and stack them vertically. Return the generated image. "  # noqa: E501
DESC_INST = "Also describe the generated image in detail."
gen_msg = [
    'Generate an image of two bananas. '
]
edit_msg = [
    IMAGE_PTH,
    EDIT_INST
]
interleave_msg = [
    IMAGE_PTH,
    EDIT_INST + DESC_INST
]


@click.command()
@click.argument('model_name', type=str)
def CHECK(model_name):
    from vlmeval.config import supported_VLM, supported_ULM
    if model_name in supported_VLM or model_name in supported_ULM:
        if model_name in supported_VLM:
            model = supported_VLM[model_name]()
            print(f'Model: {model_name}')
            for i, msg in enumerate([msg1, msg2, msg3, msg4]):
                if i > 1 and not model.INTERLEAVE:
                    continue
                res = model.generate(msg)
                print(f'Test {i + 1}: {res}')
        if model_name in supported_ULM:
            model = supported_ULM[model_name]()
            print(f'Model: {model_name}')
            assert getattr(model, 'SUPPORT_GEN', False), model

            # SUPPORT_GEN means that model supports T2I and TI2I
            def display_response(msg, counter):
                if isinstance(msg, str):
                    return msg
                elif isinstance(msg, Image.Image):
                    fname = f"{model_name}_{counter + 1}.png"
                    msg.save(fname)
                    return fname
                elif isinstance(msg, list):
                    img_counter = 0
                    ret = []
                    for item in msg:
                        if isinstance(item, str):
                            ret.append(item)
                        elif isinstance(item, Image.Image):
                            fname = f"{model_name}_{counter + 1}_{img_counter + 1}.png"
                            item.save(fname)
                            ret.append(fname)
                            img_counter += 1
                    return ret

            for i, msg in enumerate([gen_msg, edit_msg, interleave_msg]):
                if msg is edit_msg and 'TI2I' not in model.EXPERTISE:
                    continue
                if msg is interleave_msg and 'TI2TI' not in model.EXPERTISE:
                    continue
                res = model.generate(msg)
                print(f'Test {i + 1}: {display_response(res, i)}')
    elif len(model_name) == 18:
        # which is a model card id on the merlin seed platform
        # Current ModelCard API are all VLMs for understanding (2025.11.12)
        from vlmeval.api.modelcard_api import ModelCardAPI
        model = ModelCardAPI(model_name, verbose=False)
        model.temperature = 1.0
        print(f'Model: {model_name}')
        for i, msg in enumerate([msg1, msg2, msg3, msg4]):
            if i > 1 and not model.INTERLEAVE:
                continue
            res = model.generate(msg)
            print(f'Test {i + 1}: {res}')
