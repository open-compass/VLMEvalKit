import sys
from vlmeval.smp import *
from vlmeval.config import transformer_ver, supported_VLM
PTH = osp.realpath(__file__)
IMAGE_PTH = osp.join(osp.dirname(PTH), '../assets/apple.jpg')

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

def check_VLM():
    val = sys.argv[1]
    if len(sys.argv) > 2:
        for m in sys.argv[1:]:
            try:
                res = run_command(f'python {PTH} {m}')
                print(res)
            except:
                pass
    elif val in supported_VLM:
        model = supported_VLM[val]()
        print(f'Model: {val}')
        for i, msg in enumerate([msg1, msg2, msg3, msg4]):
            if i > 1 and not model.INTERLEAVE:
                continue
            res = model.generate(msg)
            print(f'Test {i + 1}: {res}')
    elif val in transformer_ver:
        models = transformer_ver[val]
        for m in models:
            try:
                res = run_command(f'python {PTH} {m}')
                print(res)
            except:
                pass

if __name__ == '__main__':
    check_VLM()